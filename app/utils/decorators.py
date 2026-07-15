"""
============================================
自定义装饰器
认证、权限检查、API限流等装饰器
============================================
"""

import time
from functools import wraps

from flask import g, jsonify, request
from flask_login import current_user

from app.services.auth_service import AuthService


def api_login_required(func):
    """
    API 认证装饰器
    支持三种认证方式: Session、JWT Bearer Token、API Key

    认证后将 user 存入 g.current_user，避免 get_current_user() 重复验证
    """

    @wraps(func)
    def decorated(*args, **kwargs):
        from app.utils.jwt_helpers import get_user_from_jwt

        # 方式1: Session 认证
        if current_user.is_authenticated:
            g.current_user = current_user
            return func(*args, **kwargs)

        # 方式2: JWT Bearer Token
        user = get_user_from_jwt()
        if user:
            g.current_user = user
            return func(*args, **kwargs)

        # 方式3: API Key 认证 (兼容旧版)
        api_key = request.headers.get('X-API-Key')
        if api_key:
            user = AuthService.get_user_by_api_key(api_key)
            if user:
                g.current_user = user
                return func(*args, **kwargs)

        return jsonify(
            {
                'success': False,
                'message': '认证失败。请提供有效的 Bearer Token、API Key 或登录。',
            }
        ), 401

    return decorated


def api_admin_required(func):
    """
    API 管理员权限装饰器
    支持三种认证方式: Session、JWT Bearer Token、API Key
    """

    @wraps(func)
    def decorated(*args, **kwargs):
        from app.utils.jwt_helpers import get_user_from_jwt

        # 方式1: Session 用户
        if current_user.is_authenticated:
            if not current_user.is_admin:
                return jsonify({'success': False, 'message': '需要管理员权限。'}), 403
            g.current_user = current_user
            return func(*args, **kwargs)

        # 方式2: JWT Bearer Token
        user = get_user_from_jwt()
        if user:
            if not user.is_admin:
                return jsonify({'success': False, 'message': '需要管理员权限。'}), 403
            g.current_user = user
            return func(*args, **kwargs)

        # 方式3: API Key 用户
        api_key = request.headers.get('X-API-Key')
        if api_key:
            user = AuthService.get_user_by_api_key(api_key)
            if user:
                if not user.is_admin:
                    return jsonify({'success': False, 'message': '需要管理员权限。'}), 403
                g.current_user = user
                return func(*args, **kwargs)

        return jsonify(
            {
                'success': False,
                'message': '认证失败。请提供有效的凭据。',
            }
        ), 401

    return decorated


def rate_limit(max_calls: int = 60, period: int = 60):
    """
    简单的 API 限流装饰器 (基于内存, 线程安全)

    Args:
        max_calls: 在周期内允许的最大请求数
        period: 限流周期 (秒)

    注意: 测试环境 (TESTING=True) 自动跳过限流。
          每 100 次请求触发一次惰性清理, 移除超过 2*period 未活动的 IP 记录。
    """
    import threading

    _store = {}
    _lock = threading.Lock()
    _cleanup_counter = [0]  # 可变容器: 请求计数器, 每100次触发清理

    def _cleanup_expired(now_ts: float, horizon: float):
        """移除超过 horizon 秒未活动的 IP 记录"""
        stale = [ip for ip, calls in _store.items() if not calls or max(calls) < now_ts - horizon]
        for ip in stale:
            del _store[ip]
        if stale:
            # 安全获取 logger (不在应用上下文时静默跳过)
            try:
                from flask import current_app

                current_app.logger.debug(f'rate_limit: 清理 {len(stale)} 个过期 IP 记录')
            except RuntimeError:
                pass

    def decorator(func):
        @wraps(func)
        def decorated(*args, **kwargs):
            # 测试环境跳过限流 (避免跨测试状态累积)
            try:
                from flask import current_app

                if current_app.config.get('TESTING'):
                    return func(*args, **kwargs)
            except RuntimeError:
                pass

            identifier = request.remote_addr or 'unknown'
            now = time.time()
            window_start = now - period

            exceeded = False
            retry_after = 0

            with _lock:
                calls = _store.get(identifier, [])
                calls = [t for t in calls if t > window_start]
                _store[identifier] = calls

                # 惰性清理: 每 100 次请求清理超过 2*period 未活动的记录
                _cleanup_counter[0] += 1
                if _cleanup_counter[0] >= 100:
                    _cleanup_counter[0] = 0
                    _cleanup_expired(now, period * 2)

                # 原子检查+追加, 避免锁外门控的竞态条件
                if len(calls) >= max_calls:
                    retry_after = int(calls[0] + period - now) + 1
                    exceeded = True
                else:
                    calls.append(now)
                    _store[identifier] = calls

            if exceeded:
                # API 请求返回 JSON, Web 请求返回 HTML 错误页
                if request.path.startswith('/api/'):
                    return jsonify(
                        {
                            'success': False,
                            'message': f'请求过于频繁，请在 {retry_after} 秒后重试。',
                            'retry_after': retry_after,
                        }
                    ), 429
                else:
                    from flask import flash, redirect

                    flash(f'请求过于频繁，请在 {retry_after} 秒后重试。', 'warning')
                    return redirect(request.referrer or request.url)

            return func(*args, **kwargs)

        return decorated

    return decorator


def log_execution_time(func):
    """
    记录函数执行时间的装饰器
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        from flask import current_app

        current_app.logger.debug(f'{func.__name__} 执行耗时: {elapsed:.4f} 秒')
        return result

    return wrapper
