"""
============================================
自定义装饰器
认证、权限检查、API限流等装饰器
============================================
"""
import time
from functools import wraps
from flask import request, jsonify
from flask_login import current_user
from app.services.auth_service import AuthService


def api_login_required(func):
    """
    API 认证装饰器
    支持 Session 认证和 API Key 认证两种方式
    """
    @wraps(func)
    def decorated(*args, **kwargs):
        # 方式1: Session 认证
        if current_user.is_authenticated:
            return func(*args, **kwargs)

        # 方式2: API Key 认证
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if api_key:
            user = AuthService.get_user_by_api_key(api_key)
            if user:
                return func(*args, **kwargs)

        return jsonify({
            'success': False,
            'message': '认证失败。请提供有效的 API Key 或登录。',
        }), 401

    return decorated


def api_admin_required(func):
    """
    API 管理员权限装饰器
    要求请求者具有管理员角色
    """
    @wraps(func)
    def decorated(*args, **kwargs):
        # 检查 Session 用户
        if current_user.is_authenticated and current_user.is_admin:
            return func(*args, **kwargs)

        # 检查 API Key 用户
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if api_key:
            user = AuthService.get_user_by_api_key(api_key)
            if user and user.is_admin:
                return func(*args, **kwargs)

        return jsonify({
            'success': False,
            'message': '需要管理员权限。',
        }), 403

    return decorated


def rate_limit(max_calls: int = 60, period: int = 60):
    """
    简单的 API 限流装饰器 (基于内存)

    Args:
        max_calls: 在周期内允许的最大请求数
        period: 限流周期 (秒)
    """
    # 内存存储: { key: [timestamp, ...] }
    _store = {}

    def decorator(func):
        @wraps(func)
        def decorated(*args, **kwargs):
            # 获取客户端标识
            identifier = request.remote_addr or 'unknown'

            now = time.time()
            window_start = now - period

            # 获取并清理过期记录
            calls = _store.get(identifier, [])
            calls = [t for t in calls if t > window_start]
            _store[identifier] = calls

            if len(calls) >= max_calls:
                retry_after = int(calls[0] + period - now) + 1
                return jsonify({
                    'success': False,
                    'message': f'请求过于频繁，请在 {retry_after} 秒后重试。',
                    'retry_after': retry_after,
                }), 429

            calls.append(now)
            _store[identifier] = calls

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
        current_app.logger.debug(
            f'{func.__name__} 执行耗时: {elapsed:.4f} 秒'
        )
        return result
    return wrapper
