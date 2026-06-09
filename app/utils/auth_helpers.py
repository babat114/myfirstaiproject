"""
认证辅助函数
提供 API 和 Web 路由共用的用户获取逻辑
"""
from flask import request, g
from flask_login import current_user
from app.services.auth_service import AuthService


def get_current_user():
    """
    从请求上下文中获取当前用户

    支持三种认证方式 (优先级从高到低):
    0. g.current_user 缓存 (由 api_login_required 装饰器设置)
    1. Session 认证 (Web 页面 — Flask-Login)
    2. JWT Bearer Token (API — Authorization: Bearer <token>)
    3. API Key 认证 (API — X-API-Key header)

    Returns:
        User 或 None
    """
    # 方式0: 从装饰器缓存的 g.current_user (避免重复认证)
    cached = getattr(g, 'current_user', None)
    if cached:
        return cached

    # 方式1: Session 认证 (Web 页面)
    if current_user.is_authenticated:
        g.current_user = current_user
        return current_user

    # 方式2: JWT Bearer Token (API 认证)
    from app.utils.jwt_helpers import get_user_from_jwt
    user = get_user_from_jwt()
    if user:
        g.current_user = user
        return user

    # 方式3: API Key 认证 (兼容旧版)
    api_key = request.headers.get('X-API-Key')
    if api_key:
        user = AuthService.get_user_by_api_key(api_key)
        if user:
            g.current_user = user
            return user

    return None
