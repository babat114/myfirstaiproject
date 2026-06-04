"""
认证辅助函数
提供 API 和 Web 路由共用的用户获取逻辑
"""
from flask import request
from flask_login import current_user
from app.services.auth_service import AuthService


def get_current_user():
    """
    从请求上下文中获取当前用户

    支持三种认证方式 (优先级从高到低):
    1. Session 认证 (Web 页面 — Flask-Login)
    2. JWT Bearer Token (API — Authorization: Bearer <token>)
    3. API Key 认证 (API — X-API-Key header)

    Returns:
        User 或 None
    """
    # 方式1: Session 认证 (Web 页面)
    if current_user.is_authenticated:
        return current_user

    # 方式2: JWT Bearer Token (API 认证)
    from app.utils.jwt_helpers import get_user_from_jwt
    user = get_user_from_jwt()
    if user:
        return user

    # 方式3: API Key 认证 (兼容旧版)
    api_key = request.headers.get('X-API-Key')
    if api_key:
        return AuthService.get_user_by_api_key(api_key)

    return None
