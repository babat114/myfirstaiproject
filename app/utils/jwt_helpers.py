"""
============================================
JWT 认证工具
生成、验证、刷新 JSON Web Token
============================================
"""
import jwt
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any
from flask import current_app
from app import logger


def _get_jwt_secret() -> str:
    """获取 JWT 签名密钥"""
    return current_app.config.get('JWT_SECRET_KEY', 'jwt-secret-key')


def _get_access_expiry() -> timedelta:
    """获取 Access Token 过期时间"""
    return current_app.config.get('JWT_ACCESS_TOKEN_EXPIRES', timedelta(hours=2))


def _get_refresh_expiry() -> timedelta:
    """获取 Refresh Token 过期时间"""
    return current_app.config.get('JWT_REFRESH_TOKEN_EXPIRES', timedelta(days=30))


def generate_access_token(user_id: int, username: str, role: str) -> str:
    """
    生成 JWT Access Token (短有效期)

    Args:
        user_id: 用户ID
        username: 用户名
        role: 用户角色 (admin / researcher / viewer)

    Returns:
        编码后的 JWT 字符串
    """
    now = datetime.now(timezone.utc)
    payload = {
        'sub': str(user_id),              # 用户ID — PyJWT 2.10+ 要求 sub 为字符串
        'username': username,             # 用户名 (方便日志/调试)
        'role': role,                     # 角色
        'type': 'access',                 # token 类型
        'iat': now,                       # 签发时间 (issued at)
        'exp': now + _get_access_expiry(),  # 过期时间
    }
    token = jwt.encode(payload, _get_jwt_secret(), algorithm='HS256')
    return token


def generate_refresh_token(user_id: int, username: str) -> str:
    """
    生成 JWT Refresh Token (长有效期, 仅用于刷新)

    Args:
        user_id: 用户ID
        username: 用户名

    Returns:
        编码后的 JWT 字符串
    """
    now = datetime.now(timezone.utc)
    payload = {
        'sub': str(user_id),              # PyJWT 2.10+ 要求 sub 为字符串
        'username': username,
        'type': 'refresh',
        'iat': now,
        'exp': now + _get_refresh_expiry(),
    }
    token = jwt.encode(payload, _get_jwt_secret(), algorithm='HS256')
    return token


def generate_token_pair(user_id: int, username: str, role: str) -> Dict[str, str]:
    """
    生成 Access + Refresh Token 对

    Returns:
        {'access_token': str, 'refresh_token': str, 'token_type': 'Bearer', 'expires_in': int}
    """
    return {
        'access_token': generate_access_token(user_id, username, role),
        'refresh_token': generate_refresh_token(user_id, username),
        'token_type': 'Bearer',
        'expires_in': int(_get_access_expiry().total_seconds()),
    }


def decode_token(token: str, expected_type: str = 'access') -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    解码并验证 JWT Token

    Args:
        token: JWT 字符串
        expected_type: 期望的 token 类型 ('access' 或 'refresh')

    Returns:
        (payload, error_message): 成功返回 (payload_dict, None)，
                                  失败返回 (None, error_msg)
    """
    try:
        payload = jwt.decode(token, _get_jwt_secret(), algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        return None, 'Token 已过期。'
    except jwt.InvalidTokenError as e:
        return None, f'Token 无效: {str(e)}'

    # 验证 token 类型
    if payload.get('type') != expected_type:
        return None, f'Token 类型不匹配，期望 {expected_type}，实际 {payload.get("type")}。'

    return payload, None


def decode_access_token(token: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """解码 Access Token (便捷方法)"""
    return decode_token(token, expected_type='access')


def decode_refresh_token(token: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """解码 Refresh Token (便捷方法)"""
    return decode_token(token, expected_type='refresh')


def extract_token_from_header() -> Optional[str]:
    """
    从请求头提取 Bearer token

    支持两种 Authorization 格式:
      - Bearer <token>
      - <token> (直接传 token)

    Returns:
        JWT 字符串 或 None
    """
    from flask import request

    auth_header = request.headers.get('Authorization', '')
    if not auth_header:
        return None

    # Bearer 格式
    if auth_header.startswith('Bearer '):
        return auth_header[7:]

    # 直接传 token (兼容)
    if auth_header.count('.') >= 2:  # JWT 格式: header.payload.signature
        return auth_header

    return None


def get_user_from_jwt() -> Optional[Any]:
    """
    从当前请求的 JWT Bearer token 中获取 User 对象

    Returns:
        User 对象 或 None
    """
    token = extract_token_from_header()
    if not token:
        return None

    payload, error = decode_access_token(token)
    if error:
        logger.debug(f"JWT 解码失败: {error}")
        return None

    user_id = payload.get('sub')
    if not user_id:
        return None

    # 延迟导入避免循环依赖
    from app.models.user import User
    # sub 存储为字符串，查询时转为 int
    from app import db
    return db.session.get(User, int(user_id))
