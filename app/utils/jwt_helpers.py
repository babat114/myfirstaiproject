"""
============================================
JWT 认证工具
生成、验证、刷新 JSON Web Token
============================================
"""
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from flask import current_app

from app import logger

# ── Token 黑名单 (数据库持久化, 服务重启不丢失) ──


def is_token_revoked(jti: str) -> bool:
    """检查 token jti 是否已被撤销 (数据库查询)"""
    from app import db
    from app.models.revoked_token import RevokedToken
    return db.session.execute(
        db.select(RevokedToken.id).filter_by(jti=jti).limit(1)
    ).scalar() is not None


def revoke_token(jti: str, exp_timestamp: float):
    """将 token jti 加入数据库黑名单, 有效期至 exp_timestamp"""
    from app import db
    from app.models.revoked_token import RevokedToken
    try:
        expires_at = datetime.fromtimestamp(exp_timestamp, tz=UTC)
        entry = RevokedToken(jti=jti, expires_at=expires_at)
        db.session.add(entry)
        db.session.commit()
    except Exception:
        db.session.rollback()
        # 唯一约束冲突 — token 已在黑名单中, 忽略
        pass


def cleanup_expired_revocations():
    """清理过期的黑名单条目 (可由定时任务调用)"""
    from app import db
    from app.models.revoked_token import RevokedToken
    try:
        db.session.execute(
            db.delete(RevokedToken).where(
                RevokedToken.expires_at < datetime.now(UTC)
            )
        )
        db.session.commit()
    except Exception:
        db.session.rollback()



def _get_jwt_secret() -> str:
    """获取 JWT 签名密钥 — 生产环境必须显式配置, 无默认回退"""
    secret = current_app.config.get('JWT_SECRET_KEY')
    if not secret:
        raise RuntimeError(
            'JWT_SECRET_KEY 未配置。生产环境必须在 .env 中设置 JWT_SECRET_KEY。'
        )
    return secret


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
    now = datetime.now(UTC)
    payload = {
        'sub': str(user_id),              # 用户ID — PyJWT 2.10+ 要求 sub 为字符串
        'username': username,             # 用户名 (方便日志/调试)
        'role': role,                     # 角色
        'type': 'access',                 # token 类型
        'jti': str(uuid.uuid4()),         # JWT ID — 用于黑名单/撤销
        'iat': now,                       # 签发时间 (issued at)
        'exp': now + _get_access_expiry(),  # 过期时间
    }
    token = jwt.encode(payload, _get_jwt_secret(), algorithm='HS256')
    return token


def generate_refresh_token(user_id: int, username: str, token_version: int = 1) -> str:
    """
    生成 JWT Refresh Token (长有效期, 仅用于刷新)

    Args:
        user_id: 用户ID
        username: 用户名
        token_version: Token 版本号 (改密码时递增, 使旧 Token 失效)

    Returns:
        编码后的 JWT 字符串
    """
    now = datetime.now(UTC)
    payload = {
        'sub': str(user_id),              # PyJWT 2.10+ 要求 sub 为字符串
        'username': username,
        'type': 'refresh',
        'jti': str(uuid.uuid4()),         # JWT ID — 用于刷新轮换时撤销旧 token
        'ver': token_version,             # Token 版本 — 改密码时递增以撤销旧 Token
        'iat': now,
        'exp': now + _get_refresh_expiry(),
    }
    token = jwt.encode(payload, _get_jwt_secret(), algorithm='HS256')
    return token


def generate_token_pair(user_id: int, username: str, role: str,
                        token_version: int = 1) -> dict[str, str]:
    """
    生成 Access + Refresh Token 对

    Returns:
        {'access_token': str, 'refresh_token': str, 'token_type': 'Bearer', 'expires_in': int}
    """
    return {
        'access_token': generate_access_token(user_id, username, role),
        'refresh_token': generate_refresh_token(user_id, username, token_version),
        'token_type': 'Bearer',
        'expires_in': int(_get_access_expiry().total_seconds()),
    }


def decode_token(token: str, expected_type: str = 'access') -> tuple[dict[str, Any] | None, str | None]:
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


def decode_access_token(token: str) -> tuple[dict[str, Any] | None, str | None]:
    """解码 Access Token (便捷方法)"""
    return decode_token(token, expected_type='access')


def decode_refresh_token(token: str) -> tuple[dict[str, Any] | None, str | None]:
    """解码 Refresh Token (便捷方法, 额外验证 token_version 和撤销状态)"""
    payload, error = decode_token(token, expected_type='refresh')
    if error:
        return None, error

    # 验证 token 未被撤销 (refresh token 轮换)
    jti = payload.get('jti')
    if jti and is_token_revoked(jti):
        return None, 'Token 已被撤销，请重新登录。'

    # 验证 token_version 与数据库一致 (改密码后旧 Token 自动失效)
    token_ver = payload.get('ver')
    if token_ver is not None:
        user_id = payload.get('sub')
        if user_id:
            from app import db as _db
            from app.models.user import User
            user = _db.session.get(User, int(user_id))
            if user and user.token_version != token_ver:
                return None, 'Token 已失效 (密码已更改)，请重新登录。'

    return payload, None


def extract_token_from_header() -> str | None:
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


def get_user_from_jwt() -> Any | None:
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
    # sub 存储为字符串，查询时转为 int
    from app import db
    from app.models.user import User
    return db.session.get(User, int(user_id))
