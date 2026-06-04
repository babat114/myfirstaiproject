"""
============================================
认证服务
处理用户注册、登录、API密钥管理
============================================
"""
import secrets
from datetime import datetime, timezone
from typing import Optional, Tuple
from sqlalchemy import or_
from app import db, logger
from app.models.user import User


class AuthService:
    """用户认证服务类"""

    @staticmethod
    def register(username: str, email: str, password: str,
                 full_name: str = None) -> Tuple[Optional[User], Optional[str]]:
        """
        注册新用户

        Args:
            username: 用户名
            email: 邮箱
            password: 密码
            full_name: 全名 (可选)

        Returns:
            (User, error_message): 成功返回 (user, None)，失败返回 (None, error_msg)
        """
        # 验证用户名唯一性
        if User.query.filter_by(username=username).first():
            return None, '用户名已被注册。'

        # 验证邮箱唯一性
        if User.query.filter_by(email=email).first():
            return None, '邮箱已被注册。'

        # 验证密码强度
        if len(password) < 8:
            return None, '密码长度至少为 8 个字符。'

        if not any(c.isupper() for c in password):
            return None, '密码必须包含至少一个大写字母。'

        if not any(c.isdigit() for c in password):
            return None, '密码必须包含至少一个数字。'

        # 创建用户
        try:
            user = User(
                username=username,
                email=email,
                full_name=full_name,
            )
            user.set_password(password)
            user.api_key = AuthService._generate_api_key()

            db.session.add(user)
            db.session.commit()

            logger.info(f"新用户注册: {username} ({email})")
            return user, None

        except Exception as e:
            db.session.rollback()
            logger.error(f"注册用户失败: {e}")
            return None, '注册失败，请稍后重试。'

    @staticmethod
    def login(login_id: str, password: str) -> Tuple[Optional[User], Optional[str]]:
        """
        用户登录

        Args:
            login_id: 用户名或邮箱
            password: 密码

        Returns:
            (User, error_message)
        """
        # 查找用户 (支持用户名或邮箱登录)
        user = User.query.filter(
            or_(User.username == login_id, User.email == login_id)
        ).first()

        if not user:
            return None, '用户名或邮箱不存在。'

        if not user.is_active:
            return None, '账户已被禁用，请联系管理员。'

        if not user.check_password(password):
            return None, '密码错误。'

        # 更新最后登录时间
        user.last_login_at = datetime.now(timezone.utc)
        db.session.commit()

        logger.info(f"用户登录: {user.username}")
        return user, None

    @staticmethod
    def update_profile(user: User, data: dict) -> Tuple[bool, Optional[str]]:
        """
        更新用户资料

        Args:
            user: 用户对象
            data: 包含要更新的字段的字典

        Returns:
            (success, error_message)
        """
        allowed_fields = {'full_name', 'bio', 'organization', 'avatar_url'}

        try:
            for field, value in data.items():
                if field in allowed_fields and hasattr(user, field):
                    setattr(user, field, value)

            user.updated_at = datetime.now(timezone.utc)
            db.session.commit()
            return True, None

        except Exception as e:
            db.session.rollback()
            logger.error(f"更新用户资料失败: {e}")
            return False, '更新失败，请稍后重试。'

    @staticmethod
    def change_password(user: User, old_password: str,
                        new_password: str) -> Tuple[bool, Optional[str]]:
        """
        修改密码

        Returns:
            (success, error_message)
        """
        if not user.check_password(old_password):
            return False, '当前密码错误。'

        if len(new_password) < 8:
            return False, '新密码长度至少为 8 个字符。'

        user.set_password(new_password)
        db.session.commit()

        logger.info(f"用户 {user.username} 修改了密码")
        return True, None

    @staticmethod
    def regenerate_api_key(user: User) -> str:
        """
        重新生成 API 密钥

        Returns:
            新的 API 密钥
        """
        user.api_key = AuthService._generate_api_key()
        db.session.commit()
        return user.api_key

    @staticmethod
    def get_user_by_api_key(api_key: str) -> Optional[User]:
        """
        通过 API 密钥获取用户 (用于 API 认证)

        Returns:
            User 或 None
        """
        return User.query.filter_by(api_key=api_key, is_active=True).first()

    @staticmethod
    def list_users(page: int = 1, per_page: int = 20) -> dict:
        """
        获取用户列表 (管理员功能)

        Returns:
            分页用户数据
        """
        pagination = User.query.order_by(User.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        return {
            'users': [u.to_dict() for u in pagination.items],
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev,
        }

    @staticmethod
    def login_jwt(login_id: str, password: str) -> Tuple[Optional[dict], Optional[str], int]:
        """
        JWT 登录 — 验证凭据并返回 Token 对

        Args:
            login_id: 用户名或邮箱
            password: 密码

        Returns:
            (token_pair, error_message, status_code)
            成功: ({'access_token': ..., 'refresh_token': ..., ...}, None, 200)
            失败: (None, error_msg, 401)
        """
        from app.utils.jwt_helpers import generate_token_pair

        user, error = AuthService.login(login_id, password)
        if error:
            return None, error, 401

        tokens = generate_token_pair(user.id, user.username, user.role)
        return tokens, None, 200

    @staticmethod
    def refresh_jwt(refresh_token: str) -> Tuple[Optional[dict], Optional[str], int]:
        """
        使用 Refresh Token 刷新 Access Token

        Args:
            refresh_token: 有效的 refresh token

        Returns:
            (new_token_pair, error_message, status_code)
        """
        from app.utils.jwt_helpers import decode_refresh_token, generate_token_pair
        from app.models.user import User

        payload, error = decode_refresh_token(refresh_token)
        if error:
            return None, error, 401

        user_id = payload.get('sub')
        user = db.session.get(User, int(user_id))
        if not user:
            return None, '用户不存在或已被删除。', 401

        if not user.is_active:
            return None, '账户已被禁用。', 401

        tokens = generate_token_pair(user.id, user.username, user.role)
        return tokens, None, 200

    @staticmethod
    def _generate_api_key() -> str:
        """生成唯一的 API 密钥"""
        return f"ak_{secrets.token_hex(24)}"

    @staticmethod
    def delete_user(user: User) -> bool:
        """删除用户 (管理员功能)"""
        try:
            db.session.delete(user)
            db.session.commit()
            logger.info(f"用户 {user.username} 已被删除")
            return True
        except Exception as e:
            db.session.rollback()
            logger.error(f"删除用户失败: {e}")
            return False
