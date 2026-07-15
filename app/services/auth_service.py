"""
============================================
认证服务
处理用户注册、登录、API密钥管理
============================================
"""

import hashlib
import secrets

from sqlalchemy import or_

from app import db, logger
from app._timezone import localnow
from app.models.user import User


class AuthService:
    """用户认证服务类"""

    @staticmethod
    def _validate_password_strength(password: str, username: str = None, email: str = None) -> str | None:
        """验证密码强度，返回错误消息或 None

        要求: 10+ 字符, 含大小写+数字+特殊字符, 不包含用户名/邮箱
        """
        if len(password) < 10:
            return '密码长度至少为 10 个字符。'
        if not any(c.isupper() for c in password):
            return '密码必须包含至少一个大写字母。'
        if not any(c.islower() for c in password):
            return '密码必须包含至少一个小写字母。'
        if not any(c.isdigit() for c in password):
            return '密码必须包含至少一个数字。'
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?/~`' for c in password):
            return '密码必须包含至少一个特殊字符 (!@#$%^&*...)。'
        # 密码不应包含用户名或邮箱 (大小写不敏感)
        password_lower = password.lower()
        if username and username.lower() in password_lower:
            return '密码不能包含用户名。'
        if email:
            email_prefix = email.split('@')[0].lower()
            if len(email_prefix) >= 4 and email_prefix in password_lower:
                return '密码不能包含邮箱前缀。'
        return None

    @staticmethod
    def register(username: str, email: str, password: str, full_name: str = None) -> tuple[User | None, str | None]:
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
        # 验证用户名唯一性 (SQLAlchemy 2.0 风格)
        if db.session.execute(db.select(User).filter_by(username=username)).scalar_one_or_none():
            return None, '用户名已被注册。'

        # 验证邮箱唯一性
        if db.session.execute(db.select(User).filter_by(email=email)).scalar_one_or_none():
            return None, '邮箱已被注册。'

        # 验证密码强度
        pw_error = AuthService._validate_password_strength(password, username, email)
        if pw_error:
            return None, pw_error

        # 创建用户
        try:
            user = User(
                username=username,
                email=email,
                full_name=full_name,
            )
            user.set_password(password)
            raw_key = AuthService._generate_api_key()
            user.api_key = AuthService._hash_api_key(raw_key)

            db.session.add(user)
            db.session.commit()

            logger.info(f'新用户注册: {username} ({email})')
            return user, None

        except Exception as e:
            db.session.rollback()
            from app.utils.helpers import sanitize_service_error

            return None, sanitize_service_error(e, '注册用户失败')

    @staticmethod
    def login(login_id: str, password: str) -> tuple[User | None, str | None]:
        """
        用户登录

        Args:
            login_id: 用户名或邮箱
            password: 密码

        Returns:
            (User, error_message)
        """
        # 查找用户 (支持用户名或邮箱登录, SQLAlchemy 2.0 风格)
        user = db.session.execute(
            db.select(User).filter(or_(User.username == login_id, User.email == login_id))
        ).scalar_one_or_none()

        if not user:
            return None, '用户名或密码错误。'

        if not user.is_active:
            return None, '账户已被禁用，请联系管理员。'

        # 账号锁定检查 (≥5次失败后锁定15分钟)
        if user.is_locked:
            remaining = int((user.locked_until - localnow()).total_seconds() / 60) + 1
            return None, f'账号已被临时锁定，请在 {remaining} 分钟后重试。'

        if not user.check_password(password):
            # 记录失败尝试, 超过阈值自动锁定
            user.record_failed_attempt()
            db.session.commit()
            if user.is_locked:
                logger.warning(f'账号 {user.username} 已被锁定 ({user.failed_login_attempts} 次失败)')
                return None, f'登录失败次数过多，账号已被锁定 {User.LOCKOUT_DURATION} 分钟。'
            return None, '用户名或密码错误。'

        # 登录成功 — 重置失败计数 + 更新最后登录时间
        user.reset_lockout()
        user.last_login_at = localnow()
        db.session.commit()

        logger.info(f'用户登录: {user.username}')
        return user, None

    @staticmethod
    def update_profile(user: User, data: dict) -> tuple[bool, str | None]:
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

            user.updated_at = localnow()
            db.session.commit()
            return True, None

        except Exception as e:
            db.session.rollback()
            from app.utils.helpers import sanitize_service_error

            return False, sanitize_service_error(e, '更新用户资料失败')

    @staticmethod
    def change_password(user: User, old_password: str, new_password: str) -> tuple[bool, str | None]:
        """
        修改密码

        Returns:
            (success, error_message)
        """
        if not user.check_password(old_password):
            return False, '当前密码错误。'

        pw_error = AuthService._validate_password_strength(new_password, user.username)
        if pw_error:
            return False, pw_error

        user.set_password(new_password)
        user.token_version = (user.token_version or 0) + 1  # 使所有旧 Refresh Token 失效
        db.session.commit()

        logger.info(f'用户 {user.username} 修改了密码 (token_version={user.token_version})')
        return True, None

    @staticmethod
    def regenerate_api_key(user: User) -> str:
        """
        重新生成 API 密钥 (仅此时返回原始密钥, 之后无法再次获取)

        Returns:
            新的 API 密钥 (原始值, 非哈希)
        """
        raw_key = AuthService._generate_api_key()
        user.api_key = AuthService._hash_api_key(raw_key)
        db.session.commit()
        return raw_key

    @staticmethod
    def get_user_by_api_key(api_key: str) -> User | None:
        """
        通过 API 密钥获取用户 (用于 API 认证)

        密钥在数据库中哈希存储, 查找时先哈希再匹配。

        Returns:
            User 或 None
        """
        key_hash = AuthService._hash_api_key(api_key)
        return db.session.execute(db.select(User).filter_by(api_key=key_hash, is_active=True)).scalar_one_or_none()

    @staticmethod
    def list_users(page: int = 1, per_page: int = 20, role: str = None, search: str = None) -> dict:
        """
        获取用户列表 (管理员功能)

        Returns:
            分页用户数据
        """
        from app.utils.helpers import paginate_query  # 延迟导入避免循环依赖

        stmt = db.select(User).order_by(User.created_at.desc())

        if role:
            stmt = stmt.filter_by(role=role)
        if search:
            term = f'%{search}%'
            stmt = stmt.filter(
                db.or_(
                    User.username.ilike(term),
                    User.email.ilike(term),
                    User.full_name.ilike(term),
                    User.organization.ilike(term),
                )
            )

        return paginate_query(stmt, page, per_page, item_key='users', transform_fn=lambda x: x.to_dict())

    @staticmethod
    def login_jwt(login_id: str, password: str) -> tuple[dict | None, str | None, int]:
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

        tokens = generate_token_pair(user.id, user.username, user.role, user.token_version)
        return tokens, None, 200

    @staticmethod
    def refresh_jwt(refresh_token: str) -> tuple[dict | None, str | None, int]:
        """
        使用 Refresh Token 刷新 Access Token

        Args:
            refresh_token: 有效的 refresh token

        Returns:
            (new_token_pair, error_message, status_code)
        """
        from app.models.user import User
        from app.utils.jwt_helpers import decode_refresh_token, generate_token_pair, revoke_token

        payload, error = decode_refresh_token(refresh_token)
        if error:
            return None, error, 401

        user_id = payload.get('sub')
        user = db.session.get(User, int(user_id))
        if not user:
            return None, '用户不存在或已被删除。', 401

        if not user.is_active:
            return None, '账户已被禁用。', 401

        # 轮换: 撤销旧 refresh token (防止重放攻击)
        old_jti = payload.get('jti')
        old_exp = payload.get('exp')
        if old_jti and old_exp:
            revoke_token(old_jti, old_exp)

        tokens = generate_token_pair(user.id, user.username, user.role, user.token_version)
        return tokens, None, 200

    @staticmethod
    def _generate_api_key() -> str:
        """生成唯一的 API 密钥"""
        return f'ak_{secrets.token_hex(24)}'

    @staticmethod
    def _hash_api_key(key: str) -> str:
        """对 API 密钥进行 SHA256 哈希 (数据库存储哈希值, 非明文)"""
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def delete_user(user: User) -> bool:
        """删除用户 (管理员功能)"""
        try:
            db.session.delete(user)
            db.session.commit()
            logger.info(f'用户 {user.username} 已被删除')
            return True
        except Exception as e:
            db.session.rollback()
            logger.error(f'删除用户失败: {e}')
            return False
