"""
============================================
用户模型
管理用户账户、角色和认证信息
============================================
"""

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from app import db, login_manager
from app._timezone import localnow


class User(UserMixin, db.Model):
    """用户模型 - 存储用户账户信息"""

    __tablename__ = 'users'

    # 主键
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # 基本信息
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)

    # 用户资料
    full_name = db.Column(db.String(120), nullable=True)
    avatar_url = db.Column(db.String(512), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    organization = db.Column(db.String(200), nullable=True)

    # 角色和状态
    role = db.Column(db.Enum('admin', 'researcher', 'viewer', name='user_roles'), default='researcher', nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_verified = db.Column(db.Boolean, default=False, nullable=False)

    # API 密钥 (SHA256 哈希存储, 原始密钥仅在生成时返回一次)
    api_key = db.Column(db.String(128), unique=True, nullable=True)

    # JWT Refresh Token 版本 — 改密码时递增, 使所有已签发 Refresh Token 失效
    token_version = db.Column(db.Integer, default=1, nullable=False)

    # 账号锁定 — 连续登录失败 ≥ MAX_FAILED_ATTEMPTS 后锁定 LOCKOUT_DURATION 分钟
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION = 15  # 分钟
    failed_login_attempts = db.Column(db.Integer, default=0, nullable=False)
    locked_until = db.Column(db.DateTime, nullable=True)

    # 时间戳
    created_at = db.Column(db.DateTime, default=lambda: localnow(), nullable=False)
    updated_at = db.Column(db.DateTime, default=lambda: localnow(), onupdate=lambda: localnow(), nullable=False)
    last_login_at = db.Column(db.DateTime, nullable=True)

    # 关联关系
    datasets = db.relationship('Dataset', back_populates='owner', lazy='dynamic', cascade='all, delete-orphan')
    model_records = db.relationship('ModelRecord', back_populates='owner', lazy='dynamic', cascade='all, delete-orphan')
    training_jobs = db.relationship('TrainingJob', back_populates='owner', lazy='dynamic', cascade='all, delete-orphan')

    # ============ 密码管理 ============

    def set_password(self, password: str):
        """设置密码 (自动哈希)"""
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    def check_password(self, password: str) -> bool:
        """验证密码"""
        return check_password_hash(self.password_hash, password)

    # ============ 权限检查 ============

    @property
    def is_admin(self) -> bool:
        return self.role == 'admin'

    @property
    def can_upload(self) -> bool:
        return self.role in ('admin', 'researcher')

    @property
    def can_delete(self) -> bool:
        return self.role == 'admin'

    @property
    def is_locked(self) -> bool:
        """检查账号是否处于锁定状态"""
        return bool(self.locked_until and self.locked_until > localnow())

    def reset_lockout(self):
        """重置登录失败计数和锁定状态 (成功登录后调用)"""
        self.failed_login_attempts = 0
        self.locked_until = None

    def record_failed_attempt(self):
        """记录一次登录失败, 超过阈值则锁定"""
        self.failed_login_attempts = (self.failed_login_attempts or 0) + 1
        if self.failed_login_attempts >= self.MAX_FAILED_ATTEMPTS:
            from datetime import timedelta

            self.locked_until = localnow() + timedelta(minutes=self.LOCKOUT_DURATION)

    # ============ 统计属性 ============

    @property
    def dataset_count(self) -> int:
        return self.datasets.count()

    @property
    def model_count(self) -> int:
        return self.model_records.count()

    @property
    def training_count(self) -> int:
        return self.training_jobs.count()

    # ============ 序列化 ============

    def to_dict(self, include_private: bool = False) -> dict:
        """转换为字典"""
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'organization': self.organization,
            'role': self.role,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'avatar_url': self.avatar_url,
            'bio': self.bio,
            'dataset_count': self.dataset_count,
            'model_count': self.model_count,
            'training_count': self.training_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login_at': self.last_login_at.isoformat() if self.last_login_at else None,
        }
        if include_private:
            # 注意: api_key 已哈希存储, 原始密钥仅在生成/重新生成时返回一次
            data['failed_login_attempts'] = self.failed_login_attempts
            data['locked_until'] = self.locked_until.isoformat() if self.locked_until else None
            data['is_locked'] = self.is_locked
        return data

    def __repr__(self):
        return f'<User {self.username} ({self.role})>'


@login_manager.user_loader
def load_user(user_id: str) -> User | None:
    """Flask-Login 用户加载器"""
    from app import db

    return db.session.get(User, int(user_id))
