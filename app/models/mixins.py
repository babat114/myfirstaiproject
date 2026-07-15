"""
============================================
模型 Mixin 类
共享的模型行为: 权限检查 / 软删除 / 时间戳
============================================
"""

from app import db
from app._timezone import localnow


class AccessControlMixin:
    """统一权限检查 — 所有拥有 owner_id 的模型可混入此类的 is_viewable_by / is_editable_by

    要求:
        - 模型必须有 owner_id 列
        - 如果支持公开访问, 模型必须有 is_public 列 (否则 is_viewable_by 仅允许 owner + admin)
    """

    # 子类可覆盖此属性控制公开访问行为
    _supports_public_access: bool = True

    def is_viewable_by(self, user) -> bool:
        if user is None:
            return getattr(self, 'is_public', False)
        return getattr(self, 'is_public', False) or self.owner_id == user.id or user.is_admin

    def is_editable_by(self, user) -> bool:
        if user is None:
            return False
        return self.owner_id == user.id or user.is_admin


class TimestampMixin:
    """自动 created_at / updated_at 时间戳"""

    created_at = db.Column(db.DateTime, default=localnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=localnow, onupdate=localnow, nullable=False)
