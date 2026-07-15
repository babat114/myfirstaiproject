"""
============================================
评论模型
管理公开模型的用户评论区 — 支持审核、回复、删除
============================================
"""
from app import db
from app._timezone import localnow


class Comment(db.Model):
    """评论表 — 用户对公开模型的评论

    审核机制:
        is_visible=True  → 评论正常显示
        is_visible=False → 评论被自动屏蔽 (含违规内容) 或管理员手动屏蔽

    权限:
        - 普通用户: 可发表评论, 可删除自己的评论
        - 管理员: 可删除任何评论, 可恢复被屏蔽评论
        - 评论作者: 可删除自己的评论
    """

    __tablename__ = 'comments'

    # 主键
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # 外键
    model_id = db.Column(
        db.Integer,
        db.ForeignKey('model_records.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )

    # 评论内容
    content = db.Column(db.Text, nullable=False)

    # 审核字段
    is_visible = db.Column(db.Boolean, default=True, nullable=False,
                           comment='False=被屏蔽/待审核，True=正常显示')
    moderation_reason = db.Column(
        db.String(200), nullable=True,
        comment='屏蔽原因: auto_filtered(自动过滤)/admin_removed(管理员删除)'
    )

    # 回复支持 (自引用外键)
    parent_id = db.Column(
        db.Integer,
        db.ForeignKey('comments.id', ondelete='CASCADE'),
        nullable=True,
        index=True,
    )

    # 时间戳
    created_at = db.Column(db.DateTime, default=lambda: localnow(), nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=lambda: localnow(),
        onupdate=lambda: localnow(),
        nullable=False,
    )

    # ============ 关联关系 ============

    model = db.relationship(
        'ModelRecord',
        backref=db.backref('comments', lazy='dynamic', cascade='all, delete-orphan'),
    )
    user = db.relationship('User', backref=db.backref('comments', lazy='dynamic'))

    # 自引用: 回复关系
    parent = db.relationship(
        'Comment',
        remote_side=[id],
        backref=db.backref('replies', lazy='dynamic', cascade='all, delete-orphan'),
    )

    # ============ 属性 ============

    @property
    def is_reply(self) -> bool:
        """是否为回复 (非顶级评论)"""
        return self.parent_id is not None

    @property
    def reply_count(self) -> int:
        """回复数量"""
        return self.replies.filter_by(is_visible=True).count()

    @property
    def is_deleted_by_owner(self) -> bool:
        """是否被作者删除"""
        return not self.is_visible and self.moderation_reason == 'owner_deleted'

    # ============ 方法 ============

    def soft_delete(self, reason: str = 'owner_deleted'):
        """软删除评论 (标记为不可见而非物理删除)"""
        self.is_visible = False
        self.moderation_reason = reason
        self.updated_at = localnow()

    def restore(self):
        """恢复被屏蔽的评论"""
        self.is_visible = True
        self.moderation_reason = None
        self.updated_at = localnow()

    def to_dict(self, include_user: bool = True) -> dict:
        """转换为字典"""
        data = {
            'id': self.id,
            'model_id': self.model_id,
            'content': self.content,
            'is_visible': self.is_visible,
            'moderation_reason': self.moderation_reason,
            'parent_id': self.parent_id,
            'is_reply': self.is_reply,
            'reply_count': self.reply_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_user and self.user:
            data['user'] = {
                'id': self.user.id,
                'username': self.user.username,
                'full_name': self.user.full_name,
                'avatar_url': self.user.avatar_url,
                'is_admin': self.user.is_admin,
            }
        return data

    def __repr__(self):
        return f'<Comment {self.id} by {self.user.username if self.user else "?"} on model {self.model_id}>'
