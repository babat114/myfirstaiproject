"""
============================================
已撤销 Token 模型
持久化 JWT Refresh Token 黑名单 (替代内存 dict)
============================================
"""
from app import db
from app._timezone import localnow


class RevokedToken(db.Model):
    """已撤销 Token 表 — 持久化 JWT 黑名单, 服务重启不丢失"""

    __tablename__ = 'revoked_tokens'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # JWT ID (jti claim) — 唯一标识一个已签发的 token
    jti = db.Column(db.String(64), unique=True, nullable=False, index=True)

    # Token 过期时间 — 超过此时间的条目可被定期清理
    expires_at = db.Column(db.DateTime, nullable=False)

    # 记录创建时间 (用于审计/调试)
    created_at = db.Column(db.DateTime, default=lambda: localnow(), nullable=False)

    def __repr__(self):
        return f'<RevokedToken jti={self.jti[:12]}... expires={self.expires_at}>'
