"""
============================================
数据集模型
管理上传的数据集及其元数据
============================================
"""
import os
import uuid
from datetime import datetime
from app import db


class Dataset(db.Model):
    """数据集模型 - 存储数据集信息和文件路径"""

    __tablename__ = 'datasets'

    # 主键
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # 基本信息
    name = db.Column(db.String(200), nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)
    file_path = db.Column(db.String(512), nullable=False)
    file_size = db.Column(db.BigInteger, default=0)  # 字节
    file_format = db.Column(db.String(20), nullable=False)  # csv, json, txt, etc.

    # 元数据
    version = db.Column(db.String(20), default='1.0.0')
    tags = db.Column(db.String(500), nullable=True)  # 逗号分隔的标签
    category = db.Column(
        db.Enum(
            'image', 'text', 'tabular', 'audio', 'video', 'other',
            name='dataset_categories'
        ),
        default='other',
        nullable=False
    )

    # 统计信息
    row_count = db.Column(db.Integer, default=0)
    column_count = db.Column(db.Integer, default=0)
    # 数据集统计摘要 (JSON 字符串)
    summary_json = db.Column(db.Text, nullable=True)

    # 状态
    status = db.Column(
        db.Enum('uploading', 'ready', 'processing', 'error', name='dataset_status'),
        default='uploading',
        nullable=False
    )
    is_public = db.Column(db.Boolean, default=False, nullable=False)

    # 唯一标识符 (用于 API)
    uuid = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))

    # 外键
    owner_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    # 关联关系
    owner = db.relationship('User', back_populates='datasets')
    training_jobs = db.relationship(
        'TrainingJob',
        back_populates='dataset',
        lazy='dynamic'
    )

    # ============ 属性 ============

    @property
    def tags_list(self) -> list:
        """将标签字符串转为列表"""
        if not self.tags:
            return []
        return [t.strip() for t in self.tags.split(',') if t.strip()]

    @tags_list.setter
    def tags_list(self, tags: list):
        """从列表设置标签"""
        self.tags = ','.join(tags) if tags else None

    @property
    def file_size_mb(self) -> float:
        """文件大小 (MB)"""
        return round(self.file_size / (1024 * 1024), 2)

    @property
    def filename(self) -> str:
        """获取文件名"""
        return os.path.basename(self.file_path) if self.file_path else ''

    # ============ 方法 ============

    def update_statistics(self, row_count: int = None, column_count: int = None,
                          summary: str = None):
        """更新数据集统计信息"""
        if row_count is not None:
            self.row_count = row_count
        if column_count is not None:
            self.column_count = column_count
        if summary is not None:
            self.summary_json = summary
        self.status = 'ready'
        db.session.commit()

    def to_dict(self, include_file_path: bool = False) -> dict:
        """转换为字典"""
        data = {
            'id': self.id,
            'uuid': self.uuid,
            'name': self.name,
            'description': self.description,
            'file_format': self.file_format,
            'file_size': self.file_size,
            'file_size_mb': self.file_size_mb,
            'version': self.version,
            'tags': self.tags_list,
            'category': self.category,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'status': self.status,
            'is_public': self.is_public,
            'owner_id': self.owner_id,
            'owner_name': self.owner.username if self.owner else None,
            'training_job_count': self.training_jobs.count(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_file_path:
            data['file_path'] = self.file_path
        return data

    def __repr__(self):
        return f'<Dataset {self.name} v{self.version}>'
