"""
============================================
AI模型注册模型
管理已训练的AI模型版本和元数据
============================================
"""
import uuid
from datetime import datetime
from app import db


class ModelRecord(db.Model):
    """AI模型注册 - 存储模型版本和性能指标"""

    __tablename__ = 'model_records'

    # 主键
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # 基本信息
    name = db.Column(db.String(200), nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)
    version = db.Column(db.String(20), default='1.0.0')

    # 模型类型
    model_type = db.Column(
        db.Enum(
            'classification', 'regression', 'clustering',
            'nlp', 'computer_vision', 'reinforcement',
            'generative', 'other',
            name='model_types'
        ),
        default='other',
        nullable=False
    )
    framework = db.Column(
        db.String(50),
        nullable=True,
        comment='PyTorch, TensorFlow, scikit-learn, etc.'
    )

    # 文件路径
    model_file_path = db.Column(db.String(512), nullable=True)
    weights_file_path = db.Column(db.String(512), nullable=True)
    config_file_path = db.Column(db.String(512), nullable=True)
    file_size = db.Column(db.BigInteger, default=0)

    # 超参数和配置 (JSON 字符串)
    hyperparameters_json = db.Column(db.Text, nullable=True)
    architecture_json = db.Column(db.Text, nullable=True)

    # 性能指标 (JSON 字符串)
    metrics_json = db.Column(db.Text, nullable=True)

    # 关键指标
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    loss = db.Column(db.Float, nullable=True)

    # 训练信息
    training_dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)
    training_job_id = db.Column(db.Integer, db.ForeignKey('training_jobs.id'), nullable=True)
    training_duration_seconds = db.Column(db.Integer, nullable=True)

    # 状态和可见性
    status = db.Column(
        db.Enum('draft', 'trained', 'deployed', 'archived', 'failed',
                name='model_status'),
        default='draft',
        nullable=False
    )
    is_public = db.Column(db.Boolean, default=False)
    deployment_url = db.Column(db.String(512), nullable=True)

    # 唯一标识符
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
    owner = db.relationship('User', back_populates='model_records')
    training_dataset = db.relationship('Dataset', foreign_keys=[training_dataset_id])
    training_job = db.relationship('TrainingJob', foreign_keys=[training_job_id])

    # ============ 属性 ============

    @property
    def metrics_dict(self) -> dict:
        """获取指标字典"""
        import json
        if self.metrics_json:
            return json.loads(self.metrics_json)
        return {}

    @property
    def hyperparameters_dict(self) -> dict:
        """获取超参数字典"""
        import json
        if self.hyperparameters_json:
            return json.loads(self.hyperparameters_json)
        return {}

    @property
    def file_size_mb(self) -> float:
        return round(self.file_size / (1024 * 1024), 2)

    # ============ 方法 ============

    def set_metrics(self, metrics: dict):
        """设置性能指标"""
        import json
        self.metrics_json = json.dumps(metrics, ensure_ascii=False)
        self.accuracy = metrics.get('accuracy')
        self.precision = metrics.get('precision')
        self.recall = metrics.get('recall')
        self.f1_score = metrics.get('f1_score')
        self.loss = metrics.get('loss')

    def set_hyperparameters(self, params: dict):
        """设置超参数"""
        import json
        self.hyperparameters_json = json.dumps(params, ensure_ascii=False)

    def deploy(self, url: str):
        """标记模型为已部署"""
        self.status = 'deployed'
        self.deployment_url = url
        db.session.commit()

    def to_dict(self, include_files: bool = False) -> dict:
        """转换为字典"""
        data = {
            'id': self.id,
            'uuid': self.uuid,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'model_type': self.model_type,
            'framework': self.framework,
            'status': self.status,
            'is_public': self.is_public,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'loss': self.loss,
            'metrics': self.metrics_dict,
            'hyperparameters': self.hyperparameters_dict,
            'training_duration_seconds': self.training_duration_seconds,
            'deployment_url': self.deployment_url,
            'file_size': self.file_size,
            'file_size_mb': self.file_size_mb,
            'owner_id': self.owner_id,
            'owner_name': self.owner.username if self.owner else None,
            'training_dataset_id': self.training_dataset_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_files:
            data['model_file_path'] = self.model_file_path
            data['weights_file_path'] = self.weights_file_path
            data['config_file_path'] = self.config_file_path
        return data

    def __repr__(self):
        return f'<ModelRecord {self.name} v{self.version} ({self.status})>'
