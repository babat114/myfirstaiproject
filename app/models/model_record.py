"""
============================================
AI模型注册模型
管理已训练的AI模型版本和元数据
============================================
"""

import json
import uuid

from sqlalchemy import CheckConstraint

from app import db
from app._timezone import localnow
from app.models.mixins import AccessControlMixin


class ModelRecord(AccessControlMixin, db.Model):
    """AI模型注册表 — 存储模型版本元数据和性能指标

    模型类型 (model_type):
        classification — 分类任务 (二分类/多分类)
        regression     — 回归任务 (预测连续值)
        clustering     — 聚类任务 (无监督分组)
        nlp            — 自然语言处理
        computer_vision — 计算机视觉
        reinforcement  — 强化学习
        generative     — 生成式模型 (GAN/VAE/Diffusion)

    模型状态 (status):
        draft    — 草稿: 仅元数据, 无权重文件
        trained  — 已训练: 有权重文件+指标
        deployed — 已部署: 正在生产环境服务
        archived — 已归档: 保留记录但不再使用
        failed   — 失败: 训练或注册过程出错
    """

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
            'classification',
            'regression',
            'clustering',
            'nlp',
            'computer_vision',
            'reinforcement',
            'generative',
            'other',
            name='model_types',
        ),
        default='other',
        nullable=False,
    )
    framework = db.Column(db.String(50), nullable=True, comment='PyTorch, TensorFlow, scikit-learn, etc.')

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

    # 关键指标 — 分类任务
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    loss = db.Column(db.Float, nullable=True)

    # 关键指标 — 回归任务
    r2 = db.Column(db.Float, nullable=True, comment='R² 决定系数')
    mse = db.Column(db.Float, nullable=True, comment='均方误差')
    mae = db.Column(db.Float, nullable=True, comment='平均绝对误差')

    # 训练信息
    training_dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True, index=True)
    training_job_id = db.Column(db.Integer, db.ForeignKey('training_jobs.id'), nullable=True)
    training_duration_seconds = db.Column(db.Integer, nullable=True)

    # --- 独立测试集评估 ---
    independent_test_dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)
    independent_accuracy = db.Column(db.Float, nullable=True)
    independent_f1_score = db.Column(db.Float, nullable=True)
    independent_metrics_json = db.Column(db.Text, nullable=True)

    # 状态和可见性
    status = db.Column(
        db.Enum('draft', 'trained', 'deployed', 'archived', 'failed', name='model_status'),
        default='draft',
        nullable=False,
        index=True,
    )
    is_public = db.Column(db.Boolean, default=False, index=True)
    deployment_url = db.Column(db.String(512), nullable=True)

    # 唯一标识符
    uuid = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))

    # 外键
    owner_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)

    # 时间戳
    created_at = db.Column(db.DateTime, default=lambda: localnow(), nullable=False)
    updated_at = db.Column(db.DateTime, default=lambda: localnow(), onupdate=lambda: localnow(), nullable=False)

    # 表级约束: 指标值域检查
    __table_args__ = (
        CheckConstraint('accuracy IS NULL OR (accuracy >= 0.0 AND accuracy <= 1.0)', name='ck_accuracy_range'),
        CheckConstraint('precision IS NULL OR (precision >= 0.0 AND precision <= 1.0)', name='ck_precision_range'),
        CheckConstraint('recall IS NULL OR (recall >= 0.0 AND recall <= 1.0)', name='ck_recall_range'),
        CheckConstraint('f1_score IS NULL OR (f1_score >= 0.0 AND f1_score <= 1.0)', name='ck_f1_score_range'),
        CheckConstraint('loss IS NULL OR loss >= 0.0', name='ck_loss_nonnegative'),
        CheckConstraint('mse IS NULL OR mse >= 0.0', name='ck_mse_nonnegative'),
        CheckConstraint('mae IS NULL OR mae >= 0.0', name='ck_mae_nonnegative'),
        CheckConstraint('r2 IS NULL OR r2 <= 1.0', name='ck_r2_upper'),
        # 性能索引 — leaderboard排序 + 列表过滤
        db.Index('ix_model_records_accuracy', 'accuracy'),
        db.Index('ix_model_records_f1_score', 'f1_score'),
        db.Index('ix_model_records_r2', 'r2'),
        db.Index('ix_model_records_model_type', 'model_type'),
        db.Index('ix_model_records_framework', 'framework'),
    )

    # 关联关系
    owner = db.relationship('User', back_populates='model_records')
    training_dataset = db.relationship('Dataset', foreign_keys=[training_dataset_id])
    training_job = db.relationship('TrainingJob', foreign_keys=[training_job_id])
    independent_test_dataset = db.relationship('Dataset', foreign_keys=[independent_test_dataset_id])

    # ============ 属性 ============

    @property
    def metrics_dict(self) -> dict:
        """获取指标字典 (安全反序列化, 损坏的 JSON 返回空字典)"""
        if self.metrics_json:
            try:
                return json.loads(self.metrics_json)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    @property
    def hyperparameters_dict(self) -> dict:
        """获取超参数字典 (安全反序列化, 损坏的 JSON 返回空字典)"""
        if self.hyperparameters_json:
            try:
                return json.loads(self.hyperparameters_json)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    @property
    def file_size_mb(self) -> float:
        size = self.file_size or 0
        return round(size / (1024 * 1024), 2)

    @property
    def name_slug(self) -> str:
        """文件名友好的 slug"""
        import re

        return re.sub(r'[^a-zA-Z0-9_-]', '-', self.name.lower()).strip('-')

    # ============ 方法 ============

    def set_metrics(self, metrics: dict):
        """
        设置性能指标

        存储策略:
        - 分类: accuracy/precision/recall/f1_score 列存 macro 平均值
        - 回归: r2/mse/mae 列存测试集指标 (优先 test_ 前缀)
        - metrics_json 存完整指标
        - loss 单独存储
        """
        # 检测是否包含回归指标 (含 test_/train_ 前缀 或 裸键)
        _reg_suffixes = ('mse', 'mae', 'r2', 'rmse', 'r2_score')
        has_reg = any(k.endswith(_reg_suffixes) for k in metrics)
        has_cls = (
            'accuracy' in metrics
            or 'precision_macro' in metrics
            or any('_accuracy' in k or '_precision_' in k for k in metrics)
        )
        has_cluster = any(
            k.endswith(suffix)
            for k in metrics
            for suffix in (
                'silhouette_score',
                'davies_bouldin_score',
                'calinski_harabasz_score',
                'inertia',
                'adjusted_rand_score',
                'normalized_mutual_info_score',
            )
        )

        if has_cls or has_reg or has_cluster:
            self.metrics_json = json.dumps(metrics, ensure_ascii=False)

        # 分类指标 — 优先取 test_ 前缀 (测试集指标), 回退到裸键
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.loss = None
        for primary, fallback, attr in [
            ('accuracy', None, 'accuracy'),
            ('precision_macro', 'precision', 'precision'),
            ('recall_macro', 'recall', 'recall'),
            ('f1_macro', 'f1_score', 'f1_score'),
            ('loss', None, 'loss'),
        ]:
            for prefix in ('test_', ''):
                full = f'{prefix}{primary}'
                if full in metrics:
                    setattr(self, attr, float(metrics[full]))
                    break
                if fallback:
                    full_fb = f'{prefix}{fallback}'
                    if full_fb in metrics:
                        setattr(self, attr, float(metrics[full_fb]))
                        break
            # If still None after trying both prefixes, try unprefixed fallback
            if getattr(self, attr) is None and fallback and fallback in metrics:
                setattr(self, attr, float(metrics[fallback]))

        # 回归指标 — 优先取 test_ 前缀 (测试集指标), 回退到裸键
        self.r2 = None
        self.mse = None
        self.mae = None
        for key, attr in [('r2', 'r2'), ('mse', 'mse'), ('mae', 'mae')]:
            for prefix in ('test_', ''):
                full = f'{prefix}{key}'
                if full in metrics:
                    setattr(self, attr, float(metrics[full]))
                    break  # test_ 前缀优先, 找到即停

    @property
    def independent_metrics_dict(self) -> dict:
        """获取独立测试集评估指标字典"""
        if self.independent_metrics_json:
            try:
                return json.loads(self.independent_metrics_json)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    def set_independent_metrics(self, metrics: dict):
        """
        设置独立测试集评估指标

        存储策略:
        - independent_accuracy: 独立测试集准确率
        - independent_f1_score: 独立测试集F1
        - independent_metrics_json: 完整指标
        """
        self.independent_metrics_json = json.dumps(metrics, ensure_ascii=False)
        # 提取关键指标到类型化列
        for key, attr in [
            ('ind_test_accuracy', 'independent_accuracy'),
            ('accuracy', 'independent_accuracy'),
            ('ind_test_f1_weighted', 'independent_f1_score'),
            ('ind_test_f1_macro', 'independent_f1_score'),
            ('f1_weighted', 'independent_f1_score'),
            ('f1_macro', 'independent_f1_score'),
        ]:
            if key in metrics and getattr(self, attr) is None:
                setattr(self, attr, float(metrics[key]))

    def set_hyperparameters(self, params: dict):
        """设置超参数"""
        self.hyperparameters_json = json.dumps(params, ensure_ascii=False)

    def deploy(self, url: str):
        """标记模型为已部署 (不提交 — 由调用方服务层控制)"""
        self.status = 'deployed'
        self.deployment_url = url

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
            # 分类指标 (训练集切分评估)
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'loss': self.loss,
            # 回归指标
            'r2': self.r2,
            'mse': self.mse,
            'mae': self.mae,
            # 独立测试集评估
            'independent_test_dataset_id': self.independent_test_dataset_id,
            'independent_accuracy': self.independent_accuracy,
            'independent_f1_score': self.independent_f1_score,
            'independent_metrics': self.independent_metrics_dict,
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

    # ============ 权限检查 (继承自 AccessControlMixin) ============
    # is_viewable_by / is_editable_by 由 AccessControlMixin 提供
