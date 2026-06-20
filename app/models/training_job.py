"""
============================================
训练任务模型
追踪AI模型训练任务的状态和结果
============================================
"""
import json
import uuid
from datetime import datetime
from app import db
from app._timezone import localnow


class TrainingJob(db.Model):
    """训练任务模型 — 追踪AI模型训练的全生命周期

    任务生命周期 (status):
        queued    → running → completed  (正常流程)
        queued    → running → failed     (训练异常)
        queued    → running → paused → running → ...  (暂停恢复)
        any       → cancelled            (用户取消)

    进度追踪:
        progress_percent — 0-100% 进度条
        current_epoch / total_epochs — 训练轮次
        metrics_history_json — 每轮指标历史 (用于训练曲线图)
        final_metrics_json — 最终测试集指标

    资源需求:
        gpu_count, cpu_cores, memory_gb — 调度和资源分配
    """

    __tablename__ = 'training_jobs'

    # 主键
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # 基本信息
    name = db.Column(db.String(200), nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)

    # 任务配置
    task_type = db.Column(
        db.Enum(
            'training', 'fine_tuning', 'evaluation',
            'inference', 'data_preprocessing',
            name='task_types'
        ),
        default='training',
        nullable=False
    )
    framework = db.Column(db.String(50), nullable=True)

    # 状态追踪
    status = db.Column(
        db.Enum(
            'queued', 'preparing', 'running', 'paused',
            'completed', 'failed', 'cancelled',
            name='job_status'
        ),
        default='queued',
        nullable=False,
        index=True
    )

    # 进度信息
    progress_percent = db.Column(db.Float, default=0.0)
    current_epoch = db.Column(db.Integer, default=0)
    total_epochs = db.Column(db.Integer, default=0)
    current_step = db.Column(db.Integer, default=0)
    total_steps = db.Column(db.Integer, default=0)

    # 资源配置
    gpu_count = db.Column(db.Integer, default=0)
    cpu_cores = db.Column(db.Integer, default=1)
    memory_gb = db.Column(db.Float, default=4.0)

    # 日志和输出
    log_text = db.Column(db.Text, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    output_dir = db.Column(db.String(512), nullable=True)

    # 性能记录 (JSON)
    metrics_history_json = db.Column(db.Text, nullable=True)  # 每个 epoch 的指标
    final_metrics_json = db.Column(db.Text, nullable=True)

    # 时间信息
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    estimated_duration_seconds = db.Column(db.Integer, nullable=True)

    # 唯一标识符
    uuid = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))

    # 外键
    owner_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True, index=True)
    model_id = db.Column(db.Integer, db.ForeignKey('model_records.id', use_alter=True, name='fk_training_jobs_model_id'), nullable=True)

    # 时间戳
    created_at = db.Column(db.DateTime, default=lambda: localnow(), nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=lambda: localnow(),
        onupdate=lambda: localnow(),
        nullable=False
    )

    # 关联关系
    owner = db.relationship('User', back_populates='training_jobs')
    dataset = db.relationship('Dataset', back_populates='training_jobs')
    model = db.relationship('ModelRecord', foreign_keys=[model_id])

    # ============ 属性 ============

    @property
    def duration_seconds(self) -> float | None:
        """计算已运行时长 (秒)"""
        if self.started_at:
            end = self.completed_at or localnow()
            # MySQL DATETIME 不存时区，统一转为 naive 再计算
            try:
                s = self.started_at.replace(tzinfo=None) if self.started_at.tzinfo else self.started_at
                e = end.replace(tzinfo=None) if end.tzinfo else end
                return (e - s).total_seconds()
            except Exception:
                return None
        return None

    @property
    def duration_display(self) -> str:
        """人类可读的时长"""
        secs = self.duration_seconds
        if secs is None:
            return '未开始'
        if secs < 60:
            return f'{int(secs)}秒'
        if secs < 3600:
            return f'{int(secs / 60)}分{int(secs % 60)}秒'
        hours = int(secs / 3600)
        mins = int((secs % 3600) / 60)
        return f'{hours}小时{mins}分钟'

    @property
    def is_running(self) -> bool:
        return self.status == 'running'

    @property
    def is_finished(self) -> bool:
        return self.status in ('completed', 'failed', 'cancelled')

    @property
    def metrics_history(self) -> list:
        """获取指标历史 (安全反序列化, 损坏的 JSON 返回空列表)"""
        if self.metrics_history_json:
            try:
                return json.loads(self.metrics_history_json)
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    # ============ 方法 ============

    def start(self):
        """开始训练 (不提交 — 由调用方服务层控制)"""
        self.status = 'running'
        self.started_at = localnow()

    def update_progress(self, epoch: int, step: int, metrics: dict = None):
        """更新训练进度 (不提交 — 由调用方服务层控制)"""
        self.current_epoch = epoch
        self.current_step = step
        if self.total_steps > 0:
            self.progress_percent = round((step / self.total_steps) * 100, 1)

        if metrics:
            history = self.metrics_history
            history.append({'epoch': epoch, 'step': step, **metrics})
            self.metrics_history_json = json.dumps(history, ensure_ascii=False)
            self.final_metrics_json = json.dumps(metrics, ensure_ascii=False)

    def complete(self):
        """标记训练完成 (不提交 — 由调用方服务层控制)"""
        self.status = 'completed'
        self.progress_percent = 100.0
        self.completed_at = localnow()

    def fail(self, error_msg: str):
        """标记训练失败 (不提交 — 由调用方服务层控制)"""
        self.status = 'failed'
        self.error_message = error_msg
        self.completed_at = localnow()

    def cancel(self):
        """取消训练 (不提交 — 由调用方服务层控制)"""
        self.status = 'cancelled'
        self.completed_at = localnow()

    def append_log(self, message: str):
        """追加日志"""
        timestamp = localnow().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{timestamp}] {message}'
        self.log_text = (self.log_text or '') + line + '\n'

    def log_tail_last(self, n: int = 3) -> str:
        """获取最近 N 行日志 (SSE 事件推送给前端增量追加)"""
        text = self.log_text or ''
        if not text:
            return ''
        lines = text.strip().split('\n')
        return '\n'.join(lines[-n:]) + '\n' if lines and lines[-1] else '\n'.join(lines[-n:])

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'id': self.id,
            'uuid': self.uuid,
            'name': self.name,
            'description': self.description,
            'task_type': self.task_type,
            'framework': self.framework,
            'status': self.status,
            'progress_percent': self.progress_percent,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'gpu_count': self.gpu_count,
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'duration_seconds': self.duration_seconds,
            'duration_display': self.duration_display,
            'error_message': self.error_message,
            'owner_id': self.owner_id,
            'owner_name': self.owner.username if self.owner else None,
            'dataset_id': self.dataset_id,
            'dataset_name': self.dataset.name if self.dataset else None,
            'model_id': self.model_id,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f'<TrainingJob {self.name} ({self.status}) {self.progress_percent}%>'

    # ============ 权限检查 ============

    def is_viewable_by(self, user) -> bool:
        if user is None:
            return False
        return self.owner_id == user.id or user.is_admin

    def is_editable_by(self, user) -> bool:
        if user is None:
            return False
        return self.owner_id == user.id or user.is_admin
