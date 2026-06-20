"""
训练执行引擎
基于 ThreadPoolExecutor 的轻量级训练调度器
"""
import os
import json
import atexit
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone

from flask import current_app
from app import db, logger
from app.models.training_job import TrainingJob
from app.models.dataset import Dataset
from app.models.model_record import ModelRecord


class TrainingExecutor:
    """
    训练执行引擎 (单例模式 — 全局唯一调度器)

    使用 ThreadPoolExecutor 并发执行训练任务:
    - max_workers=2: 同时最多2个训练任务并发
    - 每个训练任务运行在独立线程中 (通过 _run_wrapper)
    - 支持暂停/恢复/取消 (通过 threading.Event 信号)

    线程安全设计:
    - 只传递 job_id 和 dataset_id (int) 到工作线程
    - 工作线程内重新加载 SQLAlchemy 对象 (避免跨线程 DetachedInstanceError)
    - 工作线程结束后清理 session (db.session.remove())

    使用方式:
        executor = get_executor()          # 获取全局单例
        executor.submit(job)               # 提交训练任务
        executor.pause(job_id)             # 暂停训练
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # 从 Flask 配置读取 max_workers (默认 2)
        try:
            from flask import current_app
            self.max_workers = current_app.config.get('TRAINING_MAX_WORKERS', 2)
        except RuntimeError:
            self.max_workers = 2

        self._pool = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix='trainer-')
        self._active_trainers: dict[int, object] = {}  # {job_id: trainer_instance}
        self._futures: dict[int, Future] = {}           # {job_id: Future}
        self._app = None  # 延迟绑定 Flask app
        self._shutdown_registered = False

        # 注册进程退出时的清理钩子 — 确保线程池正常关闭, 训练状态不会卡在 running
        atexit.register(self._graceful_shutdown)

        logger.info(f'TrainingExecutor 初始化完成 (max_workers={self.max_workers})')

    def _get_app(self):
        """获取 Flask 应用实例"""
        if self._app is not None:
            return self._app
        try:
            from flask import current_app
            self._app = current_app._get_current_object()
            return self._app
        except RuntimeError:
            return None

    # ============ 公共接口 ============

    def submit(self, job: TrainingJob) -> bool:
        """
        提交训练任务到线程池

        Args:
            job: TrainingJob 实例

        Returns:
            是否成功提交
        """
        # ── 一次性恢复僵尸任务: 将上次异常退出遗留的 'running' 任务标记为 'interrupted' ──
        if not getattr(self, '_zombies_recovered', False):
            self._zombies_recovered = True
            try:
                stale = TrainingJob.query.filter_by(status='running').all()
                for j in stale:
                    j.status = 'interrupted'
                    j.error_message = '服务异常退出，训练中断。请重新训练。'
                    j.append_log('[系统] 检测到上次异常退出，训练已中断。')
                if stale:
                    db.session.commit()
                    logger.info(f'已恢复 {len(stale)} 个僵尸训练任务 (running→interrupted)')
            except Exception as e:
                try:
                    db.session.rollback()
                except Exception:
                    pass
                logger.warning(f'僵尸任务恢复失败 (非致命): {e}')

        # 从数据库刷新 job 以避免 DetachedInstanceError
        fresh_job = db.session.get(TrainingJob, job.id)
        if not fresh_job:
            logger.error(f'无法从数据库加载任务 {job.id}')
            return False

        if fresh_job.id in self._active_trainers:
            logger.warning(f'任务 {fresh_job.id} 已在运行中')
            return False

        # 获取关联数据集 (仅验证存在性)
        dataset = fresh_job.dataset
        if not dataset:
            logger.error(f'任务 {fresh_job.id} 没有关联数据集')
            return False

        if not dataset.file_path or not os.path.exists(dataset.file_path):
            logger.error(f'任务 {fresh_job.id} 的数据集文件不存在: {dataset.file_path}')
            return False

        # 缓存 app 引用供工作线程使用
        if self._app is None:
            self._get_app()

        # 只传递 job_id 和 dataset_id，在工作线程中重新加载对象
        # 避免跨线程 SQLAlchemy DetachedInstanceError
        dataset_id = dataset.id
        job_id = fresh_job.id

        # 提交到线程池 — trainer 在工作线程中创建
        future = self._pool.submit(self._run_wrapper, job_id, dataset_id)
        self._futures[job_id] = future
        future.add_done_callback(lambda f: self._on_done(job_id, f))

        logger.info(f'训练任务已提交: {job_id} ({fresh_job.name}), '
                    f'活跃任务: {len(self._active_trainers) + 1}')
        return True

    def pause(self, job_id: int) -> bool:
        """暂停训练"""
        trainer = self._active_trainers.get(job_id)
        if trainer:
            trainer.pause()
            job = db.session.get(TrainingJob, job_id)
            if job and job.status == 'running':
                job.status = 'paused'
                job.append_log('[暂停] 训练已暂停')
                db.session.commit()
            logger.info(f'训练已暂停: {job_id}')
            return True
        return False

    def resume(self, job_id: int) -> bool:
        """恢复训练"""
        trainer = self._active_trainers.get(job_id)
        if trainer:
            trainer.resume()
            job = db.session.get(TrainingJob, job_id)
            if job and job.status == 'paused':
                job.status = 'running'
                job.append_log('[恢复] 训练已恢复')
                db.session.commit()
            logger.info(f'训练已恢复: {job_id}')
            return True
        return False

    def cancel(self, job_id: int) -> bool:
        """取消训练"""
        trainer = self._active_trainers.get(job_id)
        if trainer:
            trainer.cancel()
            logger.info(f'训练已取消: {job_id}')
            return True
        return False

    def get_status(self, job_id: int) -> dict | None:
        """获取训练任务的实时状态 (refresh 直接重新查询, 无 commit 副作用)"""
        job = db.session.get(TrainingJob, job_id)
        if job:
            db.session.refresh(job)  # 从数据库重新加载最新数据
        if not job:
            return None

        full_history = job.metrics_history
        return {
            'job_id': job.id,
            'uuid': job.uuid,
            'status': job.status,
            'progress_percent': job.progress_percent,
            'current_epoch': job.current_epoch,
            'total_epochs': job.total_epochs,
            'current_step': job.current_step,
            'total_steps': job.total_steps,
            'duration_display': job.duration_display,
            'metrics_history': full_history,        # 完整历史 → 前端增量对比
            'final_metrics': job.final_metrics_json,
            'error_message': job.error_message,
            'log_tail': _tail_log(job.log_text, 100),  # 最近100行
            'log_full': job.log_text or '',             # 全量日志(用于首次填充)
            'is_finished': job.is_finished,
            'is_running': job.is_running,
        }

    def get_queue_info(self) -> dict:
        """获取队列信息"""
        return {
            'active_count': len(self._active_trainers),
            'max_workers': self.max_workers,
            'active_jobs': [
                {
                    'job_id': jid,
                    'name': db.session.get(TrainingJob, jid).name if db.session.get(TrainingJob, jid) else 'unknown',
                    'is_paused': t.is_paused if hasattr(t, 'is_paused') else False,
                }
                for jid, t in self._active_trainers.items()
            ],
        }

    def _graceful_shutdown(self):
        """应用退出时优雅关闭线程池 — 标记运行中任务为 paused, 等待线程完成

        注意: atexit 时 DB 连接池可能已关闭, 无法更新任务状态。
        此时仅安全关闭训练线程池; 下次启动时引擎会自动将 running→interrupted。
        """
        if getattr(self, '_shutting_down', False):
            return
        self._shutting_down = True

        # 尝试更新 DB 中运行任务的状态 (容忍连接已关闭)
        _db_available = False
        try:
            from sqlalchemy import text
            db.session.execute(text('SELECT 1'))
            _db_available = True
        except Exception:
            pass

        if _db_available:
            for jid in list(self._active_trainers.keys()):
                try:
                    job = db.session.get(TrainingJob, jid)
                    if job and job.status == 'running':
                        job.status = 'paused'
                        job.append_log('[系统] 服务关闭，训练已暂停。重启后可恢复。')
                        db.session.commit()
                except Exception:
                    try:
                        db.session.rollback()
                    except Exception:
                        pass

        # 关闭线程池 (容忍 logger 已关闭的情况)
        try:
            self._pool.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass

    # ============ 私有方法 ============

    def _resolve_trainer_class(self, job, hyperparams):
        """根据框架/算法/数据集类别解析训练器类 (从 _run_wrapper 抽取)"""
        framework = (job.framework or '').lower()
        dataset_category = (job.dataset.category or '').lower() if job.dataset else ''
        algorithm = hyperparams.get('algorithm', '')

        # ── 纠错: KMeans 参数 algorithm=lloyd/elkan 会覆盖 ML 算法名 kmeans ──
        from app.utils.algorithm_helpers import fix_kmeans_algorithm
        algorithm = fix_kmeans_algorithm(algorithm, hyperparams, job.model)

        # Transformer 迁移学习
        if algorithm.startswith('transformer'):
            from app.executor.trainers.transformers_nlp_trainer import TransformersNLPTrainer
            return TransformersNLPTrainer
        # mlp → PyTorch
        if algorithm == 'mlp':
            from app.executor.trainers.pytorch_trainer import PyTorchTrainer
            return PyTorchTrainer
        # 视觉数据 → PyTorch
        if dataset_category == 'vision':
            from app.executor.trainers.pytorch_trainer import PyTorchTrainer
            return PyTorchTrainer
        if 'pytorch' in framework or 'torch' in framework:
            from app.executor.trainers.pytorch_trainer import PyTorchTrainer
            return PyTorchTrainer
        if 'tensorflow' in framework or 'keras' in framework or 'tf' in framework:
            from app.executor.trainers.keras_trainer import KerasTrainer
            return KerasTrainer
        # 默认 sklearn
        from app.executor.trainers.sklearn_trainer import SklearnTrainer
        return SklearnTrainer

    def _run_wrapper(self, job_id: int, dataset_id: int):
        """在线程中运行训练的包装器 (推送 Flask 应用上下文)

        在 worker 线程内重新加载 job/dataset 并创建 trainer，
        避免跨线程传递 SQLAlchemy 对象导致 DetachedInstanceError。
        """
        app = self._get_app()
        if app is None:
            logger.error(f'训练任务 {job_id}: 无法获取 Flask app, 训练终止')
            return

        with app.app_context():
            trainer = None
            try:
                logger.info(f'训练线程启动: job_id={job_id}')

                # 在工作线程中重新加载 job 和 dataset
                job = db.session.get(TrainingJob, job_id)
                if not job:
                    logger.error(f'训练任务 {job_id}: 数据库中不存在')
                    return

                dataset = db.session.get(Dataset, dataset_id)
                if not dataset:
                    logger.error(f'训练任务 {job_id}: 数据集 {dataset_id} 不存在')
                    return

                # 解析超参数
                hyperparams = {}
                if job.model and job.model.hyperparameters_json:
                    try:
                        hyperparams = json.loads(job.model.hyperparameters_json)
                    except (json.JSONDecodeError, TypeError):
                        pass

                # 决定使用哪个训练器
                trainer_cls = self._resolve_trainer_class(job, hyperparams)

                # 在工作线程中创建 trainer (对象绑定到此线程的 session)
                trainer = trainer_cls(job, dataset, hyperparams)
                self._active_trainers[job_id] = trainer

                logger.info(f'训练器已创建: {trainer_cls.__name__} (job {job_id})')
                trainer.run()
            except Exception as e:
                logger.error(f'训练任务 {job_id} 异常退出: {e}', exc_info=True)
                # 确保任务 + 关联模型被标记为失败
                try:
                    job = db.session.get(TrainingJob, job_id)
                    if job and job.status not in ('completed', 'failed', 'cancelled'):
                        job.status = 'failed'
                        job.error_message = str(e)
                        job.append_log(f'[严重错误] 训练线程异常: {e}')
                        # 同步标记关联的 ModelRecord 为失败 (避免孤立 draft 记录)
                        if job.model_id:
                            model = db.session.get(ModelRecord, job.model_id)
                            if model and model.status == 'draft':
                                model.status = 'failed'
                        db.session.commit()
                except Exception as db_err:
                    logger.error(f'无法更新任务状态: {db_err}')
            finally:
                # 清理线程的数据库会话
                try:
                    db.session.remove()
                except Exception:
                    pass

    def _on_done(self, job_id: int, future: Future):
        """训练完成/失败后的清理"""
        self._active_trainers.pop(job_id, None)
        self._futures.pop(job_id, None)

        try:
            future.result()  # 如果有未捕获异常这里会抛出
        except Exception as e:
            logger.error(f'训练任务 {job_id} 执行失败: {e}')

        logger.info(f'训练任务 {job_id} 已清理, 剩余活跃: {len(self._active_trainers)}')


def _tail_log(log_text: str | None, lines: int = 50) -> str:
    """获取日志的最后 N 行"""
    if not log_text:
        return ''
    all_lines = log_text.strip().split('\n')
    return '\n'.join(all_lines[-lines:])


# ============ 全局单例获取 ============

def get_executor() -> 'TrainingExecutor':
    """获取全局 TrainingExecutor 单例 (委托给 __new__ 的双重检查锁)"""
    return TrainingExecutor()
