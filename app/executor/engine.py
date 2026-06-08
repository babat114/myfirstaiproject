"""
训练执行引擎
基于 ThreadPoolExecutor 的轻量级训练调度器
"""
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone

from flask import current_app
from app import db, logger
from app.models.training_job import TrainingJob
from app.models.dataset import Dataset


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
        """获取训练任务的实时状态 (确保读取最新数据)"""
        # 刷新会话以读取训练线程的最新提交
        db.session.commit()
        db.session.expire_all()
        job = db.session.get(TrainingJob, job_id)
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

    # ============ 私有方法 ============

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
                framework = (job.framework or '').lower()
                dataset_category = (dataset.category or '').lower()
                algorithm = hyperparams.get('algorithm', '')

                # ── 纠错: KMeans 参数 algorithm=lloyd/elkan 会覆盖 ML 算法名 kmeans ──
                # 如果 hyperparams 里的 algorithm 是 KMeans 的参数值而非算法名, 自动修正
                _KMEANS_ALGO_PARAMS = {'lloyd', 'elkan', 'auto', 'full'}
                _KNOWN_CLUSTERING_ALGOS = {'kmeans', 'dbscan', 'agglomerative', 'minibatch_kmeans'}
                model_type = (getattr(job.model, 'model_type', '') or '').lower()
                task_type = hyperparams.get('task_type', hyperparams.get('ml_task_type', ''))
                if algorithm.lower() in _KMEANS_ALGO_PARAMS:
                    logger.warning(
                        f'检测到错误的algorithm值 "{algorithm}" (KMeans参数值), '
                        f'自动修正为 "kmeans"'
                    )
                    algorithm = 'kmeans'
                    # 同时修复存储的hyperparams, 防止下次再出错
                    hyperparams['algorithm'] = 'kmeans'
                    if job.model:
                        try:
                            job.model.hyperparameters_json = json.dumps(
                                hyperparams, ensure_ascii=False
                            )
                        except Exception:
                            pass

                # Transformer 迁移学习 — 仅当显式指定 algorithm 时启用
                # (NLP 数据集可能含预提取的特征向量如 TF-IDF，不适合 BERT tokenizer)
                if algorithm.startswith('transformer'):
                    from app.executor.trainers.transformers_nlp_trainer import TransformersNLPTrainer
                    trainer_cls = TransformersNLPTrainer
                # mlp 算法固定用 PyTorch (UI 标注为 "PyTorch MLP")
                elif algorithm == 'mlp':
                    from app.executor.trainers.pytorch_trainer import PyTorchTrainer
                    trainer_cls = PyTorchTrainer
                # 视觉数据 → PyTorch 深度学习
                elif dataset_category == 'vision':
                    from app.executor.trainers.pytorch_trainer import PyTorchTrainer
                    trainer_cls = PyTorchTrainer
                elif 'pytorch' in framework or 'torch' in framework:
                    from app.executor.trainers.pytorch_trainer import PyTorchTrainer
                    trainer_cls = PyTorchTrainer
                elif 'tensorflow' in framework or 'keras' in framework or 'tf' in framework:
                    from app.executor.trainers.keras_trainer import KerasTrainer
                    trainer_cls = KerasTrainer
                else:
                    from app.executor.trainers.sklearn_trainer import SklearnTrainer
                    trainer_cls = SklearnTrainer

                # 在工作线程中创建 trainer (对象绑定到此线程的 session)
                trainer = trainer_cls(job, dataset, hyperparams)
                self._active_trainers[job_id] = trainer

                logger.info(f'训练器已创建: {trainer_cls.__name__} (job {job_id})')
                trainer.run()
            except Exception as e:
                logger.error(f'训练任务 {job_id} 异常退出: {e}', exc_info=True)
                # 确保任务状态被标记为失败
                try:
                    job = db.session.get(TrainingJob, job_id)
                    if job and job.status not in ('completed', 'failed', 'cancelled'):
                        job.status = 'failed'
                        job.error_message = str(e)
                        job.append_log(f'[严重错误] 训练线程异常: {e}')
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
