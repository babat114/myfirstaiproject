"""
============================================
训练任务服务
管理AI模型训练任务的生命周期
============================================
"""
import json
from datetime import datetime, timezone
from typing import Optional, Tuple
from app import db, logger
from app.models.training_job import TrainingJob
from app.models.dataset import Dataset
from app.models.model_record import ModelRecord
from app.models.user import User


class TrainingService:
    """训练任务管理服务"""

    @staticmethod
    def create_job(user: User, name: str, dataset_id: int = None,
                   description: str = None, task_type: str = 'training',
                   framework: str = None, total_epochs: int = 0,
                   total_steps: int = 0, gpu_count: int = 0,
                   cpu_cores: int = 1, memory_gb: float = 4.0,
                   hyperparameters: dict = None,
                   ml_task_type: str = 'classification',
                   algorithm: str = 'random_forest',
                   target_column: str = None,
                   test_size: float = 0.2,
                   model_type: str = None) -> Tuple[Optional[TrainingJob], Optional[str]]:
        """
        创建新训练任务

        Args:
            user: 当前用户
            name: 任务名称
            dataset_id: 数据集 ID
            description: 任务描述
            task_type: 任务类型 (training/fine_tuning/evaluation)
            framework: 框架 (sklearn/pytorch)
            total_epochs: 总 epoch 数
            total_steps: 总 step 数
            gpu_count: GPU 数量
            cpu_cores: CPU 核心数
            memory_gb: 内存 (GB)
            hyperparameters: 超参数字典
            ml_task_type: 机器学习任务类型 (classification/regression)
            algorithm: 算法名称
            target_column: 目标列名
            test_size: 测试集比例

        Returns:
            (TrainingJob, error_message)
        """
        # 验证数据集
        if dataset_id:
            dataset = db.session.get(Dataset, dataset_id)
            if not dataset:
                return None, '指定的数据集不存在。'
            if dataset.status != 'ready':
                return None, '数据集尚未就绪，无法开始训练。'

        # 构建完整的超参数字典
        full_hyperparams = hyperparameters or {}
        full_hyperparams.update({
            'task_type': ml_task_type,
            'algorithm': algorithm,
            'target_column': target_column,
            'test_size': test_size,
        })

        try:
            # 创建关联的模型记录
            model = ModelRecord(
                name=f'{name} - 模型',
                description=f'为训练任务 "{name}" 自动创建的模型记录',
                model_type=model_type or ml_task_type,
                framework=framework or 'sklearn',
                owner_id=user.id,
                status='draft',
            )
            model.set_hyperparameters(full_hyperparams)
            db.session.add(model)
            db.session.flush()  # 获取 model.id

            job = TrainingJob(
                name=name,
                description=description,
                task_type=task_type,
                framework=framework or 'sklearn',
                total_epochs=total_epochs,
                total_steps=total_steps,
                gpu_count=gpu_count,
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                owner_id=user.id,
                dataset_id=dataset_id,
                model_id=model.id,
                status='queued',
            )

            db.session.add(job)
            db.session.commit()

            logger.info(f"训练任务创建: {name} (by {user.username}), "
                        f"算法: {algorithm}, 类型: {ml_task_type}")
            return job, None

        except Exception as e:
            db.session.rollback()
            logger.error(f"创建训练任务失败: {e}")
            return None, str(e)

    @staticmethod
    def start_job(job: TrainingJob) -> Tuple[bool, Optional[str]]:
        """启动训练任务 — 提交到训练执行引擎"""
        if job.status not in ('queued', 'paused'):
            return False, f'当前状态 ({job.status}) 无法启动训练。'

        try:
            from app.executor.engine import get_executor

            # 如果是首次启动 (非暂停恢复)
            if job.status == 'queued':
                job.status = 'queued'
                job.log_text = None
                job.metrics_history_json = None
                job.final_metrics_json = None
                job.error_message = None
                db.session.commit()

            executor = get_executor()
            success = executor.submit(job)
            if not success:
                return False, '任务提交失败，可能已在运行中。'

            logger.info(f"训练任务已提交到执行引擎: {job.name}")
            return True, None
        except Exception as e:
            return False, str(e)

    @staticmethod
    def pause_job(job: TrainingJob) -> Tuple[bool, Optional[str]]:
        """暂停训练任务"""
        if job.status != 'running':
            return False, '只能暂停正在运行的任务。'

        from app.executor.engine import get_executor
        executor = get_executor()
        if executor.pause(job.id):
            return True, None
        return False, '暂停失败，任务可能不在活跃列表中。'

    @staticmethod
    def resume_job(job: TrainingJob) -> Tuple[bool, Optional[str]]:
        """恢复训练任务"""
        if job.status != 'paused':
            return False, '只能恢复已暂停的任务。'

        from app.executor.engine import get_executor
        executor = get_executor()
        if executor.resume(job.id):
            return True, None
        return False, '恢复失败，任务可能不在活跃列表中。'

    @staticmethod
    def complete_job(job: TrainingJob) -> Tuple[bool, Optional[str]]:
        """完成训练 (由训练线程自动调用，一般不手动触发)"""
        if job.status != 'running':
            return False, '只能完成正在运行的任务。'

        job.complete()
        job.append_log('训练任务已完成')
        logger.info(f"训练完成: {job.name}")
        return True, None

    @staticmethod
    def fail_job(job: TrainingJob, error: str) -> Tuple[bool, Optional[str]]:
        """标记训练失败"""
        job.fail(error)
        job.append_log(f'训练失败: {error}')
        logger.error(f"训练失败: {job.name} - {error}")
        return True, None

    @staticmethod
    def cancel_job(job: TrainingJob) -> Tuple[bool, Optional[str]]:
        """取消训练"""
        if job.is_finished:
            return False, '任务已结束，无法取消。'

        from app.executor.engine import get_executor
        executor = get_executor()
        if executor.cancel(job.id):
            return True, None

        # 如果不在活跃列表 (比如还在 queued)，直接标记取消
        job.cancel()
        job.append_log('训练任务已取消')
        db.session.commit()
        logger.info(f"训练取消: {job.name}")
        return True, None

    @staticmethod
    def retrain_job(job: TrainingJob) -> Tuple[bool, Optional[str]]:
        """重新训练 — 重置任务状态为 queued 并提交到执行引擎

        适用状态: failed, cancelled, completed, paused
        操作: 清空进度/日志/指标, 重置状态为 queued, 清空错误信息
        """
        if job.status in ('running', 'queued'):
            return False, f'任务已在 {job.status} 状态，无需重新训练。'

        try:
            from app.executor.engine import get_executor

            # 清理旧数据
            job.status = 'queued'
            job.progress_percent = 0.0
            job.current_epoch = 0
            job.total_epochs = job.total_epochs or 10
            job.current_step = 0
            job.total_steps = 0
            job.log_text = None
            job.metrics_history_json = None
            job.final_metrics_json = None
            job.error_message = None
            job.started_at = None
            job.completed_at = None
            job.append_log(f'[重训] 任务重置，准备重新训练...')
            db.session.commit()

            # 提交到执行引擎
            executor = get_executor()
            success = executor.submit(job)
            if not success:
                return False, '重新训练提交失败，请稍后重试。'

            logger.info(f"训练任务已重置并提交: {job.name} (id={job.id})")
            return True, None
        except Exception as e:
            db.session.rollback()
            return False, str(e)

    @staticmethod
    def retrain_job_with_params(job: TrainingJob,
                                 new_params: dict) -> Tuple[bool, Optional[str]]:
        """使用新参数重新训练 — 更新超参数后重置并启动

        Args:
            job: 原 TrainingJob
            new_params: 新超参数字典, 可包含:
                - learning_rate, batch_size, epochs, hidden_layers,
                  dropout, test_size, weight_decay, algorithm,
                  ml_task_type, target_column, framework

        自动更新关联 ModelRecord 的 hyperparameters_json
        """
        if job.status in ('running', 'queued'):
            return False, f'任务正在 {job.status} 状态，请先等待完成或取消后再重试。'

        try:
            from app.executor.engine import get_executor

            # 更新关联模型的超参数
            if job.model:
                try:
                    existing = job.model.hyperparameters_dict
                    existing.update(new_params)
                    job.model.hyperparameters_json = json.dumps(
                        existing, ensure_ascii=False
                    )
                    # 同步更新关键字段
                    if 'ml_task_type' in new_params:
                        job.model.model_type = new_params['ml_task_type']
                    if 'algorithm' in new_params:
                        pass  # algorithm 在 model 中没有独立字段
                except Exception as e:
                    logger.warning(f'更新模型超参数失败: {e}')

            # 应用新参数到训练任务
            if 'epochs' in new_params or 'total_epochs' in new_params:
                job.total_epochs = new_params.get('total_epochs', new_params.get('epochs', 10))
            if 'framework' in new_params:
                job.framework = new_params['framework']

            # 清理并重置
            job.status = 'queued'
            job.progress_percent = 0.0
            job.current_epoch = 0
            job.current_step = 0
            job.total_steps = 0
            job.log_text = None
            job.metrics_history_json = None
            job.final_metrics_json = None
            job.error_message = None
            job.started_at = None
            job.completed_at = None
            job.append_log(f'[重训] 使用新参数重置: {json.dumps(new_params, ensure_ascii=False)}')
            db.session.commit()

            executor = get_executor()
            success = executor.submit(job)
            if not success:
                return False, '重新训练提交失败，请稍后重试。'

            logger.info(f"训练任务已使用新参数重置: {job.name}, 参数: {new_params}")
            return True, None
        except Exception as e:
            db.session.rollback()
            return False, str(e)

    @staticmethod
    def get_job_status(job_id: int) -> dict | None:
        """获取训练任务的实时运行状态 (含进度/日志)"""
        from app.executor.engine import get_executor
        executor = get_executor()
        return executor.get_status(job_id)

    @staticmethod
    def update_progress(job: TrainingJob, epoch: int, step: int,
                        metrics: dict = None) -> Tuple[bool, Optional[str]]:
        """更新训练进度"""
        if job.status != 'running':
            return False, '任务未在运行中。'

        job.update_progress(epoch, step, metrics)
        return True, None

    @staticmethod
    def get_job_by_id(job_id: int) -> Optional[TrainingJob]:
        """根据 ID 获取任务"""
        return db.session.get(TrainingJob, job_id)

    @staticmethod
    def get_job_by_uuid(job_uuid: str) -> Optional[TrainingJob]:
        """根据 UUID 获取任务"""
        return TrainingJob.query.filter_by(uuid=job_uuid).first()

    @staticmethod
    def delete_job(job: TrainingJob) -> Tuple[bool, Optional[str]]:
        """删除训练任务"""
        try:
            db.session.delete(job)
            db.session.commit()
            logger.info(f"训练任务已删除: {job.name}")
            return True, None
        except Exception as e:
            db.session.rollback()
            return False, str(e)

    @staticmethod
    def list_jobs(page: int = 1, per_page: int = 15,
                  status: str = None, task_type: str = None,
                  owner_id: int = None, dataset_id: int = None,
                  search: str = None) -> dict:
        """
        获取训练任务列表

        Returns:
            分页结果字典
        """
        query = TrainingJob.query

        if status:
            query = query.filter_by(status=status)
        if task_type:
            query = query.filter_by(task_type=task_type)
        if owner_id:
            query = query.filter_by(owner_id=owner_id)
        if dataset_id:
            query = query.filter_by(dataset_id=dataset_id)
        if search:
            term = f'%{search}%'
            query = query.filter(
                db.or_(
                    TrainingJob.name.ilike(term),
                    TrainingJob.description.ilike(term),
                )
            )

        query = query.order_by(TrainingJob.created_at.desc())

        pagination = query.paginate(
            page=page, per_page=per_page, error_out=False
        )

        return {
            'items': [j.to_dict() for j in pagination.items],
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev,
        }

    @staticmethod
    def get_job_statistics(user_id: int = None) -> dict:
        """获取训练任务统计"""
        query = TrainingJob.query
        if user_id:
            query = query.filter_by(owner_id=user_id)

        jobs = query.all()

        status_counts = {}
        type_counts = {}
        for j in jobs:
            status_counts[j.status] = status_counts.get(j.status, 0) + 1
            type_counts[j.task_type] = type_counts.get(j.task_type, 0) + 1

        completed = [j for j in jobs if j.status == 'completed']
        avg_duration = None
        if completed:
            durations = [
                (j.completed_at - j.started_at).total_seconds()
                for j in completed
                if j.started_at and j.completed_at
            ]
            if durations:
                avg_duration = sum(durations) / len(durations)

        return {
            'total_count': len(jobs),
            'running_count': status_counts.get('running', 0),
            'queued_count': status_counts.get('queued', 0),
            'completed_count': status_counts.get('completed', 0),
            'failed_count': status_counts.get('failed', 0),
            'statuses': status_counts,
            'types': type_counts,
            'avg_completion_seconds': round(avg_duration, 1) if avg_duration else None,
        }
