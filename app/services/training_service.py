"""
============================================
训练任务服务
管理AI模型训练任务的生命周期
============================================
"""
import json
from datetime import datetime
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
                   hyperparameters: dict = None) -> Tuple[Optional[TrainingJob], Optional[str]]:
        """
        创建新训练任务

        Returns:
            (TrainingJob, error_message)
        """
        # 验证数据集
        if dataset_id:
            dataset = Dataset.query.get(dataset_id)
            if not dataset:
                return None, '指定的数据集不存在。'
            if dataset.status != 'ready':
                return None, '数据集尚未就绪，无法开始训练。'

        try:
            job = TrainingJob(
                name=name,
                description=description,
                task_type=task_type,
                framework=framework,
                total_epochs=total_epochs,
                total_steps=total_steps,
                gpu_count=gpu_count,
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                owner_id=user.id,
                dataset_id=dataset_id,
                status='queued',
            )

            db.session.add(job)
            db.session.commit()

            logger.info(f"训练任务创建: {name} (by {user.username})")
            return job, None

        except Exception as e:
            db.session.rollback()
            logger.error(f"创建训练任务失败: {e}")
            return None, str(e)

    @staticmethod
    def start_job(job: TrainingJob) -> Tuple[bool, Optional[str]]:
        """启动训练任务"""
        if job.status not in ('queued', 'paused'):
            return False, f'当前状态 ({job.status}) 无法启动训练。'

        try:
            job.start()
            job.append_log('训练任务已启动')
            logger.info(f"训练任务启动: {job.name}")
            return True, None
        except Exception as e:
            return False, str(e)

    @staticmethod
    def pause_job(job: TrainingJob) -> Tuple[bool, Optional[str]]:
        """暂停训练任务"""
        if job.status != 'running':
            return False, '只能暂停正在运行的任务。'

        job.status = 'paused'
        job.append_log('训练任务已暂停')
        db.session.commit()
        return True, None

    @staticmethod
    def resume_job(job: TrainingJob) -> Tuple[bool, Optional[str]]:
        """恢复训练任务"""
        if job.status != 'paused':
            return False, '只能恢复已暂停的任务。'

        job.status = 'running'
        job.append_log('训练任务已恢复')
        db.session.commit()
        return True, None

    @staticmethod
    def complete_job(job: TrainingJob) -> Tuple[bool, Optional[str]]:
        """完成训练"""
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

        job.cancel()
        job.append_log('训练任务已取消')
        logger.info(f"训练取消: {job.name}")
        return True, None

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
        return TrainingJob.query.get(job_id)

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
