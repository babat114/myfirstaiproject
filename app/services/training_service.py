"""
============================================
训练任务服务
管理AI模型训练任务的生命周期
============================================
"""

import contextlib
import json

from sqlalchemy.orm import joinedload

from app import db, logger
from app.models.dataset import Dataset
from app.models.model_record import ModelRecord
from app.models.training_job import TrainingJob
from app.models.user import User
from app.utils.cache import dashboard_cache
from app.utils.helpers import paginate_query, sanitize_service_error


class TrainingService:
    """训练任务管理服务"""

    @staticmethod
    def create_job(
        user: User,
        name: str,
        dataset_id: int = None,
        description: str = None,
        task_type: str = 'training',
        framework: str = None,
        total_epochs: int = 0,
        total_steps: int = 0,
        gpu_count: int = 0,
        cpu_cores: int = 1,
        memory_gb: float = 4.0,
        hyperparameters: dict = None,
        ml_task_type: str = 'classification',
        algorithm: str = 'random_forest',
        target_column: str = None,
        test_size: float = 0.2,
        model_type: str = None,
    ) -> tuple[TrainingJob | None, str | None]:
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
        full_hyperparams.update(
            {
                'task_type': ml_task_type,
                'algorithm': algorithm,
                'target_column': target_column,
                'test_size': test_size,
            }
        )

        try:
            # 创建关联的模型记录 — 使用增强描述 (应用场景+使用方式+算法原理)
            from app.services.model_recommender import generate_enhanced_description

            dataset_name = dataset.name if dataset_id and dataset else ''
            model_description = generate_enhanced_description(
                dataset_name=dataset_name,
                task_type=ml_task_type,
                algorithm=algorithm,
                target_column=target_column or '',
                model_name=name,
                class_labels=[],
                feature_names=[],
            )

            model = ModelRecord(
                name=f'{name} - 模型',
                description=model_description,
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
            dashboard_cache.invalidate('job_stats:')
            dashboard_cache.invalidate('dashboard:')

            logger.info(f'训练任务创建: {name} (by {user.username}), 算法: {algorithm}, 类型: {ml_task_type}')
            return job, None

        except Exception as e:
            db.session.rollback()
            return None, sanitize_service_error(e, '创建训练任务失败')

    @staticmethod
    def start_job(job: TrainingJob) -> tuple[bool, str | None]:
        """启动训练任务 — 提交到训练执行引擎"""
        if job.status not in ('queued', 'paused'):
            return False, f'当前状态 ({job.status}) 无法启动训练。'

        try:
            from app.executor.engine import get_executor

            # 如果是首次启动 (非暂停恢复)，清空旧数据
            if job.status == 'queued':
                job.log_text = None
                job.metrics_history_json = None
                job.final_metrics_json = None
                job.error_message = None
                db.session.commit()

            executor = get_executor()
            success = executor.submit(job)
            if not success:
                return False, '任务提交失败，可能已在运行中。'

            logger.info(f'训练任务已提交到执行引擎: {job.name}')
            return True, None
        except Exception as e:
            return False, str(e)

    @staticmethod
    def pause_job(job: TrainingJob) -> tuple[bool, str | None]:
        """暂停训练任务"""
        if job.status != 'running':
            return False, '只能暂停正在运行的任务。'

        from app.executor.engine import get_executor

        executor = get_executor()
        if executor.pause(job.id):
            return True, None
        return False, '暂停失败，任务可能不在活跃列表中。'

    @staticmethod
    def resume_job(job: TrainingJob) -> tuple[bool, str | None]:
        """恢复训练任务"""
        if job.status != 'paused':
            return False, '只能恢复已暂停的任务。'

        from app.executor.engine import get_executor

        executor = get_executor()
        if executor.resume(job.id):
            return True, None
        return False, '恢复失败，任务可能不在活跃列表中。'

    @staticmethod
    def complete_job(job: TrainingJob) -> tuple[bool, str | None]:
        """完成训练 (由训练线程自动调用，一般不手动触发)"""
        if job.status != 'running':
            return False, '只能完成正在运行的任务。'

        job.complete()
        job.append_log('训练任务已完成')
        db.session.commit()
        dashboard_cache.invalidate('job_stats:')
        dashboard_cache.invalidate('dashboard:')
        logger.info(f'训练完成: {job.name}')
        return True, None

    @staticmethod
    def fail_job(job: TrainingJob, error: str) -> tuple[bool, str | None]:
        """标记训练失败"""
        job.fail(error)
        job.append_log(f'训练失败: {error}')
        db.session.commit()
        dashboard_cache.invalidate('job_stats:')
        dashboard_cache.invalidate('dashboard:')
        logger.error(f'训练失败: {job.name} - {error}')
        return True, None

    @staticmethod
    def cancel_job(job: TrainingJob) -> tuple[bool, str | None]:
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
        dashboard_cache.invalidate('job_stats:')
        dashboard_cache.invalidate('dashboard:')
        logger.info(f'训练取消: {job.name}')
        return True, None

    @staticmethod
    def _reset_job_to_queued(job: TrainingJob):
        """重置训练任务为 queued 状态 (清空进度/日志/指标) — 减轻 retrain 重复代码"""
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
        # created_at 保留原始创建时间, 不覆盖

    @staticmethod
    def retrain_job(job: TrainingJob, new_params: dict = None) -> tuple[bool, str | None]:
        """重新训练 — 可选使用新参数, 重置任务状态为 queued 并提交到执行引擎

        适用状态: failed, cancelled, completed, paused
        操作: 清空进度/日志/指标, 重置状态为 queued, 清空错误信息

        Args:
            job: 原 TrainingJob
            new_params: 新超参数字典 (None = 使用原参数重训)
        """
        if job.status in ('running', 'queued'):
            return False, f'任务已在 {job.status} 状态，无需重新训练。'

        try:
            from app.executor.engine import get_executor

            # ── 新参数模式: 更新模型超参数 ──
            if new_params and job.model:
                from app.utils.algorithm_helpers import fix_kmeans_algorithm

                try:
                    existing = job.model.hyperparameters_dict
                    existing.update(new_params)
                    curr_algo = existing.get('algorithm', '')
                    fix_kmeans_algorithm(curr_algo, existing, job.model)
                    job.model.hyperparameters_json = json.dumps(existing, ensure_ascii=False)
                    if 'ml_task_type' in new_params:
                        job.model.model_type = new_params['ml_task_type']
                    # 应用新参数到训练任务字段
                    if 'epochs' in new_params or 'total_epochs' in new_params:
                        job.total_epochs = new_params.get('total_epochs', new_params.get('epochs', 10))
                    if 'framework' in new_params:
                        job.framework = new_params['framework']
                except Exception as e:
                    logger.warning(f'更新模型超参数失败: {e}')

            # ── 同步 model_type → task_type + algorithm (两种模式共用) ──
            if job.model:
                try:
                    hp = job.model.hyperparameters_dict
                    model_type = job.model.model_type
                    updated = False
                    if model_type == 'clustering' and hp.get('task_type') != 'clustering':
                        hp['task_type'] = 'clustering'
                        hp['algorithm'] = 'kmeans'
                        updated = True
                    elif model_type == 'regression' and hp.get('task_type') != 'regression':
                        hp['task_type'] = 'regression'
                        updated = True
                    if updated:
                        job.model.hyperparameters_json = json.dumps(hp, ensure_ascii=False)
                        job.append_log(f'[重训] 已同步 task_type={hp["task_type"]} algorithm={hp.get("algorithm")}')
                except Exception:
                    pass

            # ── 清理并重置 (共用) ──
            TrainingService._reset_job_to_queued(job)
            log_msg = (
                f'[重训] 使用新参数重置: {json.dumps(new_params, ensure_ascii=False)}'
                if new_params
                else '[重训] 任务重置，准备重新训练...'
            )
            job.append_log(log_msg)
            db.session.commit()

            executor = get_executor()
            success = executor.submit(job)
            if not success:
                return False, '重新训练提交失败，请稍后重试。'

            logger.info(f'训练任务已重置并提交: {job.name} (id={job.id}), 新参数: {bool(new_params)}')
            return True, None
        except Exception as e:
            db.session.rollback()
            return False, sanitize_service_error(e, '重新训练失败')

    @staticmethod
    def get_job_status(job_id: int) -> dict | None:
        """获取训练任务的实时运行状态 (含进度/日志)"""
        from app.executor.engine import get_executor

        executor = get_executor()
        return executor.get_status(job_id)

    @staticmethod
    def update_progress(job: TrainingJob, epoch: int, step: int, metrics: dict = None) -> tuple[bool, str | None]:
        """更新训练进度"""
        if job.status != 'running':
            return False, '任务未在运行中。'

        job.update_progress(epoch, step, metrics)
        db.session.commit()
        return True, None

    @staticmethod
    def get_job_by_id(job_id: int) -> TrainingJob | None:
        """根据 ID 获取任务 (预加载关联对象, 避免模板中 DetachedInstanceError)"""
        from sqlalchemy.orm import joinedload

        return db.session.execute(
            db.select(TrainingJob)
            .filter_by(id=job_id)
            .options(joinedload(TrainingJob.owner), joinedload(TrainingJob.dataset), joinedload(TrainingJob.model))
        ).scalar_one_or_none()

    @staticmethod
    def get_job_by_uuid(job_uuid: str) -> TrainingJob | None:
        """根据 UUID 获取任务 (预加载关联对象, 避免 N+1)"""
        from sqlalchemy.orm import joinedload

        return db.session.execute(
            db.select(TrainingJob)
            .filter_by(uuid=job_uuid)
            .options(joinedload(TrainingJob.owner), joinedload(TrainingJob.dataset), joinedload(TrainingJob.model))
        ).scalar_one_or_none()

    @staticmethod
    def delete_job(job: TrainingJob) -> tuple[bool, str | None]:
        """删除训练任务"""
        try:
            db.session.delete(job)
            db.session.commit()
            dashboard_cache.invalidate('job_stats:')
            dashboard_cache.invalidate('dashboard:')
            logger.info(f'训练任务已删除: {job.name}')
            return True, None
        except Exception as e:
            db.session.rollback()
            return False, sanitize_service_error(e, '删除训练任务失败')

    @staticmethod
    def list_jobs(
        page: int = 1,
        per_page: int = 15,
        status: str = None,
        task_type: str = None,
        owner_id: int = None,
        dataset_id: int = None,
        search: str = None,
    ) -> dict:
        """
        获取训练任务列表

        Returns:
            分页结果字典
        """
        query = TrainingJob.query.options(
            joinedload(TrainingJob.owner),
            joinedload(TrainingJob.dataset),
            joinedload(TrainingJob.model),  # 预加载模型信息, 避免 N+1
        )

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

        return paginate_query(query, page, per_page, transform_fn=lambda x: x.to_dict())

    # 超参数键名冲突集合 — 合并 best_params 时必须跳过 (如 KMeans 的 algorithm=lloyd 覆盖 algorithm=kmeans)
    _TUNING_CONFLICT_KEYS = {'algorithm', 'ml_task_type', 'task_type', 'framework'}

    @staticmethod
    def apply_tuning_hyperparameters(
        job: TrainingJob, best_params: dict, best_score: float, search_time: float = None
    ) -> bool:
        """将调优结果写入训练任务关联模型 (封装 DB 写入, 供路由层调用)

        Returns:
            是否成功
        """
        try:
            if job.model:
                hp = job.model.hyperparameters_dict
                hp['tuning_result'] = {
                    'best_params': best_params,
                    'best_cv_score': best_score,
                    'search_time': search_time,
                }
                job.model.set_hyperparameters(hp)
                db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            logger.error(f'应用调优结果失败: {e}')
            return False

    @staticmethod
    def parse_extra_hyperparams(form_data: dict) -> dict:
        """从表单数据提取防过拟合超参数 (PyTorch/TensorFlow 可选参数)

        集中处理路由层重复的数值类型转换 + hidden_layers 逗号分隔解析。
        """
        extra = {}
        numeric_fields = {
            'val_size': float,
            'dropout': float,
            'learning_rate': float,
            'weight_decay': float,
            'batch_size': int,
            'early_stopping_patience': int,
        }
        for field, cast in numeric_fields.items():
            raw = (form_data.get(field) or '').strip()
            if raw:
                with contextlib.suppress(ValueError, TypeError):
                    extra[field] = cast(raw)

        hl_raw = (form_data.get('hidden_layers_str') or '').strip()
        if hl_raw:
            try:
                layers = [int(x.strip()) for x in hl_raw.split(',') if x.strip()]
                if layers:
                    extra['hidden_layers'] = layers
            except (ValueError, TypeError):
                pass
        return extra

    @staticmethod
    def build_retrain_params(
        best_params: dict, existing_hparams: dict, algorithm: str, ml_task_type: str, is_mlp: bool = False
    ) -> dict:
        """从调优结果构建重训练参数 — 跳过冲突键, 保留既有参数

        解决 gridsearch_retrain / apply_tuning_result 中的重复合并逻辑。
        """
        retrain = {
            'algorithm': algorithm,
            'ml_task_type': ml_task_type,
            'framework': 'pytorch' if is_mlp else existing_hparams.get('framework', 'sklearn'),
        }
        for k, v in best_params.items():
            if k not in TrainingService._TUNING_CONFLICT_KEYS:
                retrain[k] = v
        for k in ('hidden_layers', 'dropout', 'test_size'):
            if k in existing_hparams and k not in retrain:
                retrain[k] = existing_hparams[k]
        return retrain

    @staticmethod
    def get_job_statistics(user_id: int = None) -> dict:
        """获取训练任务统计 (使用 SQL GROUP BY 聚合, 避免全量加载)"""
        cache_key = f'job_stats:{user_id or "all"}'
        cached = dashboard_cache.get(cache_key)
        if cached is not None:
            return cached

        from sqlalchemy import func

        # 辅助: 构建带可选 user_id 过滤的查询
        def _filtered_query(*cols):
            q = db.select(*cols)
            if user_id:
                q = q.filter_by(owner_id=user_id)
            return q

        # 按状态聚合
        status_rows = db.session.execute(
            _filtered_query(TrainingJob.status, func.count(TrainingJob.id)).group_by(TrainingJob.status)
        ).all()
        status_counts = {row[0]: row[1] for row in status_rows}

        # 按任务类型聚合
        type_rows = db.session.execute(
            _filtered_query(TrainingJob.task_type, func.count(TrainingJob.id)).group_by(TrainingJob.task_type)
        ).all()
        type_counts = {row[0]: row[1] for row in type_rows}

        total_count = sum(status_counts.values())

        # 平均完成时间 — SQL AVG (仅 completed 任务, 移除 tzinfo 后计算差值)
        avg_duration = None
        try:
            avg_row = db.session.execute(
                _filtered_query(
                    func.avg(func.timestampdiff(db.text('SECOND'), TrainingJob.started_at, TrainingJob.completed_at))
                ).filter(
                    TrainingJob.status == 'completed',
                    TrainingJob.started_at.isnot(None),
                    TrainingJob.completed_at.isnot(None),
                )
            ).scalar()
            if avg_row:
                avg_duration = round(float(avg_row), 1)
        except Exception:
            # SQLite 不支持 TIMESTAMPDIFF, 回退到 Python 计算
            try:
                completed_jobs = db.session.execute(
                    _filtered_query(TrainingJob.started_at, TrainingJob.completed_at).filter(
                        TrainingJob.status == 'completed',
                        TrainingJob.started_at.isnot(None),
                        TrainingJob.completed_at.isnot(None),
                    )
                ).all()
                durations = []
                for started_at, completed_at in completed_jobs:
                    if started_at and completed_at:
                        delta = (completed_at - started_at).total_seconds()
                        if delta >= 0:
                            durations.append(delta)
                if durations:
                    avg_duration = round(float(sum(durations) / len(durations)), 1)
            except Exception:
                avg_duration = None

        running = status_counts.get('running', 0)
        queued = status_counts.get('queued', 0)
        paused = status_counts.get('paused', 0)

        result = {
            'total_count': total_count,
            'running_count': running,
            'queued_count': queued,
            'paused_count': paused,
            'active_or_paused_count': running + queued + status_counts.get('preparing', 0) + paused,
            'completed_count': status_counts.get('completed', 0),
            'failed_count': status_counts.get('failed', 0),
            'cancelled_count': status_counts.get('cancelled', 0),
            'statuses': status_counts,
            'types': type_counts,
            'avg_completion_seconds': avg_duration,
        }
        dashboard_cache.set(cache_key, result)
        return result
