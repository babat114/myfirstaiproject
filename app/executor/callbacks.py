"""
训练回调系统
在训练过程中更新数据库进度、日志和指标
训练完成后自动保存实验记录到 experiments/{job_uuid}/
"""
import json
import os
import pickle
from datetime import datetime
from app import db, logger
from app.models.training_job import TrainingJob
from app._timezone import localnow


class TrainingCallback:
    """训练回调 — 每个 epoch 后更新 TrainingJob 状态，同时发布事件到事件总线"""

    def __init__(self, job_id: int):
        self.job_id = job_id

    def _publish(self, event_type: str, data: dict):
        """发布事件到事件总线 (非阻塞，总线不可用时静默丢弃)"""
        try:
            from app.utils.event_bus import get_event_bus
            get_event_bus().publish(self.job_id, event_type, data)
        except Exception:
            pass  # 事件总线故障不应影响训练

    def on_epoch_end(self, epoch: int, total_epochs: int, metrics: dict):
        """每个 epoch 结束时调用"""
        job = db.session.get(TrainingJob, self.job_id)
        if not job:
            return

        job.current_epoch = epoch + 1
        job.total_epochs = total_epochs

        if total_epochs > 0:
            job.progress_percent = round(((epoch + 1) / total_epochs) * 100, 1)

        # 追加指标历史
        history = job.metrics_history
        record = {'epoch': epoch + 1, 'timestamp': localnow().isoformat()}
        record.update(metrics)
        history.append(record)
        job.metrics_history_json = json.dumps(history, ensure_ascii=False)
        job.final_metrics_json = json.dumps(metrics, ensure_ascii=False)

        db.session.commit()

        # 推送实时事件 — 携带完整 history 让前端直接渲染 (无需额外 HTTP 轮询)
        self._publish('metrics', {
            'current_epoch': job.current_epoch,
            'total_epochs': job.total_epochs,
            'progress_percent': job.progress_percent,
            'metrics': metrics,
            'metrics_history': history,          # 完整历史 → 前端直接渲染图表+关键帧
            'log_tail': job.log_tail_last(3),    # 最近3行日志 → 前端增量追加
        })

    def on_log(self, message: str):
        """追加训练日志"""
        job = db.session.get(TrainingJob, self.job_id)
        if not job:
            return
        job.append_log(message)
        db.session.commit()

        # 推送日志事件
        self._publish('log', {'message': message})

    def _localnow(self):
        """返回 naive datetime 用于 MySQL DATETIME 列 (北京时间 UTC+8, 去时区)"""
        return localnow().replace(tzinfo=None)

    def _safe_timedelta(self, started, completed) -> int:
        """安全计算两个 datetime 的秒数差 (兼容 aware/naive)"""
        if not started or not completed:
            return 0
        try:
            s = started.replace(tzinfo=None) if started.tzinfo else started
            c = completed.replace(tzinfo=None) if completed.tzinfo else completed
            return int((c - s).total_seconds())
        except Exception:
            return 0

    def on_start(self):
        """训练开始时调用"""
        job = db.session.get(TrainingJob, self.job_id)
        if not job:
            return
        job.status = 'running'
        job.started_at = self._localnow()
        job.append_log(f'[启动] 训练任务开始')
        job.error_message = None
        db.session.commit()
        self._publish('status_change', {'status': 'running', 'message': '训练任务开始'})

    def on_complete(self, final_metrics: dict = None):
        """训练完成时调用"""
        job = db.session.get(TrainingJob, self.job_id)
        if not job:
            return
        job.status = 'completed'
        job.progress_percent = 100.0
        job.completed_at = self._localnow()
        if final_metrics:
            job.final_metrics_json = json.dumps(final_metrics, ensure_ascii=False)
        job.append_log(f'[完成] 训练成功完成')
        self._publish('status_change', {'status': 'completed', 'message': '训练成功完成'})
        if final_metrics:
            self._publish('metrics', {'final_metrics': final_metrics})

        # 更新关联模型的指标和训练时长
        if job.model:
            try:
                if final_metrics:
                    from app.models.model_record import ModelRecord
                    model = db.session.get(ModelRecord, job.model.id)
                    if model:
                        # 委托给 ModelRecord.set_metrics() — 自动检测分类/回归/聚类
                        model.set_metrics(final_metrics)
                        # 同时保存完整的评估指标到 metrics_json
                        model.metrics_json = json.dumps(final_metrics, ensure_ascii=False)
                        model.status = 'trained'
                        model.training_duration_seconds = self._safe_timedelta(
                            job.started_at, job.completed_at
                        )
                        model.training_job_id = job.id
                        model.training_dataset_id = job.dataset_id
                        # 设置模型文件路径为实验目录下的模型文件
                        exp_model_pkl = os.path.join('experiments', job.uuid, 'model.pkl')
                        exp_model_pt = os.path.join('experiments', job.uuid, 'model.pt')
                        exp_model_keras = os.path.join('experiments', job.uuid, 'model.keras')
                        if os.path.exists(exp_model_pkl):
                            model.model_file_path = exp_model_pkl
                        elif os.path.exists(exp_model_pt):
                            model.model_file_path = exp_model_pt
                        elif os.path.exists(exp_model_keras):
                            model.model_file_path = exp_model_keras
                        job.append_log(f'[模型] 已更新关联模型指标')

                        # ---- 自动独立测试集评估 ----
                        if final_metrics and job.dataset_id:
                            try:
                                from app.models.dataset import Dataset as DsModel
                                ind_test = DsModel.query.filter_by(
                                    source_dataset_id=job.dataset_id,
                                    is_test_set=True,
                                    status='ready'
                                ).first()
                                if ind_test:
                                    from app.services.inference_service import ModelInferenceService
                                    ind_result = ModelInferenceService.test_model_with_split(
                                        model, test_dataset=ind_test
                                    )
                                    if ind_result.get('success'):
                                        ind_metrics = {
                                            'ind_test_accuracy': ind_result.get('accuracy'),
                                            'ind_test_f1_macro': ind_result.get('f1_macro'),
                                            'ind_test_f1_weighted': ind_result.get('f1_weighted'),
                                            'test_dataset_name': ind_test.name,
                                            'test_dataset_uuid': ind_test.uuid,
                                            'collection_method': ind_test.collection_method,
                                        }
                                        model.set_independent_metrics(ind_metrics)
                                        model.independent_test_dataset_id = ind_test.id
                                        job.append_log(
                                            f'[独立评估] 独立测试集 "{ind_test.name}" 评估完成: '
                                            f'accuracy={ind_result.get("accuracy")}'
                                        )
                                    else:
                                        job.append_log(
                                            f'[独立评估] 评估失败: {ind_result.get("error", "未知错误")}'
                                        )
                            except Exception as e:
                                job.append_log(f'[独立评估] 自动评估异常 (非致命): {e}')

                        # 回填实际 sklearn 模型参数到 hyperparameters_json
                        if job.framework == 'sklearn' and model.model_file_path:
                            try:
                                _backfill_sklearn_params_to_model(model, job)
                            except Exception as e:
                                logger.warning(f'回填sklearn参数失败 model_id={model.id}: {e}')
                                job.append_log(f'[参数] 捕获sklearn参数失败: {e}')
            except Exception as e:
                logger.error(f'更新模型指标失败: {e}')
                job.append_log(f'[警告] 更新模型指标失败: {e}')

        db.session.commit()

        # 保存实验记录到 experiments/{job_uuid}/
        self._save_experiment(job, final_metrics)

    def _save_experiment(self, job, final_metrics: dict = None):
        """保存实验记录到本地 JSON 文件 (PRD 第五节)"""
        try:
            exp_dir = os.path.join('experiments', job.uuid)
            os.makedirs(exp_dir, exist_ok=True)

            # 1. config.json — 超参数快照
            config = {
                'job_name': job.name,
                'task_type': job.task_type,
                'framework': job.framework,
                'total_epochs': job.total_epochs,
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            }
            if job.model:
                config['hyperparameters'] = job.model.hyperparameters_dict
                config['model_type'] = job.model.model_type
            if job.dataset:
                config['dataset_name'] = job.dataset.name
                config['dataset_rows'] = job.dataset.row_count
                config['dataset_columns'] = job.dataset.column_count

            with open(os.path.join(exp_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            # 2. metrics.json — epoch 级别指标数组
            metrics_data = {
                'history': job.metrics_history,
                'final': final_metrics or {},
            }
            with open(os.path.join(exp_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, ensure_ascii=False, indent=2)

            # 3. training.log — 纯文本日志
            if job.log_text:
                with open(os.path.join(exp_dir, 'training.log'), 'w', encoding='utf-8') as f:
                    f.write(job.log_text)

            job.append_log(f'[实验] 记录已保存到 {exp_dir}/')
            db.session.commit()
            logger.info(f'实验记录已保存: {exp_dir}')

        except Exception as e:
            logger.error(f'保存实验记录失败: {e}')
            job.append_log(f'[警告] 保存实验记录失败: {e}')
            db.session.commit()

    def on_error(self, error: str):
        """训练出错时调用"""
        job = db.session.get(TrainingJob, self.job_id)
        if not job:
            return
        job.status = 'failed'
        job.error_message = error
        job.completed_at = self._localnow()
        job.append_log(f'[错误] {error}')
        db.session.commit()
        self._publish('status_change', {'status': 'failed', 'message': error})

    def on_cancel(self):
        """训练取消时调用"""
        job = db.session.get(TrainingJob, self.job_id)
        if not job:
            return
        job.status = 'cancelled'
        job.completed_at = self._localnow()
        job.append_log(f'[取消] 训练任务已取消')
        db.session.commit()
        self._publish('status_change', {'status': 'cancelled', 'message': '训练任务已取消'})


# ═══════════════════════════════════════════════════════════════
# 训练完成后捕获实际 sklearn 模型参数
# ═══════════════════════════════════════════════════════════════

def _backfill_sklearn_params_to_model(model, job):
    """从保存的 .pkl 文件中提取 sklearn 估计器的实际参数，
    合并到 ModelRecord.hyperparameters_json 中。

    使 DB 中的超参数记录不再只是 {task_type, algorithm, target_column, test_size}，
    而是包含实际的 sklearn 模型参数（如 n_estimators, max_depth, C 等）。
    """
    pkl_path = model.model_file_path
    if not pkl_path or not os.path.exists(pkl_path):
        return

    with open(pkl_path, 'rb') as f:
        bundle = pickle.load(f)

    estimator = bundle.get('model') if isinstance(bundle, dict) else getattr(bundle, 'model', None)
    if estimator is None or not hasattr(estimator, 'get_params'):
        return

    actual_params = estimator.get_params()

    # 获取原始用户设定参数
    hp = model.hyperparameters_dict if hasattr(model, 'hyperparameters_dict') else {}
    if not hp:
        hp = json.loads(model.hyperparameters_json) if model.hyperparameters_json else {}

    # 构建合并后的超参数字典
    merged = {
        'task_type': hp.get('task_type', job.task_type or ''),
        'algorithm': hp.get('algorithm', ''),
        'target_column': hp.get('target_column', None),
        'test_size': hp.get('test_size', 0.2),
    }

    # 保留用户显式传入的非基础参数 (如来自 ParameterGuidanceService 的数据感知参数)
    for k, v in hp.items():
        if k not in merged and k not in ('epochs', 'batch_size', 'val_size',
                                          'random_state', 'ml_task_type',
                                          'actual_params', 'param_source',
                                          'backfilled_at', 'tuned',
                                          'tuning_method', 'best_cv_score',
                                          'tuning_result', 'best_params'):
            merged[k] = v

    # 过滤掉不应作为超参数的 meta keys
    filtered = {}
    for k, v in actual_params.items():
        if k == 'random_state':
            continue  # random_state 是固定种子，不是调优参数
        # 跳过 sklearn 内部对象（如 criterion 函数、class_weight dict 转字符串等）
        if callable(v):
            continue
        if hasattr(v, '__module__'):
            try:
                json.dumps({k: v})
            except (TypeError, ValueError):
                continue
        filtered[k] = v

    merged['actual_params'] = filtered

    # 判断参数来源
    from app.executor.trainers.sklearn_trainer import SklearnTrainer
    defaults = SklearnTrainer._REGULARIZE_DEFAULTS.get(hp.get('algorithm', ''), {})
    if hp.get('tuned'):
        merged['param_source'] = 'hyperparameter_tuning'
    elif defaults and any(
        str(filtered.get(k)) == str(v) for k, v in defaults.items()
    ):
        merged['param_source'] = 'regularize_defaults'
    elif 'algorithm_params' in hp:
        merged['param_source'] = 'user_specified'
    else:
        merged['param_source'] = 'sklearn_defaults'

    merged['backfilled_at'] = localnow().isoformat()

    model.set_hyperparameters(merged)
    job.append_log(
        f'[参数] 已捕获实际sklearn模型参数 '
        f'({len(filtered)} keys, source={merged["param_source"]})'
    )
