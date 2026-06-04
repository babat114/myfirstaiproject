"""
训练回调系统
在训练过程中更新数据库进度、日志和指标
训练完成后自动保存实验记录到 experiments/{job_uuid}/
"""
import json
import os
from datetime import datetime, timezone
from app import db, logger
from app.models.training_job import TrainingJob


class TrainingCallback:
    """训练回调 — 每个 epoch 后更新 TrainingJob 状态"""

    def __init__(self, job_id: int):
        self.job_id = job_id

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
        record = {'epoch': epoch + 1, 'timestamp': datetime.now(timezone.utc).isoformat()}
        record.update(metrics)
        history.append(record)
        job.metrics_history_json = json.dumps(history, ensure_ascii=False)
        job.final_metrics_json = json.dumps(metrics, ensure_ascii=False)

        db.session.commit()

    def on_log(self, message: str):
        """追加训练日志"""
        job = db.session.get(TrainingJob, self.job_id)
        if not job:
            return
        job.append_log(message)
        db.session.commit()

    def _utcnow(self):
        """返回 naive UTC datetime (MySQL DATETIME 不存时区)"""
        return datetime.now(timezone.utc).replace(tzinfo=None)

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
        job.started_at = self._utcnow()
        job.append_log(f'[启动] 训练任务开始')
        job.error_message = None
        db.session.commit()

    def on_complete(self, final_metrics: dict = None):
        """训练完成时调用"""
        job = db.session.get(TrainingJob, self.job_id)
        if not job:
            return
        job.status = 'completed'
        job.progress_percent = 100.0
        job.completed_at = self._utcnow()
        if final_metrics:
            job.final_metrics_json = json.dumps(final_metrics, ensure_ascii=False)
        job.append_log(f'[完成] 训练成功完成')

        # 更新关联模型的指标和训练时长
        if job.model:
            try:
                if final_metrics:
                    from app.models.model_record import ModelRecord
                    model = db.session.get(ModelRecord, job.model.id)
                    if model:
                        metrics_to_set = {}
                        # 优先提取 macro 指标 (各类别等权, 能暴露类别间差异)
                        for key in ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'loss'):
                            for prefix in ('test_', 'train_', ''):
                                full_key = f'{prefix}{key}'
                                if full_key in final_metrics:
                                    metrics_to_set[key] = final_metrics[full_key]
                                    break
                        # 如果 macro 指标不存在, 回退到旧版键名 (weighted 或裸键)
                        if 'precision_macro' not in metrics_to_set:
                            for key in ('precision', 'precision_weighted'):
                                for prefix in ('test_', 'train_', ''):
                                    full_key = f'{prefix}{key}'
                                    if full_key in final_metrics:
                                        metrics_to_set['precision_macro'] = final_metrics[full_key]
                                        break
                                if 'precision_macro' in metrics_to_set:
                                    break
                        if 'recall_macro' not in metrics_to_set:
                            for key in ('recall', 'recall_weighted'):
                                for prefix in ('test_', 'train_', ''):
                                    full_key = f'{prefix}{key}'
                                    if full_key in final_metrics:
                                        metrics_to_set['recall_macro'] = final_metrics[full_key]
                                        break
                                if 'recall_macro' in metrics_to_set:
                                    break
                        if 'f1_macro' not in metrics_to_set:
                            for key in ('f1_score', 'f1_weighted'):
                                for prefix in ('test_', 'train_', ''):
                                    full_key = f'{prefix}{key}'
                                    if full_key in final_metrics:
                                        metrics_to_set['f1_macro'] = final_metrics[full_key]
                                        break
                                if 'f1_macro' in metrics_to_set:
                                    break
                        if metrics_to_set:
                            model.set_metrics(metrics_to_set)
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
                        if os.path.exists(exp_model_pkl):
                            model.model_file_path = exp_model_pkl
                        elif os.path.exists(exp_model_pt):
                            model.model_file_path = exp_model_pt
                        job.append_log(f'[模型] 已更新关联模型指标')
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
        job.completed_at = self._utcnow()
        job.append_log(f'[错误] {error}')
        db.session.commit()

    def on_cancel(self):
        """训练取消时调用"""
        job = db.session.get(TrainingJob, self.job_id)
        if not job:
            return
        job.status = 'cancelled'
        job.completed_at = self._utcnow()
        job.append_log(f'[取消] 训练任务已取消')
        db.session.commit()
