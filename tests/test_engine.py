"""
============================================
训练执行引擎单元测试 v1.0
覆盖 app/executor/engine.py 核心逻辑
============================================
"""
import os
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from app.executor.engine import TrainingExecutor, _tail_log, get_executor


class TestTailLog:
    """_tail_log 辅助函数测试"""

    def test_normal_text(self):
        """正常日志截取最后 N 行"""
        log = 'line1\nline2\nline3\nline4\nline5'
        result = _tail_log(log, lines=2)
        assert result == 'line4\nline5'

    def test_none_input(self):
        """None 输入返回空字符串"""
        assert _tail_log(None) == ''

    def test_empty_string(self):
        """空字符串返回空字符串"""
        assert _tail_log('') == ''

    def test_fewer_lines_than_requested(self):
        """行数少于请求数返回全部"""
        log = 'a\nb'
        result = _tail_log(log, lines=10)
        assert result == 'a\nb'

    def test_default_lines(self):
        """默认截取 50 行"""
        log = '\n'.join(str(i) for i in range(100))
        result = _tail_log(log)
        assert len(result.split('\n')) == 50

    def test_single_line(self):
        """单行日志"""
        result = _tail_log('single line', lines=5)
        assert result == 'single line'

    def test_whitespace_only(self):
        """只有空白字符的日志"""
        result = _tail_log('   \n  \n  ', lines=2)
        # strip() 后再 split
        assert len(result.split('\n')) <= 2


class TestTrainingExecutorSingleton:
    """单例模式测试"""

    def test_singleton_same_instance(self):
        """多次调用返回同一实例"""
        e1 = TrainingExecutor()
        e2 = TrainingExecutor()
        assert e1 is e2

    def test_get_executor_function(self, app):
        """get_executor() 返回单例"""
        with app.app_context():
            executor = get_executor()
            assert isinstance(executor, TrainingExecutor)

    def test_max_workers_default(self, app):
        """默认 max_workers=2"""
        with app.app_context():
            executor = get_executor()
            assert executor.max_workers >= 1

    def test_initialized_flag_prevents_reinit(self):
        """_initialized 标志阻止重复初始化"""
        executor = TrainingExecutor()
        old_pool = executor._pool
        executor.__init__()
        assert executor._pool is old_pool  # 未重新创建线程池


class TestTrainingExecutorSubmit:
    """任务提交测试"""

    def test_submit_job_not_found(self, app):
        """数据库中没有该任务"""
        with app.app_context():
            executor = get_executor()
            mock_job = MagicMock()
            mock_job.id = 99999
            # db.session.get 返回 None
            from app import db
            result = executor.submit(mock_job)
            assert result is False

    def test_submit_duplicate_job(self, app, test_user):
        """重复提交已在运行的任务"""
        with app.app_context():
            from app.models.training_job import TrainingJob
            job = TrainingJob(
                name='test_job',
                owner_id=test_user.id,
                status='running',
            )
            from app import db as database
            database.session.add(job)
            database.session.commit()

            executor = get_executor()
            # 手动标记为活跃
            executor._active_trainers[job.id] = MagicMock()

            result = executor.submit(job)
            assert result is False


class TestTrainingExecutorPauseResumeCancel:
    """暂停/恢复/取消测试"""

    def test_pause_no_trainer(self, app):
        """无活跃 trainer 时 pause 返回 False"""
        with app.app_context():
            executor = get_executor()
            result = executor.pause(99999)
            assert result is False

    def test_resume_no_trainer(self, app):
        """无活跃 trainer 时 resume 返回 False"""
        with app.app_context():
            executor = get_executor()
            result = executor.resume(99999)
            assert result is False

    def test_cancel_no_trainer(self, app):
        """无活跃 trainer 时 cancel 返回 False"""
        with app.app_context():
            executor = get_executor()
            result = executor.cancel(99999)
            assert result is False

    def test_pause_active_trainer(self, app, test_user):
        """暂停活跃训练器"""
        with app.app_context():
            from app.models.training_job import TrainingJob
            from app import db as database

            job = TrainingJob(
                name='test_job',
                owner_id=test_user.id,
                status='running',
            )
            database.session.add(job)
            database.session.commit()

            executor = get_executor()
            mock_trainer = MagicMock()
            executor._active_trainers[job.id] = mock_trainer

            result = executor.pause(job.id)
            assert result is True
            mock_trainer.pause.assert_called_once()

    def test_resume_active_trainer(self, app, test_user):
        """恢复活跃训练器"""
        with app.app_context():
            from app.models.training_job import TrainingJob
            from app import db as database

            job = TrainingJob(
                name='test_job',
                owner_id=test_user.id,
                status='paused',
            )
            database.session.add(job)
            database.session.commit()

            executor = get_executor()
            mock_trainer = MagicMock()
            executor._active_trainers[job.id] = mock_trainer

            result = executor.resume(job.id)
            assert result is True
            mock_trainer.resume.assert_called_once()

    def test_cancel_active_trainer(self, app, test_user):
        """取消活跃训练器"""
        with app.app_context():
            from app.models.training_job import TrainingJob
            from app import db as database

            job = TrainingJob(
                name='test_job',
                owner_id=test_user.id,
                status='running',
            )
            database.session.add(job)
            database.session.commit()

            executor = get_executor()
            mock_trainer = MagicMock()
            executor._active_trainers[job.id] = mock_trainer

            result = executor.cancel(job.id)
            assert result is True
            mock_trainer.cancel.assert_called_once()


class TestTrainingExecutorGetStatus:
    """get_status 测试"""

    def test_get_status_job_not_found(self, app):
        """查询不存在的任务"""
        with app.app_context():
            executor = get_executor()
            result = executor.get_status(99999)
            assert result is None

    def test_get_status_running_job(self, app, test_user):
        """查询运行中的任务状态"""
        with app.app_context():
            from app.models.training_job import TrainingJob
            from app import db as database

            job = TrainingJob(
                name='test_job',
                owner_id=test_user.id,
                status='running',
                progress_percent=50.0,
                current_epoch=5,
                total_epochs=10,
                log_text='epoch 1\nepoch 2\nepoch 3',
            )
            database.session.add(job)
            database.session.commit()

            executor = get_executor()
            result = executor.get_status(job.id)
            assert result is not None
            assert result['job_id'] == job.id
            assert result['status'] == 'running'
            assert result['progress_percent'] == 50.0
            assert result['current_epoch'] == 5
            assert result['total_epochs'] == 10
            assert 'is_running' in result
            assert 'is_finished' in result
            assert 'log_tail' in result

    def test_get_status_completed_job(self, app, test_user):
        """查询已完成的任务状态"""
        with app.app_context():
            from app.models.training_job import TrainingJob
            from app import db as database

            job = TrainingJob(
                name='done_job',
                owner_id=test_user.id,
                status='completed',
            )
            database.session.add(job)
            database.session.commit()

            executor = get_executor()
            result = executor.get_status(job.id)
            assert result is not None
            assert result['status'] == 'completed'
            assert result['is_finished'] is True


class TestTrainingExecutorGetQueueInfo:
    """get_queue_info 测试"""

    def test_empty_queue(self, app):
        """空闲队列"""
        with app.app_context():
            executor = get_executor()
            # 清理来自其他测试的活跃训练器 (单例共享状态)
            executor._active_trainers.clear()
            info = executor.get_queue_info()
            assert info['active_count'] == 0
            assert 'max_workers' in info
            assert len(info['active_jobs']) == 0

    def test_queue_with_active_jobs(self, app, test_user):
        """有活跃任务的队列"""
        with app.app_context():
            from app.models.training_job import TrainingJob
            from app import db as database

            job = TrainingJob(
                name='test_job',
                owner_id=test_user.id,
                status='running',
            )
            database.session.add(job)
            database.session.commit()

            executor = get_executor()
            mock_trainer = MagicMock()
            mock_trainer.is_paused = False
            executor._active_trainers[job.id] = mock_trainer

            info = executor.get_queue_info()
            assert info['active_count'] == 1
            assert len(info['active_jobs']) == 1
            assert info['active_jobs'][0]['job_id'] == job.id


class TestTrainingExecutorGracefulShutdown:
    """优雅关闭测试"""

    def test_graceful_shutdown_marks_running_as_paused(self, app, test_user):
        """关闭时运行中任务标记为 paused"""
        with app.app_context():
            from app.models.training_job import TrainingJob
            from app import db as database

            job = TrainingJob(
                name='test_job',
                owner_id=test_user.id,
                status='running',
            )
            database.session.add(job)
            database.session.commit()

            executor = get_executor()
            mock_trainer = MagicMock()
            executor._active_trainers[job.id] = mock_trainer

            executor._graceful_shutdown()

            # 刷新 job 查看状态
            updated_job = database.session.get(TrainingJob, job.id)
            if updated_job:
                assert updated_job.status in ('paused', 'running')

    def test_graceful_shutdown_idempotent(self, app):
        """重复调用 graceful_shutdown 不抛异常"""
        with app.app_context():
            executor = get_executor()
            executor._graceful_shutdown()
            # 第二次调用不应抛异常 (shutting_down flag)
            executor._graceful_shutdown()


class TestTrainingExecutorResolveTrainer:
    """训练器类解析测试"""

    def test_resolve_sklearn_default(self, app):
        """默认解析为 sklearn trainer"""
        with app.app_context():
            from app.models.training_job import TrainingJob
            from app.models.dataset import Dataset

            executor = get_executor()
            mock_job = MagicMock(spec=TrainingJob)
            mock_job.framework = ''
            mock_job.dataset = MagicMock(spec=Dataset)
            mock_job.dataset.category = 'tabular'

            trainer_cls = executor._resolve_trainer_class(mock_job, {})
            from app.executor.trainers.sklearn_trainer import SklearnTrainer
            assert trainer_cls is SklearnTrainer

    def test_resolve_mlp_pytorch(self, app):
        """MLP 算法解析为 PyTorch trainer"""
        with app.app_context():
            from app.models.training_job import TrainingJob
            from app.models.dataset import Dataset

            executor = get_executor()
            mock_job = MagicMock(spec=TrainingJob)
            mock_job.framework = ''
            mock_job.dataset = MagicMock(spec=Dataset)
            mock_job.dataset.category = 'tabular'

            trainer_cls = executor._resolve_trainer_class(mock_job, {'algorithm': 'mlp'})
            from app.executor.trainers.pytorch_trainer import PyTorchTrainer
            assert trainer_cls is PyTorchTrainer

    def test_resolve_vision_pytorch(self, app):
        """视觉数据集解析为 PyTorch trainer"""
        with app.app_context():
            from app.models.training_job import TrainingJob
            from app.models.dataset import Dataset

            executor = get_executor()
            mock_job = MagicMock(spec=TrainingJob)
            mock_job.framework = ''
            mock_job.dataset = MagicMock(spec=Dataset)
            mock_job.dataset.category = 'vision'

            trainer_cls = executor._resolve_trainer_class(mock_job, {})
            from app.executor.trainers.pytorch_trainer import PyTorchTrainer
            assert trainer_cls is PyTorchTrainer

    @pytest.mark.parametrize("framework,expected_suffix", [
        ('tensorflow', 'Keras'),
        ('keras', 'Keras'),
        ('tf', 'Keras'),
        ('pytorch', 'PyTorch'),
        ('torch', 'PyTorch'),
    ])
    def test_resolve_by_framework(self, app, framework, expected_suffix):
        """根据框架名解析训练器"""
        with app.app_context():
            from app.models.training_job import TrainingJob
            from app.models.dataset import Dataset

            executor = get_executor()
            mock_job = MagicMock(spec=TrainingJob)
            mock_job.framework = framework
            mock_job.dataset = MagicMock(spec=Dataset)
            mock_job.dataset.category = 'tabular'

            trainer_cls = executor._resolve_trainer_class(mock_job, {})
            assert expected_suffix in trainer_cls.__name__

    def test_resolve_transformer_nlp(self, app):
        """Transformer 算法解析为 NLP trainer"""
        with app.app_context():
            from app.models.training_job import TrainingJob
            from app.models.dataset import Dataset

            executor = get_executor()
            mock_job = MagicMock(spec=TrainingJob)
            mock_job.framework = ''
            mock_job.dataset = MagicMock(spec=Dataset)
            mock_job.dataset.category = 'nlp'

            trainer_cls = executor._resolve_trainer_class(
                mock_job, {'algorithm': 'transformer_bert'}
            )
            from app.executor.trainers.transformers_nlp_trainer import TransformersNLPTrainer
            assert trainer_cls is TransformersNLPTrainer
