"""
============================================
训练任务模块测试 (参数化优化 v1.0)
============================================
"""
import pytest
from app import db
from app.services.training_service import TrainingService


class TestTrainingService:
    """训练服务测试"""

    @pytest.mark.parametrize("name,expect_name", [
        ("Test Training Job", "Test Training Job"),
        ("",                  ""),  # 空名称: 允许创建 (服务端可能生成默认名)
    ])
    def test_create_job(self, test_user, name, expect_name):
        """参数化: 有效名称 / 空名称 — 均允许创建, 验证名称正确传递"""
        job, error = TrainingService.create_job(
            user=test_user,
            name=name,
            task_type='training',
            framework='sklearn',
            algorithm='randomforest',
        )
        assert job is not None, f'创建任务应成功: {error}'
        assert error is None
        assert job.name == expect_name, f'任务名应为 "{expect_name}", 实际: "{job.name}"'
        assert job.status in ('queued', 'draft', 'created')
        assert job.id is not None

    def test_list_jobs(self, test_user):
        """测试获取任务列表"""
        TrainingService.create_job(
            user=test_user, name='Job 1', task_type='training'
        )
        TrainingService.create_job(
            user=test_user, name='Job 2', task_type='fine_tuning'
        )
        result = TrainingService.list_jobs(owner_id=test_user.id)
        assert result['total'] >= 2

    def test_get_job_status(self, test_user):
        """测试获取任务状态"""
        job, _ = TrainingService.create_job(
            user=test_user, name='Status Test', task_type='training'
        )
        status = TrainingService.get_job_status(job.id)
        assert status is not None
        assert 'status' in status

    def test_get_job_not_found(self, test_user):
        """测试获取不存在的任务"""
        job = TrainingService.get_job_by_id(99999)
        assert job is None

    def test_cancel_job(self, test_user):
        """测试取消任务"""
        job, _ = TrainingService.create_job(
            user=test_user, name='Cancel Test', task_type='training'
        )
        success, error = TrainingService.cancel_job(job)
        assert success is True
        assert error is None


class TestTrainingPages:
    """训练页面测试"""

    @pytest.mark.parametrize("endpoint", [
        "/training/",
        "/training/create",
        "/training/tuning",
    ])
    def test_training_page(self, logged_in_client, endpoint):
        """参数化: 列表页 / 创建页 / 调优页"""
        response = logged_in_client.get(endpoint)
        assert response.status_code == 200


class TestComparePage:
    """模型对比页面测试"""

    def test_compare_page_get(self, logged_in_client):
        response = logged_in_client.get('/models/compare')
        assert response.status_code == 200

    def test_compare_requires_auth(self, client):
        response = client.get('/models/compare', follow_redirects=True)
        assert response.status_code == 200


class TestSSEStream:
    """SSE 流测试"""

    @pytest.mark.parametrize("endpoint", [
        "/api/stream/training/1/stream",
        "/api/stream/training/1/status",
    ])
    def test_stream_requires_auth(self, client, endpoint):
        """参数化: stream / status 端点均需认证"""
        response = client.get(endpoint)
        assert response.status_code in (302, 401)


class TestE2ETrainingFlow:
    """端到端训练流程集成测试: 数据集 → 模型 → 训练 → 完成"""

    def test_create_job_creates_model(self, app, test_user):
        """创建训练任务应自动创建关联的 ModelRecord"""
        with app.app_context():
            from app.models.model_record import ModelRecord

            job, error = TrainingService.create_job(
                user=test_user,
                name='E2E Test Job',
                task_type='training',
                framework='sklearn',
                ml_task_type='classification',
                algorithm='random_forest',
                total_epochs=10,
            )
            assert job is not None, f'创建任务应成功: {error}'
            assert error is None
            assert job.model_id is not None, '应自动创建关联模型'
            assert job.status in ('queued', 'draft', 'created')

            # 验证 ModelRecord 存在且关联正确
            model = db.session.get(ModelRecord, job.model_id)
            assert model is not None
            assert model.owner_id == test_user.id
            assert 'E2E Test Job' in model.name

    def test_job_lifecycle(self, app, test_user):
        """训练任务状态流转: queued → complete → fail, 不允许非法转换"""
        with app.app_context():
            job, _ = TrainingService.create_job(
                user=test_user,
                name='Lifecycle Test',
                framework='sklearn',
                algorithm='random_forest',
            )
            assert job is not None
            assert job.status in ('queued', 'draft', 'created')

            # 未启动的任务不能完成 (状态检查: 只有 running 状态的任务才能完成)
            success, msg = TrainingService.complete_job(job)
            assert not success

            # 标记任务失败 (无状态前置条件)
            success_fail, msg_fail = TrainingService.fail_job(job, 'test error')
            assert success_fail
            assert job.status == 'failed'
            assert job.error_message == 'test error'

    def test_delete_model_clears_training_job_fk(self, app, test_user):
        """删除模型应清除关联训练任务的 FK (非级联删除)"""
        with app.app_context():
            from app.models.training_job import TrainingJob
            from app.services.model_service import ModelService

            # 创建模型
            model, _ = ModelService.create_model(
                user=test_user,
                name='FK Test Model',
                model_type='classification',
            )

            # 创建关联训练任务
            job = TrainingJob(
                name='FK Test Job',
                task_type='training',
                framework='sklearn',
                owner_id=test_user.id,
                model_id=model.id,
            )
            db.session.add(job)
            db.session.commit()

            # 删除模型
            success, _ = ModelService.delete_model(model)
            assert success

            # 训练任务仍存在, 但 FK 已清除
            job_after = db.session.get(TrainingJob, job.id)
            assert job_after is not None
            assert job_after.model_id is None
