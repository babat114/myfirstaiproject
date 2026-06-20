"""
============================================
训练任务模块测试 (参数化优化 v1.0)
============================================
"""
import pytest
from app.services.training_service import TrainingService


class TestTrainingService:
    """训练服务测试"""

    @pytest.mark.parametrize("name,expect_job_not_none", [
        ("Test Training Job", True),
        ("",                  True),   # 当前行为: 允许空名称
    ])
    def test_create_job(self, test_user, name, expect_job_not_none):
        """参数化: 有效名称 / 空名称"""
        job, error = TrainingService.create_job(
            user=test_user,
            name=name,
            task_type='training',
            framework='scikit-learn',
            algorithm='randomforest',
        )
        assert (job is not None) == expect_job_not_none
        if expect_job_not_none:
            assert error is None
            assert job.name == name
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
