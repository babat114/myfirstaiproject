"""
============================================
模型模块测试
============================================
"""
from app.services.model_service import ModelService


class TestModelService:
    """模型服务测试"""

    def test_create_model(self, app, test_user):
        """测试创建模型"""
        with app.app_context():
            model, error = ModelService.create_model(
                user=test_user,
                name='Test Model',
                model_type='classification',
                framework='PyTorch',
                description='A test model',
                hyperparameters={'lr': 0.001, 'epochs': 10},
            )
            assert model is not None
            assert error is None
            assert model.name == 'Test Model'
            assert model.model_type == 'classification'
            assert model.status == 'draft'

    def test_list_models(self, app, test_user):
        """测试模型列表"""
        with app.app_context():
            # 先创建一个模型
            ModelService.create_model(
                user=test_user, name='M1', model_type='classification'
            )
            ModelService.create_model(
                user=test_user, name='M2', model_type='regression'
            )

            result = ModelService.list_models(owner_id=test_user.id)
            assert result['total'] == 2

    def test_get_model_not_found(self, app):
        """测试获取不存在的模型"""
        with app.app_context():
            m = ModelService.get_model_by_id(99999)
            assert m is None

    def test_update_metrics(self, app, test_user):
        """测试更新模型指标"""
        with app.app_context():
            model, _ = ModelService.create_model(
                user=test_user, name='Metric Test'
            )
            metrics = {'accuracy': 0.95, 'f1_score': 0.93, 'loss': 0.12}
            success, error = ModelService.update_metrics(model, metrics)
            assert success is True
            assert model.accuracy == 0.95
            assert model.f1_score == 0.93
            assert model.loss == 0.12

    def test_delete_model(self, app, test_user):
        """测试删除模型"""
        with app.app_context():
            model, _ = ModelService.create_model(
                user=test_user, name='To Delete'
            )
            model_id = model.id
            success, _ = ModelService.delete_model(model)
            assert success is True
            assert ModelService.get_model_by_id(model_id) is None

    def test_get_statistics(self, app, test_user):
        """测试模型统计"""
        with app.app_context():
            ModelService.create_model(user=test_user, name='S1')
            stats = ModelService.get_model_statistics(user_id=test_user.id)
            assert stats['total_count'] == 1


class TestModelPages:
    """模型页面测试"""

    def test_list_page_authenticated(self, logged_in_client):
        """测试已登录用户访问模型列表"""
        response = logged_in_client.get('/models/')
        assert response.status_code == 200

    def test_create_page_authenticated(self, logged_in_client):
        """测试已登录用户访问创建页面"""
        response = logged_in_client.get('/models/create')
        assert response.status_code == 200

    def test_leaderboard_page(self, logged_in_client):
        """测试排行榜页面"""
        response = logged_in_client.get('/models/leaderboard')
        assert response.status_code == 200
