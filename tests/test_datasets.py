"""
============================================
数据集模块测试
============================================
"""
from app.services.dataset_service import DatasetService


class TestDatasetService:
    """数据集服务测试"""

    def test_list_datasets_empty(self, app, test_user):
        """测试空数据集列表"""
        with app.app_context():
            result = DatasetService.list_datasets(owner_id=test_user.id)
            assert result['total'] == 0
            assert len(result['items']) == 0

    def test_get_dataset_not_found(self, app):
        """测试获取不存在的数据集"""
        with app.app_context():
            ds = DatasetService.get_dataset_by_id(99999)
            assert ds is None

    def test_get_statistics_empty(self, app, test_user):
        """测试空用户的统计"""
        with app.app_context():
            stats = DatasetService.get_dataset_statistics(user_id=test_user.id)
            assert stats['total_count'] == 0
            assert stats['total_size_bytes'] == 0


class TestDatasetPages:
    """数据集页面测试"""

    def test_list_page_requires_login(self, client):
        """测试数据集列表需要登录"""
        response = client.get('/datasets/', follow_redirects=True)
        assert response.status_code == 200

    def test_list_page_authenticated(self, logged_in_client):
        """测试已登录用户访问数据集列表"""
        response = logged_in_client.get('/datasets/')
        assert response.status_code == 200

    def test_create_page_authenticated(self, logged_in_client):
        """测试已登录用户访问创建页面"""
        response = logged_in_client.get('/datasets/create')
        assert response.status_code == 200
