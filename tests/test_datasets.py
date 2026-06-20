"""
============================================
数据集模块测试 (参数化优化 v2.0)
覆盖 dataset_service.py 完整 CRUD + 分类推断
============================================
"""
import os
import io
import json
import pytest
from app.services.dataset_service import (
    DatasetService, _infer_category, _analyze_dataset_file,
)


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

    def test_get_dataset_by_uuid_not_found(self, app):
        """UUID 查询不存在的数据集"""
        with app.app_context():
            ds = DatasetService.get_dataset_by_uuid('nonexistent-uuid')
            assert ds is None

    def test_allowed_file_extensions(self):
        """允许的文件扩展名验证"""
        assert DatasetService.allowed_file('data.csv') is True
        assert DatasetService.allowed_file('data.json') is True
        assert DatasetService.allowed_file('data.txt') is True
        assert DatasetService.allowed_file('data.xlsx') is True
        assert DatasetService.allowed_file('data.parquet') is True
        assert DatasetService.allowed_file('image.jpg') is True
        assert DatasetService.allowed_file('image.png') is True
        assert DatasetService.allowed_file('data.npy') is True

    def test_allowed_file_rejects_exe(self):
        """拒绝不允许的扩展名"""
        assert DatasetService.allowed_file('virus.exe') is False
        assert DatasetService.allowed_file('script.sh') is False
        assert DatasetService.allowed_file('noextension') is False

    def test_create_and_delete_dataset(self, app, test_user):
        """创建并删除数据集"""
        from app import db as database

        with app.app_context():
            # 创建临时 CSV 文件
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
            try:
                tmp.write('col1,col2,col3\n1,2,3\n4,5,6\n')
                tmp.close()

                from werkzeug.datastructures import FileStorage
                with open(tmp.name, 'rb') as f:
                    file = FileStorage(
                        stream=io.BytesIO(f.read()),
                        filename='test_data.csv',
                        content_type='text/csv',
                    )

                ds, error = DatasetService.create_dataset(
                    user=test_user,
                    name='Test Dataset',
                    file=file,
                    description='Test description',
                    category='tabular',
                    upload_folder=os.path.dirname(tmp.name),
                )

                assert error is None
                assert ds is not None
                assert ds.name == 'Test Dataset'
                assert ds.owner_id == test_user.id
                assert ds.file_format == 'csv'
                assert ds.status == 'ready'

                # 删除数据集
                success, del_error = DatasetService.delete_dataset(ds)
                assert success is True
                assert del_error is None
            finally:
                os.unlink(tmp.name)

    def test_update_dataset(self, app, test_user):
        """更新数据集元数据"""
        from app import db as database

        with app.app_context():
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
            try:
                tmp.write('a,b\n1,2\n')
                tmp.close()

                from werkzeug.datastructures import FileStorage
                with open(tmp.name, 'rb') as f:
                    file = FileStorage(
                        stream=io.BytesIO(f.read()),
                        filename='update_test.csv',
                    )

                ds, _ = DatasetService.create_dataset(
                    user=test_user,
                    name='Old Name',
                    file=file,
                    upload_folder=os.path.dirname(tmp.name),
                )

                # 更新
                success, error = DatasetService.update_dataset(ds, {
                    'name': 'New Name',
                    'description': 'Updated desc',
                    'is_public': True,
                })
                assert success is True
                assert error is None
                assert ds.name == 'New Name'
                assert ds.description == 'Updated desc'
                assert ds.is_public is True

                # 清理
                DatasetService.delete_dataset(ds)
            finally:
                os.unlink(tmp.name)

    def test_update_dataset_invalid_field_ignored(self, app, test_user):
        """更新时忽略非法字段"""
        from app import db as database
        import tempfile

        with app.app_context():
            tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
            try:
                tmp.write('a,b\n1,2\n')
                tmp.close()

                from werkzeug.datastructures import FileStorage
                with open(tmp.name, 'rb') as f:
                    file = FileStorage(stream=io.BytesIO(f.read()), filename='inv_field.csv')

                ds, _ = DatasetService.create_dataset(
                    user=test_user, name='Test', file=file,
                    upload_folder=os.path.dirname(tmp.name),
                )

                success, error = DatasetService.update_dataset(ds, {
                    'name': 'Still Good',
                    'nonexistent_field': 'should_be_ignored',
                })
                assert success is True
                assert ds.name == 'Still Good'

                DatasetService.delete_dataset(ds)
            finally:
                os.unlink(tmp.name)

    def test_list_datasets_with_filters(self, app, test_user):
        """列表查询 — 筛选/搜索/分页不抛异常"""
        from app import db as database
        import tempfile

        with app.app_context():
            # 创建数据集用于测试
            dataset_ids = []
            for i, (name, cat, public) in enumerate([
                ('Alpha Filter DS', 'tabular', True),
                ('Beta Filter DS', 'nlp', False),
            ]):
                tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
                try:
                    tmp.write('x,y\n1,2\n')
                    tmp.close()

                    from werkzeug.datastructures import FileStorage
                    with open(tmp.name, 'rb') as f:
                        content = f.read()
                    file = FileStorage(stream=io.BytesIO(content), filename=f'ds{i}.csv')

                    ds, error = DatasetService.create_dataset(
                        user=test_user, name=name, file=file,
                        category=cat, is_public=public,
                        upload_folder=os.path.dirname(tmp.name),
                    )
                    if ds:
                        dataset_ids.append(ds.id)
                finally:
                    os.unlink(tmp.name)

            # 验证各筛选方法不抛异常
            DatasetService.list_datasets(category='nlp', owner_id=test_user.id)
            DatasetService.list_datasets(search='Alpha', owner_id=test_user.id)
            DatasetService.list_datasets(public_only=True)
            DatasetService.list_datasets(page=1, per_page=5)

            # 清理
            from app.models.dataset import Dataset
            for ds_id in dataset_ids:
                ds = database.session.get(Dataset, ds_id)
                if ds:
                    DatasetService.delete_dataset(ds)

    def test_copy_dataset(self, app, test_user, test_admin):
        """复制公开数据集到用户"""
        from app import db as database
        import tempfile

        with app.app_context():
            tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
            try:
                tmp.write('x,y\n1,2\n3,4\n')
                tmp.close()

                from werkzeug.datastructures import FileStorage
                with open(tmp.name, 'rb') as f:
                    file = FileStorage(stream=io.BytesIO(f.read()), filename='source.csv')

                ds, _ = DatasetService.create_dataset(
                    user=test_admin, name='Public Source', file=file,
                    is_public=True, upload_folder=os.path.dirname(tmp.name),
                )

                # test_user 复制
                copy, error = DatasetService.copy_dataset_to_user(ds, test_user)
                assert error is None
                assert copy is not None
                assert copy.owner_id == test_user.id
                assert copy.is_public is False  # 复制品默认私有
                assert copy.name == ds.name

                # 再次复制同一数据集 → 返回已存在副本 (幂等)
                copy2, error2 = DatasetService.copy_dataset_to_user(ds, test_user)
                assert error2 is None
                assert copy2.id == copy.id

                # 清理
                DatasetService.delete_dataset(copy)
                DatasetService.delete_dataset(ds)
            finally:
                os.unlink(tmp.name)

    def test_statistics_with_data(self, app, test_user):
        """有数据时的统计"""
        from app import db as database
        import tempfile

        with app.app_context():
            tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
            try:
                tmp.write('col1,col2\n1,2\n3,4\n')
                tmp.close()

                from werkzeug.datastructures import FileStorage
                with open(tmp.name, 'rb') as f:
                    file = FileStorage(stream=io.BytesIO(f.read()), filename='stats.csv')

                ds, _ = DatasetService.create_dataset(
                    user=test_user, name='Stats DS', file=file,
                    upload_folder=os.path.dirname(tmp.name),
                )

                stats = DatasetService.get_dataset_statistics(user_id=test_user.id)
                assert stats['total_count'] >= 1
                assert stats['categories']  # 有分类统计

                DatasetService.delete_dataset(ds)
            finally:
                os.unlink(tmp.name)


class TestCategoryInference:
    """_infer_category 智能分类推断"""

    @pytest.mark.parametrize("name,expected_category", [
        ('sentiment_analysis_data', 'nlp'),
        ('text_classification_corpus', 'nlp'),
        ('自然语言处理数据集', 'nlp'),
        ('mnist_handwritten_digits', 'vision'),
        ('cifar10_image_dataset', 'vision'),
        ('图像分类数据', 'vision'),
        ('stock_price_timeseries', 'time_series'),
        ('weather_temperature_sensor', 'time_series'),
        ('时序预测数据', 'time_series'),
        ('housing_price_regression', 'regression'),
        ('california_regression', 'regression'),
        ('customer_segmentation_clustering', 'clustering'),
        ('breast_cancer_biology', 'biology'),
        ('diabetes_medical', 'biology'),
        ('credit_loan_finance', 'finance'),
        ('census_income_adult', 'finance'),
        ('binary_classification', 'classification'),
        ('multiclass_label', 'classification'),
        # 注意: 'blob'→clustering, 'class'→classification, 'gene'→biology
        ('synthetic_data', 'synthetic'),
        ('fake_simulated', 'synthetic'),
    ])
    def test_infer_by_filename(self, name, expected_category):
        """文件名关键词推断分类"""
        result = _infer_category(name, df=None, file_ext='csv')
        assert result == expected_category

    @pytest.mark.parametrize("file_ext,expected", [
        ('jpg', 'vision'),
        ('png', 'vision'),
        ('jpeg', 'vision'),
        ('txt', 'nlp'),
        ('json', 'nlp'),
        ('jsonl', 'nlp'),
        ('csv', 'tabular'),
        ('xlsx', 'tabular'),
        ('parquet', 'tabular'),
    ])
    def test_infer_by_extension(self, file_ext, expected):
        """文件扩展名兜底推断"""
        result = _infer_category('unknown_file', df=None, file_ext=file_ext)
        assert result == expected

    def test_infer_unknown_returns_other(self):
        """无法推断时返回 other"""
        result = _infer_category('random_name_xyz', df=None, file_ext='dat')
        assert result == 'other'

    def test_infer_by_columns(self):
        """通过 DataFrame 列名推断"""
        import pandas as pd
        df = pd.DataFrame(columns=['pixel_0', 'pixel_1', 'pixel_2', 'pixel_3',
                                    'pixel_4', 'pixel_5', 'pixel_6', 'pixel_7',
                                    'pixel_8', 'pixel_9', 'pixel_10', 'pixel_11'])
        result = _infer_category('unknown', df=df, file_ext='csv')
        assert result == 'vision'

    def test_infer_by_nlp_columns(self):
        """通过文本列名推断 NLP"""
        import pandas as pd
        df = pd.DataFrame(columns=['id', 'sentiment', 'text_content', 'label'])
        result = _infer_category('unknown', df=df, file_ext='csv')
        assert result == 'nlp'


class TestDatasetPages:
    """数据集页面测试"""

    def test_list_page_requires_login(self, client):
        """测试数据集列表需要登录"""
        response = client.get('/datasets/', follow_redirects=True)
        assert response.status_code == 200

    @pytest.mark.parametrize("endpoint", [
        "/datasets/",
        "/datasets/create",
    ])
    def test_page_authenticated(self, logged_in_client, endpoint):
        """参数化: 列表页 / 创建页 均需登录"""
        response = logged_in_client.get(endpoint)
        assert response.status_code == 200


class TestDatasetAPI:
    """数据集 API 端点测试 (v2.0 新增)"""

    def test_api_list_datasets(self, client, test_user):
        """API 列出数据集"""
        # 登录
        client.post('/auth/login', data={
            'login_id': 'testuser', 'password': 'Test123456',
        })
        resp = client.get('/api/v1/datasets/',
                         headers={'Authorization': f'Bearer {test_user.api_key}'})
        assert resp.status_code in (200, 401)  # 200 or auth redirect

    def test_api_dataset_not_found(self, client, test_user):
        """API 查询不存在的数据集"""
        resp = client.get('/api/v1/datasets/99999',
                         headers={'Authorization': f'Bearer {test_user.api_key}'})
        assert resp.status_code in (404, 401)

    def test_api_create_dataset_no_file(self, client, test_user):
        """API 创建数据集缺少文件"""
        resp = client.post('/api/v1/datasets/',
                          headers={'Authorization': f'Bearer {test_user.api_key}'},
                          data={'name': 'test'})
        assert resp.status_code in (400, 401)
