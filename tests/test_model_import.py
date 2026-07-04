"""
============================================
模型导入功能测试
测试 AI 推荐、metadata.json 双向导入、API 端点
============================================
"""
import os
import io
import json
import pickle
import tempfile
import shutil
import zipfile
import pytest
from app import db
from app.services.model_service import ModelService
from app.services.model_recommender import ModelRecommender
from app.services.model_export_service import ModelExportService
from app.models.model_record import ModelRecord


# 模块级 FakeModel (可被 pickle)
class _FakeModel:
    def predict(self, X):
        return [0]


class TestModelRecommender:
    """AI 智能推荐测试"""

    def test_recommend_classification(self):
        """分类模型推荐"""
        info = {
            'algorithm': 'RandomForestClassifier',
            'task_type': 'classification',
            'feature_names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            'class_labels': ['setosa', 'versicolor', 'virginica'],
            'filename': 'iris_model.pkl',
        }
        result = ModelRecommender.recommend(info)
        assert result['name'] is not None
        assert len(result['name']) >= 4
        assert '分类' in result['description'] or '分类' in result['name']
        assert result['version'] == '1.0.0'

    def test_recommend_nlp(self):
        """NLP 模型推荐"""
        info = {
            'algorithm': 'LogisticRegression',
            'task_type': 'classification',
            'feature_names': ['tfidf_good', 'tfidf_bad', 'tfidf_amazing'],
            'class_labels': ['positive', 'negative'],
            'filename': 'sentiment_model.pkl',
        }
        result = ModelRecommender.recommend(info)
        assert result['name'] is not None
        assert result['description'] is not None

    def test_recommend_regression(self):
        """回归模型推荐"""
        info = {
            'algorithm': 'RandomForestRegressor',
            'task_type': 'regression',
            'feature_names': ['age', 'income', 'spending'],
            'filename': 'house_price.pkl',
        }
        result = ModelRecommender.recommend(info)
        assert result['name'] is not None
        # 应该检测到是个回归模型
        assert '回归' in result['name'] or '回归' in result['description'] or 'RF' in result['name']

    def test_recommend_with_existing_metadata(self):
        """有 metadata.json 时使用原有名称"""
        info = {
            'algorithm': 'LogisticRegression',
            'task_type': 'classification',
            'feature_names': ['x1', 'x2'],
            'class_labels': ['A', 'B'],
            'existing_metadata': {
                'name': 'My Production Model',
                'description': 'Original description',
                'version': '2.1.0',
            },
        }
        result = ModelRecommender.recommend(info)
        assert result['name'] == 'My Production Model'
        assert result['description'] == 'Original description'
        assert result['version'] == '2.1.0'

    def test_domain_detection_tfidf(self):
        """TF-IDF 特征 → 情感领域检测"""
        info = {
            'feature_names': ['tfidf_good', 'tfidf_bad', 'tfidf_terrible'],
            'task_type': 'classification',
        }
        # 情感关键词匹配
        info2 = dict(info)
        info2['class_labels'] = ['positive', 'negative']
        result = ModelRecommender.recommend(info2)
        assert '情感' in result['name'] or 'text' in result['name'].lower()

    def test_description_contains_usage(self):
        """描述包含应用场景+使用方式+算法原理三段"""
        info = {
            'dataset_name': '垃圾邮件识别',
            'algorithm': 'RandomForestClassifier',
            'task_type': 'classification',
            'feature_names': ['word_freq_make', 'word_freq_address', 'char_freq_$'],
            'class_labels': ['spam', 'ham'],
            'target_column': 'class',
            'filename': 'spam_model.pkl',
        }
        result = ModelRecommender.recommend(info)
        desc = result['description']
        # 三段式结构
        assert '应用场景' in desc, f"缺少应用场景: {desc}"
        assert '使用方式' in desc, f"缺少使用方式: {desc}"
        # 应包含算法原理 (新格式保留算法描述)
        assert '随机森林' in desc, f"缺少算法信息: {desc}"
        # 应体现具体用途
        assert '垃圾邮件' in desc, f"缺少数据集语境: {desc}"
        assert 'spam' in desc or 'ham' in desc, f"缺少类别信息: {desc}"

    def test_description_no_domain_uses_features(self):
        """无领域信息时, 描述应包含特征数"""
        info = {
            'algorithm': 'KMeans',
            'task_type': 'clustering',
            'feature_names': ['x1', 'x2', 'x3', 'x4'],
            'filename': 'cluster.pkl',
        }
        result = ModelRecommender.recommend(info)
        assert '聚类' in result['description']
        assert '4个特征' in result['description']

    def test_extract_from_model_record(self):
        """从 ModelRecord 提取 info"""
        class FakeModel:
            name = 'TestModel'
            model_type = 'classification'
            hyperparameters_dict = {'algorithm': 'SVC'}
            model_file_path = None
            training_dataset = None
            description = None
            version = '1.0.0'

        fake = FakeModel()
        info = ModelRecommender.extract_from_model_file(
            fake,
            {'algorithm': 'SVC', 'task_type': 'classification'}
        )
        assert info['algorithm'] == 'SVC'
        assert info['task_type'] == 'classification'


class TestModelImportService:
    """ModelService.import_model 测试"""

    def test_import_model_basic(self, app, test_user):
        """基础导入功能"""
        with app.app_context():
            model, error = ModelService.import_model(
                user=test_user,
                name='Imported Model',
                model_type='classification',
                framework='sklearn',
                description='Imported via test',
                version='1.0.0',
                metrics={'accuracy': 0.95, 'f1_score': 0.93},
            )
            assert model is not None
            assert error is None
            assert model.name == 'Imported Model'
            assert model.status == 'trained'  # 与 create_model 不同
            assert model.accuracy == 0.95
            assert model.f1_score == 0.93

    def test_import_model_with_hyperparams(self, app, test_user):
        """带超参数导入"""
        with app.app_context():
            model, _ = ModelService.import_model(
                user=test_user,
                name='HP Model',
                model_type='classification',
                hyperparameters={'C': 1.0, 'solver': 'lbfgs'},
            )
            hp = model.hyperparameters_dict
            assert hp.get('C') == 1.0
            assert hp.get('solver') == 'lbfgs'


class TestModelExportMeta:
    """metadata.json 导出测试"""

    def test_metadata_in_deploy_package(self, app, test_user, tmp_path):
        """部署包包含 metadata.json"""
        with app.app_context():
            # 创建模型并设置 model_file_path
            model, _ = ModelService.create_model(
                user=test_user, name='ExportMeta Test',
                model_type='classification', framework='sklearn',
            )

            # 给模型一个假的模型路径 (用于触发文件复制逻辑)
            fake_pkl = os.path.join(str(tmp_path), 'model.pkl')
            bundle = {
                'model': _FakeModel(),
                'feature_names': ['x1', 'x2'],
                'class_labels': ['A', 'B'],
                'task_type': 'classification',
            }
            with open(fake_pkl, 'wb') as f:
                pickle.dump(bundle, f)

            model.model_file_path = fake_pkl
            model.accuracy = 0.92
            model.f1_score = 0.90
            model.description = 'Test model for export'
            db.session.commit()

            # 生成部署包
            success, msg, pkg_dir, zip_file = ModelExportService.generate_deployment_package(model)
            assert success, f"部署包生成失败: {msg}"
            assert pkg_dir is not None

            # 验证 metadata.json 存在
            meta_path = os.path.join(pkg_dir, 'metadata.json')
            assert os.path.exists(meta_path), "metadata.json 未生成"

            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            # 验证关键字段
            assert meta['name'] == 'ExportMeta Test'
            assert meta['model_type'] == 'classification'
            assert meta['framework'] == 'sklearn'
            assert meta['accuracy'] == 0.92
            assert meta['f1_score'] == 0.90
            assert meta['version'] == '1.0.0'
            assert 'exported_at' in meta
            assert '_inference_meta' in meta
            assert meta['_inference_meta']['feature_names'] == ['x1', 'x2']
            assert meta['_inference_meta']['class_labels'] == ['A', 'B']
            assert meta['_inference_meta']['has_scaler'] is False

    def test_exported_zip_contains_metadata(self, app, test_user, tmp_path):
        """ZIP 部署包包含 metadata.json"""
        with app.app_context():
            model, _ = ModelService.create_model(
                user=test_user, name='ZipMeta Test', model_type='regression',
            )
            fake_pkl = os.path.join(str(tmp_path), 'model2.pkl')
            bundle = {'model': 'dummy', 'task_type': 'regression'}
            with open(fake_pkl, 'wb') as f:
                pickle.dump(bundle, f)
            model.model_file_path = fake_pkl
            db.session.commit()

            success, msg, pkg_dir, zip_name = ModelExportService.generate_deployment_package(model)
            assert success

            # 验证 ZIP 包含 metadata.json
            export_dir = os.path.dirname(pkg_dir) if pkg_dir else ''
            zip_path = os.path.join(os.path.dirname(pkg_dir or ''), zip_name or '')
            if not zip_path.endswith('.zip'):
                # 看 zip 的文件名
                for f in os.listdir(os.path.dirname(pkg_dir or '')):
                    if f.endswith('.zip'):
                        zip_path = os.path.join(os.path.dirname(pkg_dir or ''), f)
                        break

            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    names = zf.namelist()
                    assert any('metadata.json' in n for n in names), \
                        f"ZIP 缺少 metadata.json, 内容: {names}"
            else:
                # make_archive 生成在 exports/<uuid>/ 目录
                # 搜索 zip
                exports_dir = os.path.join('experiments', 'exports', model.uuid)
                zip_candidates = [f for f in os.listdir(exports_dir) if f.endswith('.zip')]
                if zip_candidates:
                    zip_path = os.path.join(exports_dir, zip_candidates[0])
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        names = zf.namelist()
                        assert any('metadata.json' in n for n in names), \
                            f"ZIP 缺少 metadata.json, 内容: {names}"


class TestEndToEndImport:
    """端到端导入测试 — 从部署包 ZIP 导入为完整模型"""

    def test_import_from_deploy_zip(self, app, test_user, tmp_path):
        """从部署包 ZIP 导入, 验证信息100%保留"""
        with app.app_context():
            # ── 1. 创建原始模型 ──
            orig_model, _ = ModelService.create_model(
                user=test_user, name='E2E Original', model_type='classification',
                framework='sklearn', version='2.0.0',
                description='End-to-end test model',
                hyperparameters={'C': 1.5, 'max_iter': 200},
            )
            # 设置指标
            ModelService.update_metrics(orig_model, {
                'accuracy': 0.97, 'f1_score': 0.96, 'precision': 0.95, 'recall': 0.94,
            })

            # 设置模型文件
            fake_pkl = os.path.join(str(tmp_path), 'e2e_model.pkl')
            bundle = {
                'model': 'dummy_classifier',
                'feature_names': ['f1', 'f2', 'f3'],
                'class_labels': ['cat', 'dog'],
                'task_type': 'classification',
                'algorithm': 'RandomForestClassifier',
                'scaler': None,
                'label_encoders': {},
            }
            with open(fake_pkl, 'wb') as f:
                pickle.dump(bundle, f)
            orig_model.model_file_path = fake_pkl
            db.session.commit()

            # ── 2. 导出部署包 ──
            success, msg, pkg_dir, zip_file = ModelExportService.generate_deployment_package(orig_model)
            assert success, f"导出失败: {msg}"

            # ── 3. 验证 metadata.json 内容 ──
            meta_path = os.path.join(pkg_dir, 'metadata.json')
            with open(meta_path, 'r', encoding='utf-8') as f:
                export_meta = json.load(f)

            assert export_meta['name'] == 'E2E Original'
            assert export_meta['description'] == 'End-to-end test model'
            assert export_meta['version'] == '2.0.0'
            assert export_meta['accuracy'] == 0.97
            assert export_meta['f1_score'] == 0.96
            assert export_meta['model_type'] == 'classification'
            assert '_inference_meta' in export_meta
            assert export_meta['_inference_meta']['feature_names'] == ['f1', 'f2', 'f3']

            # ── 4. 用 metadata.json 模拟导入推荐 ──
            info = {
                'algorithm': 'RandomForestClassifier',
                'task_type': 'classification',
                'feature_names': ['f1', 'f2', 'f3'],
                'class_labels': ['cat', 'dog'],
                'existing_metadata': export_meta,
            }
            rec = ModelRecommender.recommend(info)
            assert rec['name'] == 'E2E Original'  # 从 metadata 恢复
            assert rec['version'] == '2.0.0'

            # ── 5. 用 metadata 指标创建新模型 ──
            imported_metrics = {
                'accuracy': export_meta['accuracy'],
                'f1_score': export_meta['f1_score'],
                'precision': export_meta['precision'],
                'recall': export_meta['recall'],
            }
            imp_model, error = ModelService.import_model(
                user=test_user,
                name=rec['name'],
                model_type=export_meta['model_type'],
                framework=export_meta['framework'],
                description=rec['description'],
                version=rec['version'],
                hyperparameters=export_meta['hyperparameters'],
                metrics=imported_metrics,
            )
            assert imp_model is not None
            assert error is None
            assert imp_model.name == 'E2E Original'
            assert imp_model.accuracy == 0.97
            assert imp_model.f1_score == 0.96
            assert imp_model.status == 'trained'
            assert imp_model.hyperparameters_dict.get('C') == 1.5
            assert imp_model.hyperparameters_dict.get('max_iter') == 200
