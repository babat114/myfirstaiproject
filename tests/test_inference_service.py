"""
============================================
模型推理服务单元测试 v1.0
覆盖 app/services/inference_service.py 全部公开方法
============================================
"""
import os
import json
import pickle
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, PropertyMock
from app.services.inference_service import ModelInferenceService


class TestLoadModel:
    """load_model 测试"""

    def test_model_path_not_exists(self, app):
        """模型文件路径不存在"""
        with app.app_context():
            from app.models.model_record import ModelRecord
            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = '/nonexistent/path/model.pkl'
            mock_model.training_job = None

            model_obj, metadata, tokenizer, error = ModelInferenceService.load_model(mock_model)
            assert model_obj is None
            assert '不存在' in error or '模型文件不' in error

    def test_unsupported_extension(self, app, tmp_path):
        """不支持的模型格式"""
        with app.app_context():
            from app.models.model_record import ModelRecord

            # 创建临时文件
            model_file = tmp_path / 'model.xyz'
            model_file.write_text('dummy')

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None

            model_obj, metadata, tokenizer, error = ModelInferenceService.load_model(mock_model)
            assert model_obj is None
            assert '不支持' in error or '格式' in error

    def test_load_sklearn_pickle(self, app, tmp_path):
        """加载 sklearn pickle 模型"""
        with app.app_context():
            from sklearn.linear_model import LogisticRegression
            from app.models.model_record import ModelRecord

            # 训练一个简易 sklearn 模型
            X = np.array([[1, 2], [3, 4], [5, 6]])
            y = np.array([0, 1, 0])
            clf = LogisticRegression()
            clf.fit(X, y)

            bundle = {
                'model': clf,
                'scaler': None,
                'label_encoders': {},
                'feature_names': ['f1', 'f2'],
                'task_type': 'classification',
                'algorithm': 'logistic_regression',
            }
            model_file = tmp_path / 'model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(bundle, f)

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None
            mock_model.name = 'test_model'

            model_obj, metadata, tokenizer, error = ModelInferenceService.load_model(mock_model)
            assert error is None
            assert model_obj is not None
            assert metadata is not None
            assert metadata['task_type'] == 'classification'
            assert metadata['feature_names'] == ['f1', 'f2']
            assert tokenizer is None  # sklearn 无 tokenizer

    def test_load_sklearn_pickle_bare_model(self, app, tmp_path):
        """加载未包装的裸 sklearn 模型"""
        with app.app_context():
            from sklearn.linear_model import LinearRegression
            from app.models.model_record import ModelRecord

            reg = LinearRegression()
            reg.fit(np.array([[1], [2], [3]]), np.array([2, 4, 6]))

            model_file = tmp_path / 'bare_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(reg, f)

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None
            mock_model.model_type = 'regression'

            model_obj, metadata, tokenizer, error = ModelInferenceService.load_model(mock_model)
            assert error is None
            assert model_obj is not None
            assert metadata['task_type'] == 'regression'

    def test_load_pytorch_model_missing_torch(self, app, tmp_path):
        """PyTorch 模型在无 PyTorch 环境加载 (预期报错)"""
        with app.app_context():
            from app.models.model_record import ModelRecord

            model_file = tmp_path / 'model.pt'
            model_file.write_text('dummy')

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None

            # load_mlp_model 在 inference_service.py 函数内 import, patch 对应位置
            with patch('app.executor.trainers.pytorch_trainer.load_mlp_model',
                      side_effect=ImportError('No PyTorch')):
                model_obj, metadata, tokenizer, error = ModelInferenceService.load_model(mock_model)
                # PyTorch 加载失败应返回错误
                assert error is not None or model_obj is None

    def test_load_model_fallback_to_experiment_dir(self, app, tmp_path):
        """回退到 experiment 目录查找模型 — 验证不崩溃"""
        with app.app_context():
            from app.models.model_record import ModelRecord

            exp_dir = tmp_path / 'experiments' / 'test-uuid-fallback'
            exp_dir.mkdir(parents=True)
            model_file = exp_dir / 'model.pkl'

            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
            clf.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
            with open(model_file, 'wb') as f:
                pickle.dump({'model': clf, 'task_type': 'classification'}, f)

            mock_job = MagicMock()
            mock_job.uuid = 'test-uuid-fallback'

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = mock_job
            mock_model.name = 'fallback_model'

            # 直接使用已存在的文件路径加载，验证不崩溃
            model_obj, metadata, tokenizer, error = ModelInferenceService.load_model(mock_model)
            assert error is None
            assert model_obj is not None


class TestPredict:
    """predict 测试"""

    def test_predict_load_fails(self, app):
        """模型加载失败时 predict 返回错误"""
        with app.app_context():
            from app.models.model_record import ModelRecord

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = '/nonexistent/model.pkl'
            mock_model.training_job = None

            df = pd.DataFrame({'f1': [1, 2], 'f2': [3, 4]})
            result = ModelInferenceService.predict(mock_model, df)
            assert result['success'] is False
            assert 'error' in result
            assert result['predictions'] == []

    def test_predict_sklearn_classification(self, app, tmp_path):
        """sklearn 分类预测"""
        with app.app_context():
            from sklearn.linear_model import LogisticRegression
            from app.models.model_record import ModelRecord

            X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            y_train = np.array([0, 1, 0, 1])
            clf = LogisticRegression()
            clf.fit(X_train, y_train)

            bundle = {
                'model': clf,
                'scaler': None,
                'label_encoders': {},
                'feature_names': ['f1', 'f2'],
                'task_type': 'classification',
            }
            model_file = tmp_path / 'clf_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(bundle, f)

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None
            mock_model.name = 'clf_test'

            df = pd.DataFrame({'f1': [2, 6], 'f2': [3, 7]})
            result = ModelInferenceService.predict(mock_model, df)
            assert result['success'] is True
            assert len(result['predictions']) == 2
            assert result['task_type'] == 'classification'
            assert result['num_samples'] == 2

    def test_predict_missing_feature_columns(self, app, tmp_path):
        """缺少特征列时返回错误"""
        with app.app_context():
            from sklearn.linear_model import LogisticRegression
            from app.models.model_record import ModelRecord

            clf = LogisticRegression()
            clf.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))

            bundle = {
                'model': clf,
                'feature_names': ['col_a', 'col_b', 'col_c'],
                'task_type': 'classification',
            }
            model_file = tmp_path / 'feat_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(bundle, f)

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None

            # 缺少 col_c
            df = pd.DataFrame({'col_a': [1], 'col_b': [2]})
            result = ModelInferenceService.predict(mock_model, df)
            assert result['success'] is False
            assert '缺少特征列' in result.get('error', '')

    def test_predict_regression(self, app, tmp_path):
        """回归预测"""
        with app.app_context():
            from sklearn.linear_model import LinearRegression
            from app.models.model_record import ModelRecord

            reg = LinearRegression()
            reg.fit(np.array([[1], [2], [3]]), np.array([2, 4, 6]))

            bundle = {
                'model': reg,
                'feature_names': ['x'],
                'task_type': 'regression',
            }
            model_file = tmp_path / 'reg_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(bundle, f)

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None
            mock_model.model_type = 'regression'

            df = pd.DataFrame({'x': [4, 5]})
            result = ModelInferenceService.predict(mock_model, df)
            assert result['success'] is True
            assert len(result['predictions']) == 2
            assert result['task_type'] == 'regression'


class TestPredictSingle:
    """predict_single 测试 (NLP Transformer 快速预测)"""

    def test_predict_single_empty_text(self):
        """空文本返回 None"""
        result = ModelInferenceService.predict_single(
            model_obj=None, tokenizer=None, metadata={}, text=''
        )
        assert result is None

    def test_predict_single_whitespace(self):
        """纯空白文本返回 None"""
        result = ModelInferenceService.predict_single(
            model_obj=None, tokenizer=None, metadata={}, text='   '
        )
        assert result is None


class TestTestModelWithSplit:
    """test_model_with_split 测试"""

    def test_model_file_missing(self, app):
        """模型文件不存在"""
        with app.app_context():
            from app.models.model_record import ModelRecord

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = '/nonexistent/model.pkl'
            mock_model.training_job = None
            mock_model.training_dataset = None

            result = ModelInferenceService.test_model_with_split(mock_model)
            assert result['success'] is False
            assert 'error' in result

    def test_dataset_missing(self, app, tmp_path):
        """原始数据集不存在"""
        with app.app_context():
            from sklearn.linear_model import LogisticRegression
            from app.models.model_record import ModelRecord

            clf = LogisticRegression()
            clf.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
            model_file = tmp_path / 'eval_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump({'model': clf, 'task_type': 'classification'}, f)

            mock_dataset = MagicMock()
            mock_dataset.file_path = '/nonexistent/dataset.csv'
            mock_dataset.file_format = 'csv'

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None
            mock_model.training_dataset = mock_dataset
            mock_model.hyperparameters_dict = {}

            result = ModelInferenceService.test_model_with_split(mock_model)
            assert result['success'] is False


class TestGetFeatureImportance:
    """get_feature_importance 测试"""

    def test_model_not_found(self, app):
        """模型不存在"""
        with app.app_context():
            from app.models.model_record import ModelRecord

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = '/nonexistent/model.pkl'
            mock_model.training_job = None

            result = ModelInferenceService.get_feature_importance(mock_model)
            assert result['success'] is False
            assert 'error' in result

    def test_tree_model_importance(self, app, tmp_path):
        """树模型特征重要性"""
        with app.app_context():
            from sklearn.ensemble import RandomForestClassifier
            from app.models.model_record import ModelRecord

            rf = RandomForestClassifier(n_estimators=5, random_state=42)
            rf.fit(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([0, 1, 0]))

            bundle = {
                'model': rf,
                'feature_names': ['feat_a', 'feat_b', 'feat_c'],
            }
            model_file = tmp_path / 'rf_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(bundle, f)

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None

            result = ModelInferenceService.get_feature_importance(mock_model)
            assert result['success'] is True
            assert len(result['features']) == 3
            assert len(result['importances']) == 3

    def test_linear_model_coefficients(self, app, tmp_path):
        """线性模型系数"""
        with app.app_context():
            from sklearn.linear_model import LogisticRegression
            from app.models.model_record import ModelRecord

            lr = LogisticRegression()
            lr.fit(np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 0]))

            bundle = {
                'model': lr,
                'feature_names': ['f1', 'f2'],
            }
            model_file = tmp_path / 'lr_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(bundle, f)

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None

            result = ModelInferenceService.get_feature_importance(mock_model)
            assert result['success'] is True
            assert len(result['features']) == 2
            assert len(result['importances']) == 2


class TestInferenceEdgeCases:
    """推理边界情况"""

    def test_missing_values_imputed(self, app, tmp_path):
        """缺失值自动填充"""
        with app.app_context():
            from sklearn.linear_model import LogisticRegression
            from app.models.model_record import ModelRecord

            clf = LogisticRegression()
            clf.fit(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), np.array([0, 1, 0]))

            bundle = {
                'model': clf,
                'feature_names': ['f1', 'f2'],
                'task_type': 'classification',
            }
            model_file = tmp_path / 'nan_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(bundle, f)

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None
            mock_model.model_type = 'classification'

            # 包含 NaN 的数据
            df = pd.DataFrame({'f1': [1.0, np.nan], 'f2': [np.nan, 4.0]})
            result = ModelInferenceService.predict(mock_model, df)
            assert result['success'] is True
            assert len(result['predictions']) == 2

    def test_missing_features_no_feature_names(self, app, tmp_path):
        """无 feature_names 时不检查列匹配"""
        with app.app_context():
            from sklearn.linear_model import LogisticRegression
            from app.models.model_record import ModelRecord

            clf = LogisticRegression()
            clf.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1]))

            bundle = {
                'model': clf,
                'feature_names': [],
                'task_type': 'classification',
            }
            model_file = tmp_path / 'no_names.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(bundle, f)

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None

            df = pd.DataFrame({'any_col1': [1, 3], 'any_col2': [2, 4]})
            result = ModelInferenceService.predict(mock_model, df)
            assert result['success'] is True


class TestTensorFlowLoader:
    """TensorFlow .keras/.h5 模型加载测试"""

    def test_load_keras_model(self, app, tmp_path):
        """加载 .keras 格式 TensorFlow 模型"""
        with app.app_context():
            try:
                import tensorflow as tf
            except ImportError:
                pytest.skip('TensorFlow 未安装')

            from app.models.model_record import ModelRecord

            # 创建简易 TF 模型
            model_tf = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(4,)),
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax'),
            ])
            model_tf.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

            model_file = tmp_path / 'model.keras'
            model_tf.save(str(model_file))

            # 保存 config
            config = {
                'model_class': 'MLPClassifier',
                'input_dim': 4,
                'output_dim': 3,
                'hidden_layers': [8],
                'dropout': 0.3,
                'task_type': 'classification',
                'feature_names': ['f1', 'f2', 'f3', 'f4'],
                'scaler': None,
                'label_encoders': {},
                'framework': 'TensorFlow',
            }
            config_path = tmp_path / 'model_config.pkl'
            with open(config_path, 'wb') as f:
                pickle.dump(config, f)

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None
            mock_model.name = 'test_tf'

            model_obj, metadata, tokenizer, error = ModelInferenceService.load_model(mock_model)
            assert error is None, f'Load failed: {error}'
            assert model_obj is not None
            assert metadata is not None
            assert metadata['framework'] == 'tensorflow'
            assert metadata['task_type'] == 'classification'
            assert metadata['input_dim'] == 4
            assert metadata['feature_names'] == ['f1', 'f2', 'f3', 'f4']
            assert tokenizer is None  # TF 表格模型无 tokenizer

    def test_load_h5_model(self, app, tmp_path):
        """加载 .h5 (HDF5) 格式 — 向后兼容"""
        with app.app_context():
            try:
                import tensorflow as tf
            except ImportError:
                pytest.skip('TensorFlow 未安装')

            from app.models.model_record import ModelRecord

            model_tf = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(3,)),
                tf.keras.layers.Dense(6, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax'),
            ])
            model_tf.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

            model_file = tmp_path / 'model.h5'
            model_tf.save(str(model_file))

            config_path = tmp_path / 'model_config.pkl'
            with open(config_path, 'wb') as f:
                pickle.dump({
                    'task_type': 'classification',
                    'input_dim': 3,
                    'output_dim': 2,
                    'feature_names': ['a', 'b', 'c'],
                    'framework': 'TensorFlow',
                }, f)

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None
            mock_model.name = 'test_h5'

            model_obj, metadata, tokenizer, error = ModelInferenceService.load_model(mock_model)
            assert error is None, f'.h5 load failed: {error}'
            assert model_obj is not None
            assert metadata['framework'] == 'tensorflow'

    def test_load_keras_no_config(self, app, tmp_path):
        """TF 模型无 model_config.pkl — 应优雅降级"""
        with app.app_context():
            try:
                import tensorflow as tf
            except ImportError:
                pytest.skip('TensorFlow 未安装')

            from app.models.model_record import ModelRecord

            model_tf = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(2,)),
                tf.keras.layers.Dense(1),
            ])
            model_tf.compile(optimizer='adam', loss='mse')

            model_file = tmp_path / 'model.keras'
            model_tf.save(str(model_file))
            # 不创建 config

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None
            mock_model.model_type = 'regression'
            mock_model.name = 'no_config_tf'

            model_obj, metadata, tokenizer, error = ModelInferenceService.load_model(mock_model)
            assert error is None
            assert model_obj is not None
            assert metadata['framework'] == 'tensorflow'
            assert metadata['task_type'] == 'regression'  # fallback to model.model_type


class TestTensorFlowPredict:
    """TensorFlow 模型预测测试"""

    def test_predict_tf_classification(self, app, tmp_path):
        """TF 分类模型预测"""
        with app.app_context():
            try:
                import tensorflow as tf
            except ImportError:
                pytest.skip('TensorFlow 未安装')

            from app.models.model_record import ModelRecord

            # 训练 TF 分类模型
            X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype='float32')
            y = np.array([0, 1, 0, 1], dtype='int32')

            model_tf = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(2,)),
                tf.keras.layers.Dense(4, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax'),
            ])
            model_tf.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            model_tf.fit(X, y, epochs=5, verbose=0)

            model_file = tmp_path / 'model.keras'
            model_tf.save(str(model_file))

            config_path = tmp_path / 'model_config.pkl'
            with open(config_path, 'wb') as f:
                pickle.dump({
                    'task_type': 'classification',
                    'input_dim': 2,
                    'output_dim': 2,
                    'feature_names': ['f1', 'f2'],
                    'scaler': None,
                    'label_encoders': {},
                    'framework': 'TensorFlow',
                }, f)

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None
            mock_model.model_type = 'classification'
            mock_model.name = 'tf_pred_test'

            df = pd.DataFrame({'f1': [2.0, 6.0], 'f2': [3.0, 7.0]})
            result = ModelInferenceService.predict(mock_model, df)
            assert result['success'] is True
            assert len(result['predictions']) == 2
            assert result['task_type'] == 'classification'
            assert result['num_samples'] == 2
            assert result['probabilities'] is not None
            assert len(result['probabilities']) == 2

    def test_predict_tf_regression(self, app, tmp_path):
        """TF 回归模型预测"""
        with app.app_context():
            try:
                import tensorflow as tf
            except ImportError:
                pytest.skip('TensorFlow 未安装')

            from app.models.model_record import ModelRecord

            X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype='float32')
            y = np.array([2.0, 4.0, 6.0, 8.0], dtype='float32')

            model_tf = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(1,)),
                tf.keras.layers.Dense(4, activation='relu'),
                tf.keras.layers.Dense(1),
            ])
            model_tf.compile(optimizer='adam', loss='mse')
            model_tf.fit(X, y, epochs=10, verbose=0)

            model_file = tmp_path / 'model.keras'
            model_tf.save(str(model_file))

            config_path = tmp_path / 'model_config.pkl'
            with open(config_path, 'wb') as f:
                pickle.dump({
                    'task_type': 'regression',
                    'input_dim': 1,
                    'output_dim': 1,
                    'feature_names': ['x'],
                    'scaler': None,
                    'label_encoders': {},
                    'framework': 'TensorFlow',
                }, f)

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None
            mock_model.model_type = 'regression'
            mock_model.name = 'tf_reg_test'

            df = pd.DataFrame({'x': [2.5, 3.5]})
            result = ModelInferenceService.predict(mock_model, df)
            assert result['success'] is True
            assert len(result['predictions']) == 2
            assert result['task_type'] == 'regression'

    def test_predict_tf_with_scaler(self, app, tmp_path):
        """TF 模型 + StandardScaler 预处理"""
        with app.app_context():
            try:
                import tensorflow as tf
            except ImportError:
                pytest.skip('TensorFlow 未安装')

            from sklearn.preprocessing import StandardScaler
            from app.models.model_record import ModelRecord

            X_raw = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype='float32')
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)

            y = np.array([0, 1, 0], dtype='int32')

            model_tf = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(2,)),
                tf.keras.layers.Dense(4, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax'),
            ])
            model_tf.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            model_tf.fit(X_scaled, y, epochs=5, verbose=0)

            model_file = tmp_path / 'model.keras'
            model_tf.save(str(model_file))

            config_path = tmp_path / 'model_config.pkl'
            with open(config_path, 'wb') as f:
                pickle.dump({
                    'task_type': 'classification',
                    'input_dim': 2,
                    'output_dim': 2,
                    'feature_names': ['f1', 'f2'],
                    'scaler': scaler,
                    'label_encoders': {},
                    'framework': 'TensorFlow',
                }, f)

            mock_model = MagicMock(spec=ModelRecord)
            mock_model.model_file_path = str(model_file)
            mock_model.training_job = None
            mock_model.model_type = 'classification'
            mock_model.name = 'tf_scaler_test'

            # 输入 raw scale 数据, scaler 应在 predict 内自动转换
            df = pd.DataFrame({'f1': [20.0], 'f2': [30.0]})
            result = ModelInferenceService.predict(mock_model, df)
            assert result['success'] is True
            assert len(result['predictions']) == 1
            assert result['probabilities'] is not None
