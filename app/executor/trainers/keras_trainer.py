"""
============================================
Keras / TensorFlow 训练器 v2
支持 MLP 全连接网络，验证集+早停，智能模型缩放
============================================

v2 改进 (2026-06-05):
  - 3-way 分割: train/val/test
  - EarlyStopping + ReduceLROnPlateau Keras 回调
  - 智能模型规模推断
  - train + val 双指标报告
  - 更安全默认值: lr=1e-4, dropout=0.5, wd=1e-3
"""

import json
import os
import pickle

import numpy as np
import pandas as pd

from app.executor.trainers.base import BaseTrainer

# ============ Keras 延迟导入 ============
_keras_imported = None


def _ensure_tf():
    """延迟导入 TensorFlow/Keras"""
    global _keras_imported
    if _keras_imported is not None:
        return _keras_imported
    try:
        import tensorflow as tf

        tf.get_logger().setLevel('ERROR')
        _keras_imported = tf
        return tf
    except ImportError:
        raise ImportError('TensorFlow 未安装。请运行: pip install tensorflow')


def _auto_hidden_layers_tf(n_samples: int, n_features: int, n_classes: int = 2) -> list:
    """根据数据集规模自动推断隐藏层结构 (同 PyTorch 逻辑)"""
    if n_samples < 3000:
        h1 = min(n_features // 2, 32)
        h2 = min(n_features // 4, 16)
        if h2 < 8:
            return [max(h1, 8)]
        return [h1, h2]
    elif n_samples < 10000:
        half = min(n_features // 2, 64)
        quarter = min(n_features // 4, 32)
        return [half, quarter]
    elif n_samples < 50000:
        return [128, 64]
    elif n_samples < 200000:
        return [256, 128, 64]
    else:
        return [512, 256, 128, 64]


class PersistentEarlyStopping:
    """Keras EarlyStopping 包装器 — patience 在逐 epoch fit() 调用间持久化

    标准 EarlyStopping.on_train_begin() 在每次 fit() 被调用时重置 wait=0,
    导致单 epoch fit 循环中的早停永不触发。此包装器维护持久状态。
    """

    def __init__(self, monitor='val_loss', patience=10, restore_best_weights=True):
        import tensorflow as tf

        self._es = tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=restore_best_weights, verbose=0
        )
        self._initialized = False

    def __getattr__(self, name):
        # 委托所有未知属性访问到底层 EarlyStopping 实例
        return getattr(self._es, name)

    def on_train_begin(self, logs=None):
        if not self._initialized:
            self._es.on_train_begin(logs)
            self._initialized = True
        # 后续 epoch: 不重置 wait/best/best_weights

    def on_epoch_end(self, epoch, logs=None):
        self._es.on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        self._es.on_train_end(logs)


class KerasTrainer(BaseTrainer):
    """TensorFlow/Keras 深度学习训练器 v2 — MLP 全连接神经网络

    防过拟合机制:
        - 智能模型规模: 根据 n_samples 自动缩放 hidden_layers
        - 3-way 分割: train / val / test
        - EarlyStopping: 监控 val_loss, patience=10
        - ReduceLROnPlateau: val_loss 停滞 → 学习率减半
        - Dense + ReLU + BatchNorm + Dropout 堆叠
        - Adam 优化器 + L2 权重正则化
    """

    def __init__(self, job, dataset, hyperparams: dict = None):
        super().__init__(job, dataset, hyperparams)

        self.task_type = self.hyperparams.get('task_type', 'classification')
        self.hidden_layers = self.hyperparams.get('hidden_layers', None)  # None = 自动
        self.learning_rate = float(self.hyperparams.get('learning_rate', 1e-4))
        self.batch_size = int(self.hyperparams.get('batch_size', 64))
        self.test_size = float(self.hyperparams.get('test_size', 0.2))
        self.val_size = float(self.hyperparams.get('val_size', 0.15))
        self.dropout = float(self.hyperparams.get('dropout', 0.5))
        self.weight_decay = float(self.hyperparams.get('weight_decay', 1e-3))
        self.early_stopping_patience = int(self.hyperparams.get('early_stopping_patience', 10))
        self.lr_patience = int(self.hyperparams.get('lr_patience', 5))
        self.lr_factor = float(self.hyperparams.get('lr_factor', 0.5))

        self._model = None
        self._X_train = self._X_val = self._X_test = None
        self._y_train = self._y_val = self._y_test = None
        self._input_dim = self._output_dim = None
        self._scaler = None
        self._y_scaler = None
        self._label_encoders = {}
        self._feature_names = []
        self._callbacks = []

    def load_data(self):
        tf = _ensure_tf()
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        file_path = self.dataset.file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'数据集文件不存在: {file_path}')

        fmt = self.dataset.file_format.lower()
        if fmt == 'csv':
            df = pd.read_csv(file_path)
        elif fmt in ('xlsx', 'xls'):
            df = pd.read_excel(file_path)
        elif fmt == 'json':
            df = pd.read_json(file_path)
        else:
            df = pd.read_csv(file_path)

        target_col = self.hyperparams.get('target_column') or df.columns[-1]
        if target_col not in df.columns:
            raise ValueError(f'目标列 "{target_col}" 不存在')

        X = df.drop(columns=[target_col])
        y_raw = df[target_col]
        self._feature_names = list(X.columns)

        # 处理缺失值
        num_cols = X.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            X[num_cols] = SimpleImputer(strategy='mean').fit_transform(X[num_cols])
        cat_cols = X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown')

        # 编码分类特征
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self._label_encoders[col] = le

        # 编码目标变量
        if self.task_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
            self._label_encoders['__target__'] = le
            self._output_dim = len(le.classes_)
        else:
            y_raw_vals = y_raw.values.astype('float32').reshape(-1, 1)
            self._y_scaler = StandardScaler()
            y = self._y_scaler.fit_transform(y_raw_vals).ravel().astype('float32')
            self._output_dim = 1

        # 标准化特征
        X_num = X.values.astype('float32')
        self._scaler = StandardScaler()
        X_num = self._scaler.fit_transform(X_num).astype('float32')
        self._input_dim = X_num.shape[1]
        n_samples = X_num.shape[0]

        # —— 智能模型规模 ——
        if self.hidden_layers is None:
            n_classes = self._output_dim if self.task_type == 'classification' else 2
            self.hidden_layers = _auto_hidden_layers_tf(n_samples, self._input_dim, n_classes)
            self.callback.on_log(f'智能模型规模: {self.hidden_layers} (样本={n_samples}, 特征={self._input_dim})')
        else:
            self.callback.on_log(f'用户指定网络结构: {self.hidden_layers}')

        # —— 3-way 分割 ——
        stratify_y = y if self.task_type == 'classification' and len(set(y)) > 1 else None
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_num, y, test_size=self.test_size, random_state=42, stratify=stratify_y
        )
        val_ratio = self.val_size / (1.0 - self.test_size)
        stratify_tv = y_train_val if self.task_type == 'classification' and len(set(y_train_val)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=stratify_tv
        )

        self._X_train, self._X_val, self._X_test = X_train, X_val, X_test
        self._y_train, self._y_val, self._y_test = y_train, y_val, y_test

        self.callback.on_log(f'设备: CPU (TensorFlow {tf.__version__})')
        self.callback.on_log(f'输入维度: {self._input_dim}, 输出维度: {self._output_dim}')
        self.callback.on_log(f'训练: {len(X_train)} | 验证: {len(X_val)} | 测试: {len(X_test)}')
        self.callback.on_log(f'网络结构: {self.hidden_layers}')

    def build_model(self):
        tf = _ensure_tf()

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self._input_dim,)))

        for _i, units in enumerate(self.hidden_layers):
            model.add(
                tf.keras.layers.Dense(
                    units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay)
                )
            )
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(self.dropout))

        if self.task_type == 'classification':
            model.add(tf.keras.layers.Dense(self._output_dim, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(tf.keras.layers.Dense(1))
            loss = 'mse'
            metrics = ['mae']

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss=loss, metrics=metrics)

        self._model = model

        # —— Keras 回调: 持久化早停 + LR 调度 ——
        self._callbacks = [
            PersistentEarlyStopping(
                monitor='val_loss', patience=self.early_stopping_patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=self.lr_factor, patience=self.lr_patience, min_lr=1e-6, verbose=0
            ),
        ]

        total_params = model.count_params()
        self.callback.on_log(
            f'参数: 总={total_params:,} lr={self.learning_rate} dropout={self.dropout} wd={self.weight_decay}'
        )

    def train_epoch(self, epoch: int) -> dict:
        """训练一个 epoch，返回 train + val 指标"""
        if self.task_type == 'classification':
            y_train = self._y_train.astype('int64')
            y_val = self._y_val.astype('int64')
        else:
            y_train = self._y_train.astype('float32')
            y_val = self._y_val.astype('float32')

        # 整个训练过程用 model.fit + callbacks
        # 注意: Keras 的 fit 是 "训练到完成" 而非 "训练1个epoch"
        # 这里用一次 fit(epochs=1) 模拟单 epoch, callbacks 在 epoch 结束时触发
        history = self._model.fit(
            self._X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0,
            validation_data=(self._X_val, y_val),
            callbacks=self._callbacks,
        )

        result = {
            'train_loss': round(float(history.history['loss'][0]), 4),
            'val_loss': round(float(history.history['val_loss'][0]), 4),
        }
        if self.task_type == 'classification' and 'accuracy' in history.history:
            result['train_accuracy'] = round(float(history.history['accuracy'][0]), 4)
            result['val_accuracy'] = round(float(history.history['val_accuracy'][0]), 4)

        # 检测早停 (通过 lr 变化判断)
        current_lr = float(self._model.optimizer.learning_rate.numpy())
        result['lr'] = current_lr

        return result

    def evaluate(self) -> dict:
        """最终评估: 在测试集上计算指标"""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
        )

        y_test = self._y_test.astype('int64') if self.task_type == 'classification' else self._y_test.astype('float32')

        y_pred_raw = self._model.predict(self._X_test, verbose=0)

        result = {}
        if self.task_type == 'classification':
            y_pred = np.argmax(y_pred_raw, axis=1)
            result['test_accuracy'] = round(float(accuracy_score(y_test, y_pred)), 4)
            try:
                result['test_precision_weighted'] = round(
                    float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 4
                )
                result['test_recall_weighted'] = round(
                    float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 4
                )
                result['test_f1_weighted'] = round(
                    float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4
                )
                result['test_precision_macro'] = round(
                    float(precision_score(y_test, y_pred, average='macro', zero_division=0)), 4
                )
                result['test_recall_macro'] = round(
                    float(recall_score(y_test, y_pred, average='macro', zero_division=0)), 4
                )
                result['test_f1_macro'] = round(float(f1_score(y_test, y_pred, average='macro', zero_division=0)), 4)
            except Exception:
                pass
        else:
            y_pred_flat = y_pred_raw.flatten().astype('float32')
            y_test_flat = y_test.astype('float32')
            if self._y_scaler is not None:
                y_pred_flat = self._y_scaler.inverse_transform(y_pred_flat.reshape(-1, 1)).ravel()
                y_test_flat = self._y_scaler.inverse_transform(y_test_flat.reshape(-1, 1)).ravel()
            result['test_mse'] = round(float(mean_squared_error(y_test_flat, y_pred_flat)), 4)
            result['test_r2'] = round(float(r2_score(y_test_flat, y_pred_flat)), 4)
        return result

    def save_model(self, path: str):
        _ensure_tf()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self._model.save(path + '.keras')

        config = {
            'model_class': 'MLPClassifier' if self.task_type == 'classification' else 'MLPRegressor',
            'input_dim': self._input_dim,
            'output_dim': self._output_dim,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
            'task_type': self.task_type,
            'feature_names': self._feature_names,
            'scaler': self._scaler,
            'y_scaler': self._y_scaler,
            'label_encoders': self._label_encoders,
            'framework': 'TensorFlow',
            'val_size': self.val_size,
            'early_stopping_patience': self.early_stopping_patience,
        }
        with open(path + '_config.pkl', 'wb') as f:
            pickle.dump(config, f)

        self.callback.on_log(f'模型已保存到: {path}.keras + _config.pkl')

    # ============ 检查点 ============

    def save_checkpoint(self):
        """保存 Keras 训练快照: 模型权重 + 优化器权重 + epoch"""
        if self._model is None:
            return
        _ensure_tf()
        os.makedirs(self.output_dir, exist_ok=True)

        ckpt_dir = os.path.join(self.output_dir, 'checkpoint.keras')
        self._model.save(ckpt_dir)

        # 保存元数据
        meta = {
            'epoch': self._current_epoch,
            'input_dim': self._input_dim,
            'output_dim': self._output_dim,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
            'task_type': self.task_type,
            'val_size': self.val_size,
            'early_stopping_patience': self.early_stopping_patience,
        }
        with open(os.path.join(self.output_dir, 'checkpoint_meta.json'), 'w') as f:
            json.dump(meta, f)

    @staticmethod
    def load_checkpoint(output_dir: str) -> dict:
        meta_path = os.path.join(output_dir, 'checkpoint_meta.json')
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path) as f:
            meta = json.load(f)
        return {
            'epoch': meta.get('epoch', 0),
            '_restore': {
                'ckpt_dir': os.path.join(output_dir, 'checkpoint.keras'),
                'meta': meta,
            },
        }

    def restore_checkpoint(self, ckpt: dict):
        """恢复 Keras 模型权重 + 优化器状态"""
        restore_data = ckpt.get('_restore')
        if not restore_data:
            return
        tf = _ensure_tf()
        ckpt_dir = restore_data.get('ckpt_dir')
        if ckpt_dir and os.path.exists(ckpt_dir):
            self._model = tf.keras.models.load_model(ckpt_dir)
            self.callback.on_log('[检查点] Keras 模型权重+优化器已恢复')

    @staticmethod
    def has_checkpoint(output_dir: str) -> bool:
        return os.path.exists(os.path.join(output_dir, 'checkpoint.keras'))
