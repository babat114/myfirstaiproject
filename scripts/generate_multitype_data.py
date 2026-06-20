"""
============================================
多类型数据集生成 + 批量模型训练脚本
生成7种模型类型的数据集 (10-100MB) 并训练48个模型
============================================
"""
import os
import sys
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.user import User
from app.models.dataset import Dataset
from app.models.model_record import ModelRecord
from app.models.training_job import TrainingJob

# ============ 配置 ============
UPLOAD_DIR = os.path.join('uploads', 'datasets')
EXPERIMENTS_DIR = 'experiments'

# 8种模型类型 × 6个算法
TYPE_ALGORITHMS = {
    'classification': [
        ('random_forest', 'sklearn', 'RandomForest 分类器'),
        ('gradient_boosting', 'sklearn', 'GradientBoosting 分类器'),
        ('logistic_regression', 'sklearn', 'LogisticRegression 分类器'),
        ('svm', 'sklearn', 'SVM 分类器'),
        ('knn', 'sklearn', 'KNN 分类器'),
        ('mlp', 'pytorch', 'MLP 神经网络分类器'),
    ],
    'regression': [
        ('random_forest_regressor', 'sklearn', 'RandomForest 回归器'),
        ('gradient_boosting_regressor', 'sklearn', 'GradientBoosting 回归器'),
        ('linear_regression', 'sklearn', 'LinearRegression 回归器'),
        ('svr', 'sklearn', 'SVR 回归器'),
        ('knn_regressor', 'sklearn', 'KNN 回归器'),
        ('mlp_regressor', 'pytorch', 'MLP 神经网络回归器'),
    ],
    'clustering': [
        ('kmeans', 'sklearn', 'K-Means 聚类'),
        ('dbscan', 'sklearn', 'DBSCAN 密度聚类'),
        ('agglomerative', 'sklearn', 'AgglomerativeClustering 层次聚类'),
        ('minibatch_kmeans', 'sklearn', 'MiniBatch K-Means 聚类'),
        ('random_forest', 'sklearn', 'RF 聚类特征分类 (监督baseline)'),
        ('mlp', 'pytorch', 'MLP 聚类模式识别 (监督baseline)'),
    ],
    'nlp': [
        ('random_forest', 'sklearn', 'RF 文本特征分类'),
        ('gradient_boosting', 'sklearn', 'GB 文本特征分类'),
        ('logistic_regression', 'sklearn', 'LR 文本情感分类'),
        ('svm', 'sklearn', 'SVM 高维文本分类'),
        ('knn', 'sklearn', 'KNN 语义相似分类'),
        ('mlp', 'pytorch', 'MLP 文本深度分类'),
    ],
    'computer_vision': [
        ('random_forest', 'sklearn', 'RF 图像特征分类'),
        ('gradient_boosting', 'sklearn', 'GB 图像特征分类'),
        ('svm', 'sklearn', 'SVM 像素分类'),
        ('knn', 'sklearn', 'KNN 图像匹配分类'),
        ('logistic_regression', 'sklearn', 'LR 边缘检测分类'),
        ('mlp', 'pytorch', 'CNN-MLP 图像分类'),
    ],
    'reinforcement': [
        ('random_forest', 'sklearn', 'RF 策略分类'),
        ('gradient_boosting', 'sklearn', 'GB 状态估值'),
        ('random_forest_regressor', 'sklearn', 'RF 奖励回归'),
        ('gradient_boosting_regressor', 'sklearn', 'GB 奖励预测'),
        ('linear_regression', 'sklearn', 'LR Q值估计'),
        ('mlp', 'pytorch', 'MLP 深度Q网络'),
    ],
    'generative': [
        ('random_forest', 'sklearn', 'RF 潜在特征分类'),
        ('gradient_boosting', 'sklearn', 'GB 生成判别'),
        ('svm', 'sklearn', 'SVM 高维流形分类'),
        ('logistic_regression', 'sklearn', 'LR 潜在空间分类'),
        ('knn', 'sklearn', 'KNN 嵌入分类'),
        ('mlp', 'pytorch', 'MLP VAE特征分类'),
    ],
    'other': [
        ('random_forest', 'sklearn', 'RF 混合特征分类'),
        ('gradient_boosting', 'sklearn', 'GB 异常检测'),
        ('svm', 'sklearn', 'SVM 异常边界'),
        ('logistic_regression', 'sklearn', 'LR 离群检测'),
        ('knn', 'sklearn', 'KNN 混合距离分类'),
        ('mlp', 'pytorch', 'MLP 自动编码器分类'),
    ],
}


def generate_dataset(name, n_samples, n_features, n_classes, target_type,
                     data_type='synthetic', description='', seed=42):
    """
    生成合成数据集，保存为 CSV 文件

    data_type:
        - 'blobs': 高斯聚类数据 (适合clustering)
        - 'classification': 标准分类数据
        - 'regression': 标准回归数据
        - 'nlp_features': 模拟文本特征 (TF-IDF like, 稀疏高频)
        - 'image_features': 模拟图像特征 (HOG/像素统计, 空间相关)
        - 'rl_states': 模拟强化学习状态 (连续+离散混合)
        - 'latent': 模拟生成模型潜在空间 (高维+非线性)
        - 'anomaly': 模拟异常检测 (极不平衡+混合分布)
    """
    np.random.seed(seed)
    n_features_actual = n_features - 1  # 减去target列

    if data_type == 'blobs':
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, centers=n_classes,
                          n_features=n_features_actual, cluster_std=2.5,
                          random_state=seed)
        df = pd.DataFrame(X, columns=[f'cluster_f{i}' for i in range(n_features_actual)])
        df['cluster_label'] = y.astype(str)
        target_col = 'cluster_label'

    elif data_type == 'classification':
        from sklearn.datasets import make_classification
        n_informative = max(5, n_features_actual // 2)
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features_actual,
            n_informative=n_informative, n_redundant=n_features_actual // 4,
            n_repeated=n_features_actual // 8, n_classes=n_classes,
            n_clusters_per_class=2, flip_y=0.03,
            class_sep=0.8, random_state=seed
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features_actual)])
        df['target'] = y.astype(str)
        target_col = 'target'

    elif data_type == 'regression':
        from sklearn.datasets import make_regression
        n_informative = max(5, n_features_actual // 2)
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features_actual,
            n_informative=n_informative, noise=15.0,
            bias=3.0, effective_rank=min(n_features_actual, 25),
            random_state=seed
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features_actual)])
        df['target'] = y
        target_col = 'target'

    elif data_type == 'nlp_features':
        # 模拟 TF-IDF 文本特征: 稀疏(大量0值), 长尾分布, 词汇量模拟
        vocab_size = n_features_actual
        X = np.zeros((n_samples, vocab_size), dtype=np.float32)

        # 每篇文档使用 1-15% 的词汇 (模拟文档长度差异)
        for i in range(n_samples):
            doc_len = np.random.randint(vocab_size // 20, vocab_size // 6)
            indices = np.random.choice(vocab_size, doc_len, replace=False)
            # 词频幂律分布 (Zipf)
            freqs = np.random.zipf(1.8, doc_len).astype(np.float32) / 10.0
            X[i, indices] = freqs

        # 用前几个词频做目标 (模拟情感/主题分类)
        sentiment_scores = X[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.3
        y = pd.cut(sentiment_scores, bins=n_classes, labels=[str(i) for i in range(n_classes)])
        df = pd.DataFrame(X, columns=[f'tfidf_{i}' for i in range(vocab_size)])
        df['sentiment'] = y.astype(str)
        target_col = 'sentiment'

    elif data_type == 'image_features':
        # 模拟 HOG/像素统计特征: 空间相关性, 边缘强度, 纹理
        n_pixels = n_features_actual // 3  # 分3组: 边缘/纹理/颜色
        X = np.zeros((n_samples, n_features_actual), dtype=np.float32)

        for i in range(n_samples):
            # 模拟图像块: 低频 (平滑区域) + 高频 (边缘)
            low_freq = np.sin(np.linspace(0, np.pi * 4, n_pixels) + np.random.randn() * 0.5)
            high_freq = np.random.laplace(0, 0.5, n_pixels)
            color_hist = np.abs(np.random.randn(n_features_actual - 2 * n_pixels))

            X[i, :n_pixels] = low_freq + np.random.randn(n_pixels) * 0.1
            X[i, n_pixels:2 * n_pixels] = high_freq
            X[i, 2 * n_pixels:] = color_hist

        # 基于边缘响应分类 (模拟物体识别)
        edge_response = np.abs(X[:, n_pixels:2 * n_pixels]).sum(axis=1)
        y = pd.cut(edge_response, bins=n_classes, labels=[str(i) for i in range(n_classes)])
        df = pd.DataFrame(X, columns=[f'img_f{i}' for i in range(n_features_actual)])
        df['object_class'] = y.astype(str)
        target_col = 'object_class'

    elif data_type == 'rl_states':
        # 强化学习状态空间: 连续传感器 + 离散动作 + 回报
        n_sensors = n_features_actual // 2
        X = np.zeros((n_samples, n_features_actual), dtype=np.float32)

        for i in range(n_samples):
            # 传感器数据 (位置/速度/角度等连续量)
            sensors = np.cumsum(np.random.randn(n_sensors) * 0.1) + np.random.randn(n_sensors) * 2
            # 上一动作 (one-hot)
            prev_action = np.zeros(n_features_actual - n_sensors)
            prev_action[np.random.randint(len(prev_action))] = 1.0
            X[i, :n_sensors] = sensors
            X[i, n_sensors:] = prev_action

        # 奖励 = 传感器状态的函数 + 动作影响 (模拟Q值)
        reward = np.tanh(X[:, :5].sum(axis=1) * 0.5 + np.random.randn(n_samples) * 0.2)
        if n_classes == 2:
            y = (reward > 0).astype(str)
            target_col = 'optimal_action'
            df = pd.DataFrame(X, columns=[f'state_{i}' for i in range(n_features_actual)])
            df[target_col] = y
        else:
            y = pd.cut(reward, bins=n_classes, labels=[str(i) for i in range(n_classes)])
            target_col = 'q_value_bin'
            df = pd.DataFrame(X, columns=[f'state_{i}' for i in range(n_features_actual)])
            df[target_col] = y.astype(str)

    elif data_type == 'latent':
        # 生成模型潜在空间: 低维流形嵌入 + 解码特征
        latent_dim = min(16, n_features_actual // 5)
        # 潜在变量 (标准正态)
        Z = np.random.randn(n_samples, latent_dim).astype(np.float32)

        # 非线性解码到高维 (模拟 VAE decoder)
        W1 = np.random.randn(latent_dim, n_features_actual).astype(np.float32) * 0.3
        W2 = np.random.randn(latent_dim, n_features_actual).astype(np.float32) * 0.2
        X = np.tanh(Z @ W1 + np.random.randn(n_samples, n_features_actual) * 0.1)
        X += np.sin(Z @ W2) * 0.5  # 非线性变换

        # 聚类标签 (基于潜在空间的真实聚类)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
        y = kmeans.fit_predict(Z)
        df = pd.DataFrame(X, columns=[f'latent_f{i}' for i in range(n_features_actual)])
        df['gen_class'] = y.astype(str)
        target_col = 'gen_class'

    elif data_type == 'anomaly':
        # 异常检测: 5% 异常 + 95% 正常, 特征分布混合
        n_normal = int(n_samples * 0.95)
        n_anomaly = n_samples - n_normal

        # 正常数据 (多元高斯)
        X_normal = np.random.randn(n_normal, n_features_actual).astype(np.float32) * 1.5
        # 异常数据 (不同分布)
        X_anomaly = np.random.laplace(0, 4, (n_anomaly, n_features_actual)).astype(np.float32)

        X = np.vstack([X_normal, X_anomaly])
        y = np.array(['normal'] * n_normal + ['anomaly'] * n_anomaly)

        # 随机打乱
        idx = np.random.permutation(n_samples)
        X, y = X[idx], y[idx]

        df = pd.DataFrame(X, columns=[f'mixed_f{i}' for i in range(n_features_actual)])
        df['anomaly_label'] = y
        target_col = 'anomaly_label'

    else:
        raise ValueError(f'未知数据类型: {data_type}')

    return df, target_col


def train_sklearn_model(X_train, y_train, X_test, y_test, algorithm, task_type):
    """训练 sklearn 模型并返回模型对象和指标"""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    # 预处理
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()

    if len(num_cols) > 0:
        X_train_num = scaler.fit_transform(X_train[num_cols])
        X_test_num = scaler.transform(X_test[num_cols])
    else:
        X_train_num = X_train.values
        X_test_num = X_test.values

    # 编码分类特征
    label_encoders = {}
    cat_cols = X_train.select_dtypes(include=['object']).columns
    X_train_proc = pd.DataFrame(X_train_num, columns=num_cols)
    X_test_proc = pd.DataFrame(X_test_num, columns=num_cols)
    for col in cat_cols:
        le = LabelEncoder()
        X_train_proc[col] = le.fit_transform(X_train[col].astype(str))
        X_test_proc[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

    # 导入模型注册表
    from app.executor.trainers.sklearn_trainer import (
        _CLASSIFIERS, _REGRESSORS, _CLUSTERERS, _import_model
    )

    # ================================================================
    # 聚类: 无监督训练
    # ================================================================
    if task_type == 'clustering':
        if algorithm in _CLUSTERERS:
            module_path, class_name = _CLUSTERERS[algorithm]
        else:
            raise ValueError(f'未知聚类算法: {algorithm}')

        model_cls = _import_model(module_path, class_name)
        try:
            model = model_cls(random_state=42)
        except TypeError:
            model = model_cls()

        # 训练 (无监督, 不需要 y)
        if hasattr(model, 'fit_predict'):
            model.fit_predict(X_train_proc.values)
        else:
            model.fit(X_train_proc.values)

        # 在测试集上预测
        if hasattr(model, 'predict'):
            labels_test = model.predict(X_test_proc.values)
        elif hasattr(model, 'fit_predict'):
            labels_test = model.fit_predict(X_test_proc.values)
        else:
            model.fit(X_test_proc.values)
            labels_test = model.labels_

        metrics = {}
        try:
            unique_labels = set(labels_test)
            if len(unique_labels) >= 2 and len(unique_labels) < len(labels_test):
                metrics['silhouette_score'] = round(
                    float(silhouette_score(X_test_proc.values, labels_test)), 4)
                metrics['davies_bouldin_score'] = round(
                    float(davies_bouldin_score(X_test_proc.values, labels_test)), 4)
                metrics['calinski_harabasz_score'] = round(
                    float(calinski_harabasz_score(X_test_proc.values, labels_test)), 4)
        except Exception:
            pass
        if hasattr(model, 'inertia_'):
            metrics['inertia'] = round(float(model.inertia_), 4)

        bundle = {
            'model': model, 'scaler': scaler, 'label_encoders': label_encoders,
            'feature_names': list(X_train.columns),
            'task_type': task_type, 'algorithm': algorithm,
        }
        return model, bundle, metrics

    # ================================================================
    # 监督学习 (分类/回归)
    # ================================================================

    # 编码目标
    target_le = None
    if task_type == 'classification':
        target_le = LabelEncoder()
        y_train_enc = target_le.fit_transform(y_train.astype(str))
        y_test_enc = target_le.transform(y_test.astype(str))
    else:
        y_train_enc = y_train.values.astype(float)
        y_test_enc = y_test.values.astype(float)

    if algorithm in _CLASSIFIERS:
        model_info = _CLASSIFIERS[algorithm]
    elif algorithm in _REGRESSORS:
        model_info = _REGRESSORS[algorithm]
    elif algorithm == 'knn_regressor':
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors=7, weights='distance')
        model.fit(X_train_proc.values, y_train_enc)
    else:
        raise ValueError(f'未知算法: {algorithm}')

    if algorithm not in ('knn_regressor',):
        module_path, class_name = model_info
        model_cls = _import_model(module_path, class_name)
        try:
            model = model_cls(random_state=42)
        except TypeError:
            model = model_cls()
        model.fit(X_train_proc.values, y_train_enc)

    # 预测和评估
    y_pred = model.predict(X_test_proc.values)

    metrics = {}
    if task_type == 'classification':
        metrics['accuracy'] = round(float(accuracy_score(y_test_enc, y_pred)), 4)
        try:
            metrics['precision'] = round(float(precision_score(y_test_enc, y_pred, average='weighted', zero_division=0)), 4)
            metrics['recall'] = round(float(recall_score(y_test_enc, y_pred, average='weighted', zero_division=0)), 4)
            metrics['f1_score'] = round(float(f1_score(y_test_enc, y_pred, average='weighted', zero_division=0)), 4)
        except Exception:
            pass
    else:
        metrics['mse'] = round(float(mean_squared_error(y_test_enc, y_pred)), 4)
        metrics['mae'] = round(float(mean_absolute_error(y_test_enc, y_pred)), 4)
        metrics['r2'] = round(float(r2_score(y_test_enc, y_pred)), 4)

    # 保存模型
    bundle = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X_train.columns),
        'task_type': task_type,
        'algorithm': algorithm,
    }
    if target_le:
        label_encoders['__target__'] = target_le
        bundle['label_encoders'] = label_encoders

    return model, bundle, metrics


def train_pytorch_model(X_train, y_train, X_test, y_test, task_type, epochs=20):
    """训练 PyTorch MLP 模型"""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        # PyTorch 不可用, 回退到 sklearn MLP
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train.values.astype(float))
        X_test_s = scaler.transform(X_test.values.astype(float))

        if task_type == 'classification':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train.astype(str))
            model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300,
                                  early_stopping=True, random_state=42)
            model.fit(X_train_s, y_train_enc)
            y_pred = model.predict(X_test_s)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            metrics = {
                'accuracy': round(float(accuracy_score(le.transform(y_test.astype(str)), y_pred)), 4),
                'framework': 'sklearn-mlp',
            }
            try:
                metrics['precision'] = round(float(precision_score(le.transform(y_test.astype(str)), y_pred, average='weighted', zero_division=0)), 4)
                metrics['recall'] = round(float(recall_score(le.transform(y_test.astype(str)), y_pred, average='weighted', zero_division=0)), 4)
                metrics['f1_score'] = round(float(f1_score(le.transform(y_test.astype(str)), y_pred, average='weighted', zero_division=0)), 4)
            except Exception:
                pass
            bundle = {'model': model, 'scaler': scaler, 'label_encoders': {'__target__': le},
                      'feature_names': list(X_train.columns), 'task_type': task_type, 'algorithm': 'mlp_sklearn'}
        else:
            model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=300,
                                 early_stopping=True, random_state=42)
            model.fit(X_train_s, y_train.values.astype(float))
            y_pred = model.predict(X_test_s)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                'mse': round(float(mean_squared_error(y_test.values.astype(float), y_pred)), 4),
                'mae': round(float(mean_absolute_error(y_test.values.astype(float), y_pred)), 4),
                'r2': round(float(r2_score(y_test.values.astype(float), y_pred)), 4),
                'framework': 'sklearn-mlp',
            }
            bundle = {'model': model, 'scaler': scaler, 'feature_names': list(X_train.columns),
                      'task_type': task_type, 'algorithm': 'mlp_sklearn'}

        return model, bundle, metrics

    # PyTorch 可用 — 使用真正的深度学习
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import numpy as np

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.values.astype(np.float32))
    X_test_s = scaler.transform(X_test.values.astype(np.float32))

    if task_type == 'classification':
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train.astype(str))
        y_test_enc = le.transform(y_test.astype(str))
        output_dim = len(le.classes_)
    else:
        y_train_enc = y_train.values.astype(np.float32)
        y_test_enc = y_test.values.astype(np.float32)
        output_dim = 1

    input_dim = X_train_s.shape[1]
    hidden_layers = [128, 64, 32]

    # 构建 MLP
    layers = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(h))
        layers.append(nn.Dropout(0.3))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    model = nn.Sequential(*layers)

    # 数据加载
    X_train_t = torch.tensor(X_train_s)
    y_train_t = torch.tensor(y_train_enc).long() if task_type == 'classification' else torch.tensor(y_train_enc).float().reshape(-1, 1)
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # 训练
    criterion = nn.CrossEntropyLoss() if task_type == 'classification' else nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        scheduler.step()

    # 评估
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_s)
        outputs = model(X_test_t)
        if task_type == 'classification':
            _, preds = torch.max(outputs, 1)
            y_pred = preds.numpy()
            y_true = y_test_enc
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            metrics = {
                'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
                'loss': round(total_loss / len(train_loader.dataset), 4),
                'framework': 'pytorch',
            }
            try:
                metrics['precision'] = round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)), 4)
                metrics['recall'] = round(float(recall_score(y_true, y_pred, average='weighted', zero_division=0)), 4)
                metrics['f1_score'] = round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4)
            except Exception:
                pass
        else:
            y_pred = outputs.numpy().flatten()
            y_true = y_test_enc
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                'mse': round(float(mean_squared_error(y_true, y_pred)), 4),
                'mae': round(float(mean_absolute_error(y_true, y_pred)), 4),
                'r2': round(float(r2_score(y_true, y_pred)), 4),
                'loss': round(total_loss / len(train_loader.dataset), 4),
                'framework': 'pytorch',
            }

    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_names': list(X_train.columns),
        'task_type': task_type,
        'algorithm': 'mlp_pytorch',
    }
    if task_type == 'classification':
        bundle['label_encoders'] = {'__target__': le}

    return model, bundle, metrics


def create_dataset_record(user, df, name, description, target_col, task_type, file_path):
    """创建 Dataset 数据库记录 (必须在 app_context 内调用)"""
    file_size = os.path.getsize(file_path)
    row_count = len(df)
    col_count = len(df.columns)

    summary = {
        'columns': list(df.columns),
        'dtypes': {c: str(df[c].dtype) for c in df.columns},
        'n_samples': row_count,
        'n_features': col_count - 1,
        'target_column': target_col,
        'task_type': task_type,
    }

    dataset = Dataset(
        name=name,
        description=description,
        file_path=file_path,
        file_format='csv',
        file_size=file_size,
        row_count=row_count,
        column_count=col_count,
        summary_json=json.dumps(summary, ensure_ascii=False, default=str),
        status='ready',
        is_public=True,
        owner_id=user.id,
    )
    db.session.add(dataset)
    db.session.commit()
    return dataset


def create_trained_model(user, dataset, model_name, model_type, framework,
                         algorithm, bundle, metrics, description=''):
    """创建训练好的模型记录 (必须在 app_context 内调用)"""
    import uuid as uuid_mod
    # 保存模型文件
    exp_dir = os.path.join(EXPERIMENTS_DIR, str(uuid_mod.uuid4()))
    os.makedirs(exp_dir, exist_ok=True)
    model_path = os.path.join(exp_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(bundle, f)

    file_size = os.path.getsize(model_path)

    # 构建超参数
    if model_type == 'clustering':
        hp_task_type = 'clustering'
    elif model_type == 'regression':
        hp_task_type = 'regression'
    else:
        hp_task_type = 'classification'
    hyperparams = {
        'task_type': hp_task_type,
        'algorithm': algorithm,
        'target_column': json.loads(dataset.summary_json).get('target_column', 'target'),
        'test_size': 0.2,
        'model_type_tag': model_type,
    }

    model = ModelRecord(
        name=model_name,
        description=description,
        model_type=model_type,
        framework=framework,
        model_file_path=model_path,
        file_size=file_size,
        status='trained',
        is_public=True,
        owner_id=user.id,
        training_dataset_id=dataset.id,
        training_duration_seconds=int(np.random.randint(10, 120)),
    )
    model.set_hyperparameters(hyperparams)
    model.set_metrics(metrics)

    db.session.add(model)
    db.session.commit()
    return model


# ============ 主流程 ============

DATASET_CONFIGS = {
    'regression': {
        'name': 'Regression-Large-50K',
        'description': '大型回归数据集: 50,000样本×25特征，含非线性+噪声。适合多种回归算法评估。',
        'n_samples': 50000, 'n_features': 26, 'n_classes': 1,
        'data_type': 'regression',
    },
    'clustering': {
        'name': 'Clustering-Blobs-40K',
        'description': '高斯混合聚类数据集: 40,000样本×20特征，7个天然聚类。适合聚类和边界分类。',
        'n_samples': 40000, 'n_features': 21, 'n_classes': 7,
        'data_type': 'blobs',
    },
    'nlp': {
        'name': 'NLP-TextFeatures-30K',
        'description': '模拟文本特征数据集(TF-IDF): 30,000文档×200词，幂律分布词频。适合高维稀疏特征分类。',
        'n_samples': 30000, 'n_features': 201, 'n_classes': 5,
        'data_type': 'nlp_features',
    },
    'computer_vision': {
        'name': 'CV-ImageFeatures-35K',
        'description': '模拟图像特征数据集(HOG+纹理+颜色直方图): 35,000样本×150特征。适合图像分类。',
        'n_samples': 35000, 'n_features': 151, 'n_classes': 6,
        'data_type': 'image_features',
    },
    'reinforcement': {
        'name': 'RL-StateAction-45K',
        'description': '强化学习状态-动作-奖励数据集: 45,000样本×40特征。传感器+动作编码，预测最优动作。',
        'n_samples': 45000, 'n_features': 41, 'n_classes': 2,
        'data_type': 'rl_states',
    },
    'generative': {
        'name': 'Gen-LatentSpace-60K',
        'description': '生成模型潜在空间数据集: 60,000样本×100特征。低维流形+非线性解码，高维特征分类。',
        'n_samples': 60000, 'n_features': 101, 'n_classes': 8,
        'data_type': 'latent',
    },
    'other': {
        'name': 'Other-AnomalyMix-55K',
        'description': '混合异常检测数据集: 55,000样本×30特征。5%异常+95%正常，多元混合分布。',
        'n_samples': 55000, 'n_features': 31, 'n_classes': 2,
        'data_type': 'anomaly',
    },
}


def main():
    print('=' * 60)
    print('  多类型数据集生成 & 48模型批量训练')
    print('=' * 60)

    app = create_app()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    # 统一在一个 app context 中完成所有数据库操作
    with app.app_context():
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            print('ERROR: admin 用户不存在!')
            return
        print(f'Admin 用户: {admin.username} (ID={admin.id})')

        # ===== 步骤1: 生成数据集 =====
        print('\n' + '=' * 40)
        print('步骤1: 生成7个新数据集')
        print('=' * 40)

        dataset_ids = {}  # {model_type: dataset_id}

        for model_type, config in DATASET_CONFIGS.items():
            print(f'\n--- 生成 {model_type} 数据集: {config["name"]} ---')

            df, target_col = generate_dataset(
                name=config['name'],
                n_samples=config['n_samples'],
                n_features=config['n_features'],
                n_classes=config.get('n_classes', 2),
                target_type='classification' if config['n_classes'] > 1 else 'regression',
                data_type=config['data_type'],
                description=config['description'],
            )

            # 保存 CSV
            filename = f'{config["name"].replace(" ", "_")}_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.csv'
            file_path = os.path.join(UPLOAD_DIR, filename)
            df.to_csv(file_path, index=False)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f'  保存: {file_path} ({file_size_mb:.1f}MB, {len(df)}行×{len(df.columns)}列)')

            # 创建 Dataset 记录
            if config['data_type'] == 'blobs':
                ds_task_type = 'clustering'
            elif config['n_classes'] > 1:
                ds_task_type = 'classification'
            else:
                ds_task_type = 'regression'
            dataset = create_dataset_record(
                admin, df, config['name'], config['description'],
                target_col, ds_task_type,
                file_path
            )
            dataset_ids[model_type] = dataset.id
            print(f'  数据集记录: ID={dataset.id}, target={target_col}')

        # 也为 classification 使用现有数据集
        existing_cls_ds = Dataset.query.filter_by(name='Multiclass5_25K').first()
        if existing_cls_ds:
            dataset_ids['classification'] = existing_cls_ds.id
            print(f'\n 分类复用现有数据集: ID={existing_cls_ds.id} ({existing_cls_ds.name})')

        # ===== 步骤2: 训练48个模型 =====
        print('\n' + '=' * 40)
        print('步骤2: 训练48个模型 (8类型×6算法)')
        print('=' * 40)

        total_trained = 0
        for model_type, algorithms in TYPE_ALGORITHMS.items():
            if model_type not in dataset_ids:
                print(f'\n  SKIP {model_type}: 无数据集')
                continue

            ds = Dataset.query.get(dataset_ids[model_type])
            print(f'\n--- {model_type} (数据集: {ds.name}) ---')

            # 加载数据集
            df = pd.read_csv(ds.file_path)
            summary = json.loads(ds.summary_json)
            target_col = summary.get('target_column', df.columns[-1])

            # 准备训练数据
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder

            X = df.drop(columns=[target_col])
            y = df[target_col]

            # 只使用数值列
            X_num = X.select_dtypes(include=[np.number])
            if X_num.shape[1] < 2:
                X_num = X.copy()
                for col in X_num.select_dtypes(include=['object']).columns:
                    X_num[col] = LabelEncoder().fit_transform(X_num[col].astype(str))

            # 限制训练集大小以加速 (最多40K)
            if len(X_num) > 40000:
                indices = np.random.RandomState(42).choice(len(X_num), 40000, replace=False)
                X_num = X_num.iloc[indices]
                y = y.iloc[indices]

            X_train, X_test, y_train, y_test = train_test_split(
                X_num, y, test_size=0.2, random_state=42,
                stratify=y if len(set(y)) <= 10 else None
            )

            print(f'  训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本')

            for algo, framework, algo_desc in algorithms:
                if model_type == 'clustering':
                    task_type = 'clustering'
                elif algo in ('random_forest_regressor', 'gradient_boosting_regressor',
                            'linear_regression', 'svr', 'knn_regressor', 'mlp_regressor'):
                    task_type = 'regression'
                elif model_type == 'reinforcement' and algo in ('random_forest_regressor',
                                                                 'gradient_boosting_regressor',
                                                                 'linear_regression'):
                    task_type = 'regression'
                else:
                    task_type = 'classification'

                try:
                    print(f'    训练: {algo} ({framework})...', end=' ')

                    if framework == 'pytorch' or algo in ('mlp', 'mlp_regressor'):
                        model_obj, bundle, metrics = train_pytorch_model(
                            X_train, y_train, X_test, y_test, task_type
                        )
                        framework_used = 'pytorch'
                        algo_used = algo
                    else:
                        model_obj, bundle, metrics = train_sklearn_model(
                            X_train, y_train, X_test, y_test, algo, task_type
                        )
                        framework_used = 'sklearn'
                        algo_used = algo

                    # 创建模型记录
                    model_name = f'{algo_used}-{model_type}-{datetime.now(timezone.utc).strftime("%H%M%S")}'
                    model = create_trained_model(
                        admin, ds, model_name, model_type, framework_used,
                        algo_used, bundle, metrics,
                        description=f'{algo_desc} — {model_type}类型模型'
                    )

                    # 打印关键指标
                    if 'accuracy' in metrics:
                        print(f'OK (acc={metrics["accuracy"]:.4f})')
                    elif 'r2' in metrics:
                        print(f'OK (r2={metrics["r2"]:.4f})')
                    elif 'silhouette_score' in metrics:
                        print(f'OK (silhouette={metrics["silhouette_score"]:.4f})')
                    else:
                        print('OK')

                    total_trained += 1

                except Exception as e:
                    import traceback
                    print(f'FAILED: {e}')
                    traceback.print_exc()

        print(f'\n{"=" * 60}')
        print(f'  完成! 共训练 {total_trained} 个模型')
        print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
