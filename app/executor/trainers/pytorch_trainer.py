"""
PyTorch 训练器 v2
支持 MLP 全连接网络，GPU训练，完整保存/加载
"""
import os
import pickle
import numpy as np
import pandas as pd
import json

from app.executor.trainers.base import BaseTrainer


# ============ PyTorch 延迟导入 ============
# 延迟导入策略: 仅在首次使用时加载 torch，避免非 GPU 环境加载失败
# 模块级变量在 _ensure_torch() 首次调用后被缓存填充

_torch = None       # torch 主模块
_nn = None          # torch.nn — 神经网络层
_optim = None       # torch.optim — 优化器
_DataLoader = None  # DataLoader — 批量数据加载
_TensorDataset = None # TensorDataset — 特征/标签张量包装


def _ensure_torch():
    """延迟导入 PyTorch (仅在需要时)"""
    global _torch, _nn, _optim, _DataLoader, _TensorDataset
    if _torch is not None:
        return _torch, _nn, _optim, _DataLoader, _TensorDataset
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        _torch = torch
        _nn = nn
        _optim = optim
        _DataLoader = DataLoader
        _TensorDataset = TensorDataset
        return torch, nn, optim, DataLoader, TensorDataset
    except ImportError:
        raise ImportError(
            'PyTorch 未安装。请运行: pip install torch torchvision\n'
            '或使用 scikit-learn 训练器。'
        )


# ============ PyTorch 模型定义 (工厂函数) ============

def create_mlp_classifier(input_dim, hidden_layers, output_dim, dropout=0.2):
    """创建 MLP 分类网络 (使用 Sequential 工厂函数)

    结构: Input → [Linear → ReLU → BatchNorm → Dropout] × N → Linear → logits
    - BatchNorm1d: 加速收敛, 稳定训练
    - Dropout: 随机丢弃神经元, 防止过拟合
    - 输出维度 = 类别数, 配合 CrossEntropyLoss 使用
    """
    _, nn, _, _, _ = _ensure_torch()
    layers = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(h))
        layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers), input_dim, hidden_layers, output_dim


def create_mlp_regressor(input_dim, hidden_layers, dropout=0.2):
    """创建 MLP 回归网络

    结构: Input → [Linear → ReLU → BatchNorm → Dropout] × N → Linear → 1
    - 输出为单一连续值, 配合 MSELoss 使用
    """
    _, nn, _, _, _ = _ensure_torch()
    layers = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(h))
        layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers), input_dim, hidden_layers


def load_mlp_model(model_path, config_path):
    """从文件加载 PyTorch 模型"""
    torch, nn, _, _, _ = _ensure_torch()
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    input_dim = config['input_dim']
    hidden_layers = config['hidden_layers']
    task_type = config.get('task_type', 'classification')
    dropout = config.get('dropout', 0.2)

    if task_type == 'classification':
        output_dim = config['output_dim']
        model, _, _, _ = create_mlp_classifier(input_dim, hidden_layers, output_dim, dropout)
    else:
        model, _, _ = create_mlp_regressor(input_dim, hidden_layers, dropout)

    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model, config


# ============ 训练器 ============

class PyTorchTrainer(BaseTrainer):
    """PyTorch 深度学习训练器 — MLP 全连接神经网络

    架构特性:
        - 多层全连接 (Linear + ReLU + BatchNorm + Dropout) → 防过拟合
        - AdamW 优化器 (解耦权重衰减) + CosineAnnealingWarmRestarts 学习率调度
        - 梯度裁剪 (max_norm=1.0) → 防止梯度爆炸
        - CUDA GPU 自动检测加速
        - 完整 save/load 流程: .pt 权重 + _config.pkl 配置
    """

    def __init__(self, job, dataset, hyperparams: dict = None):
        super().__init__(job, dataset, hyperparams)

        self.task_type = self.hyperparams.get('task_type', 'classification')
        self.hidden_layers = self.hyperparams.get('hidden_layers', [128, 64, 32])
        self.learning_rate = float(self.hyperparams.get('learning_rate', 0.001))
        self.batch_size = int(self.hyperparams.get('batch_size', 64))
        self.test_size = float(self.hyperparams.get('test_size', 0.2))
        self.dropout = float(self.hyperparams.get('dropout', 0.3))
        self.weight_decay = float(self.hyperparams.get('weight_decay', 1e-5))

        self._model = None
        self._train_loader = self._test_loader = None
        self._input_dim = self._output_dim = None
        self._criterion = self._optimizer = self._scheduler = None
        self._scaler = None
        self._y_scaler = None  # 回归目标标准化器
        self._label_encoders = {}
        self._feature_names = []
        self._device = None
        self._X_test_tensor = self._y_test_tensor = None

    def _get_device(self):
        torch, _, _, _, _ = _ensure_torch()
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def load_data(self):
        torch, nn, optim, DataLoader, TensorDataset = _ensure_torch()
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.impute import SimpleImputer

        self._device = self._get_device()
        self.callback.on_log(f'设备: {self._device}')

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
        elif fmt == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f'不支持的文件格式: {fmt}')

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
            # 回归: 对目标值做标准化，大幅提升 MLP 收敛效果 (R² 从 ~0 提升到 >0.7)
            y_raw_vals = y_raw.values.astype(np.float32).reshape(-1, 1)
            self._y_scaler = StandardScaler()
            y = self._y_scaler.fit_transform(y_raw_vals).ravel().astype(np.float32)
            self._output_dim = 1

        # 标准化
        X_num = X.values.astype(np.float32)
        self._scaler = StandardScaler()
        X_num = self._scaler.fit_transform(X_num).astype(np.float32)

        self._input_dim = X_num.shape[1]

        # 划分
        X_train, X_test, y_train, y_test = train_test_split(
            X_num, y, test_size=self.test_size, random_state=42,
            stratify=y if self.task_type == 'classification' and len(set(y)) > 1 else None
        )

        # 转为 Tensor (显式指定 dtype，避免 numpy 2.x 兼容性问题)
        if self.task_type == 'classification':
            y_train_t = torch.tensor(np.asarray(y_train, dtype=np.int64), dtype=torch.long)
            y_test_t = torch.tensor(np.asarray(y_test, dtype=np.int64), dtype=torch.long)
        else:
            y_train_t = torch.tensor(np.asarray(y_train, dtype=np.float32), dtype=torch.float32).reshape(-1, 1)
            y_test_t = torch.tensor(np.asarray(y_test, dtype=np.float32), dtype=torch.float32).reshape(-1, 1)

        train_ds = TensorDataset(
            torch.tensor(np.asarray(X_train, dtype=np.float32), dtype=torch.float32),
            y_train_t
        )
        test_ds = TensorDataset(
            torch.tensor(np.asarray(X_test, dtype=np.float32), dtype=torch.float32),
            y_test_t
        )

        self._train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self._test_loader = DataLoader(test_ds, batch_size=self.batch_size)

        # 保存测试集用于最终评估
        self._X_test_tensor = torch.tensor(np.asarray(X_test, dtype=np.float32), dtype=torch.float32)
        self._y_test_tensor = y_test_t

        self.callback.on_log(f'输入维度: {self._input_dim}, 输出维度: {self._output_dim}')
        self.callback.on_log(f'训练: {len(train_ds)} 样本, 测试: {len(test_ds)} 样本')
        self.callback.on_log(f'网络结构: {self.hidden_layers}')

    def build_model(self):
        torch, nn, optim, _, _ = _ensure_torch()

        if self.task_type == 'classification':
            self._model, _, _, _ = create_mlp_classifier(
                self._input_dim, self.hidden_layers,
                self._output_dim, self.dropout
            )
            self._criterion = nn.CrossEntropyLoss()
        else:
            self._model, _, _ = create_mlp_regressor(
                self._input_dim, self.hidden_layers, self.dropout
            )
            self._criterion = nn.MSELoss()

        self._model = self._model.to(self._device)
        self._optimizer = optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self._scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self._optimizer, T_0=10, T_mult=2
        )

        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        self.callback.on_log(f'总参数: {total_params:,}, 可训练: {trainable_params:,}')

    def train_epoch(self, epoch: int) -> dict:
        torch, _, _, _, _ = _ensure_torch()
        self._model.train()
        total_loss = 0.0
        correct = total = 0

        for batch_x, batch_y in self._train_loader:
            batch_x, batch_y = batch_x.to(self._device), batch_y.to(self._device)

            self._optimizer.zero_grad()
            outputs = self._model(batch_x)
            loss = self._criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimizer.step()

            total_loss += loss.item() * len(batch_x)
            if self.task_type == 'classification':
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += len(batch_y)

        self._scheduler.step()

        avg_loss = total_loss / len(self._train_loader.dataset)
        result = {'loss': round(avg_loss, 4)}
        if self.task_type == 'classification' and total > 0:
            result['accuracy'] = round(correct / total, 4)
        return result

    def evaluate(self) -> dict:
        torch, _, _, _, _ = _ensure_torch()
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

        self._model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in self._test_loader:
                batch_x = batch_x.to(self._device)
                outputs = self._model(batch_x)

                if self.task_type == 'classification':
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().tolist())
                else:
                    all_preds.extend(outputs.cpu().tolist())
                all_labels.extend(batch_y.cpu().tolist())

        result = {}
        if self.task_type == 'classification':
            result['accuracy'] = round(float(accuracy_score(all_labels, all_preds)), 4)
            try:
                # weighted 平均 — 按样本数加权
                result['precision_weighted'] = round(float(precision_score(all_labels, all_preds, average='weighted', zero_division=0)), 4)
                result['recall_weighted'] = round(float(recall_score(all_labels, all_preds, average='weighted', zero_division=0)), 4)
                result['f1_weighted'] = round(float(f1_score(all_labels, all_preds, average='weighted', zero_division=0)), 4)
                # macro 平均 — 各类别等权, 暴露类别间差异
                result['precision_macro'] = round(float(precision_score(all_labels, all_preds, average='macro', zero_division=0)), 4)
                result['recall_macro'] = round(float(recall_score(all_labels, all_preds, average='macro', zero_division=0)), 4)
                result['f1_macro'] = round(float(f1_score(all_labels, all_preds, average='macro', zero_division=0)), 4)
            except Exception:
                pass
        else:
            # 回归: 反标准化预测值和标签，计算真实尺度下的指标
            all_preds_arr = np.asarray(all_preds, dtype=np.float32).reshape(-1, 1)
            all_labels_arr = np.asarray(all_labels, dtype=np.float32).reshape(-1, 1)
            if self._y_scaler is not None:
                all_preds_arr = self._y_scaler.inverse_transform(all_preds_arr)
                all_labels_arr = self._y_scaler.inverse_transform(all_labels_arr)
            result['mse'] = round(float(mean_squared_error(all_labels_arr, all_preds_arr)), 4)
            result['r2'] = round(float(r2_score(all_labels_arr, all_preds_arr)), 4)
        return result

    def save_model(self, path: str):
        torch, _, _, _, _ = _ensure_torch()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 保存 PyTorch 模型权重
        torch.save(self._model.state_dict(), path + '.pt')

        # 保存配置和预处理器
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
        }
        with open(path + '_config.pkl', 'wb') as f:
            pickle.dump(config, f)

        self.callback.on_log(f'模型已保存到: {path}.pt + _config.pkl')
