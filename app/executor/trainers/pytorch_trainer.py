"""
PyTorch 训练器 v3
支持 MLP 全连接网络，GPU训练，验证集+早停，智能模型缩放，完整保存/加载

v3 改进 (2026-06-05):
  - 3-way 分割: train/val/test，val 用于早停和 LR 调度
  - 智能模型规模: 根据 n_samples 和 n_features 自动缩放 hidden_layers
  - Early Stopping: 监控 val_loss，patience=10 轮无改善自动停止
  - ReduceLROnPlateau: val_loss 停滞时自动降学习率 (factor=0.5)
  - 每个 epoch 后报告 train + val 双指标，避免误导
  - 更安全默认值: lr=1e-4, dropout=0.5, weight_decay=1e-3
"""
import os
import pickle
import numpy as np
import pandas as pd
import json
import copy

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


# ============ 智能模型规模推断 ============

def _auto_hidden_layers(n_samples: int, n_features: int, n_classes: int = 2) -> list:
    """
    根据数据集规模自动推断合理的隐藏层结构

    设计原则 (防止过拟合):
      - 样本/参数比 > 10 是安全的 (每个参数至少有10个样本学习)
      - 小数据集 (< 5000): 使用浅窄网络 [min(features//2, 64), min(features//4, 32)]
      - 中等数据集 (< 20000): 使用中等网络 [128, 64]
      - 大数据集 (>= 20000): 使用 [256, 128, 64]
      - 超大 (>= 100000): [512, 256, 128, 64]

    原默认 [512,256,128,64] 参数量 277K，对 24000 样本过拟合严重。
    新默认 [128,64] 参数量约 10-20K，样本/参数比 > 100。

    Returns:
        list of hidden layer sizes
    """
    if n_samples < 3000:
        # 极小数据集: 最浅网络，甚至只用一个隐层
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


# ============ PyTorch 模型定义 (工厂函数) ============

def create_mlp_classifier(input_dim, hidden_layers, output_dim, dropout=0.5):
    """创建 MLP 分类网络 (使用 Sequential 工厂函数)

    结构: Input → [Linear → ReLU → BatchNorm → Dropout] × N → Linear → logits
    - BatchNorm1d: 加速收敛, 稳定训练
    - Dropout: 随机丢弃神经元, 防止过拟合 (默认 0.5)
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


def create_mlp_regressor(input_dim, hidden_layers, dropout=0.5):
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
    dropout = config.get('dropout', 0.5)

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
    """PyTorch 深度学习训练器 v3 — MLP 全连接神经网络

    防过拟合机制:
        - 智能模型规模: 根据 n_samples 自动缩放 hidden_layers
        - 3-way 分割: train(68%) / val(12%) / test(20%)
        - Early Stopping: 监控 val_loss，patience 轮无改善自动停止
        - ReduceLROnPlateau: val_loss 停滞 → 学习率减半 (factor=0.5, patience=5)
        - 梯度裁剪 (max_norm=1.0) → 防止梯度爆炸
        - AdamW 优化器 (解耦权重衰减 = 更好的 L2 正则化)
        - CUDA GPU 自动检测加速
        - 完整 save/load 流程: .pt 权重 + _config.pkl 配置
    """

    def __init__(self, job, dataset, hyperparams: dict = None):
        super().__init__(job, dataset, hyperparams)

        self.task_type = self.hyperparams.get('task_type', 'classification')
        # 智能模型规模: 在 load_data 中根据数据量自动计算 (如果用户没显式指定)
        self.hidden_layers = self.hyperparams.get('hidden_layers', None)  # None = 自动推断

        # 更安全的默认超参数 (防过拟合)
        self.learning_rate = float(self.hyperparams.get('learning_rate', 1e-4))
        self.batch_size = int(self.hyperparams.get('batch_size', 64))
        self.test_size = float(self.hyperparams.get('test_size', 0.2))
        self.val_size = float(self.hyperparams.get('val_size', 0.15))  # 新增: 验证集比例
        self.dropout = float(self.hyperparams.get('dropout', 0.5))
        self.weight_decay = float(self.hyperparams.get('weight_decay', 1e-3))
        # 早停参数
        self.early_stopping_patience = int(self.hyperparams.get('early_stopping_patience', 10))
        # LR 调度参数
        self.lr_patience = int(self.hyperparams.get('lr_patience', 5))
        self.lr_factor = float(self.hyperparams.get('lr_factor', 0.5))

        self._model = None
        # 三个 DataLoader: train / val / test
        self._train_loader = self._val_loader = self._test_loader = None
        self._input_dim = self._output_dim = None
        self._criterion = self._optimizer = self._scheduler = None
        self._lr_scheduler = None  # ReduceLROnPlateau
        self._scaler = None
        self._y_scaler = None
        self._label_encoders = {}
        self._feature_names = []
        self._device = None
        # 保存完整测试集用于最终评估
        self._X_test_tensor = self._y_test_tensor = None
        # 早停追踪
        self._best_val_loss = float('inf')
        self._best_model_state = None
        self._patience_counter = 0
        self._stopped_early = False
        self._early_stop = False
        self._best_val_metric = None

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

        # —— 智能模型规模推断 ——
        if self.hidden_layers is None:
            n_classes = self._output_dim if self.task_type == 'classification' else 2
            self.hidden_layers = _auto_hidden_layers(n_samples, self._input_dim, n_classes)
            self.callback.on_log(f'智能模型规模: {self.hidden_layers} '
                               f'(样本={n_samples}, 特征={self._input_dim})')
        else:
            self.callback.on_log(f'用户指定网络结构: {self.hidden_layers}')

        # —— 3-way 分割: train / val / test ——
        # Step 1: 分出 test 集
        stratify_y = y if self.task_type == 'classification' and len(set(y)) > 1 else None
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_num, y, test_size=self.test_size, random_state=42,
            stratify=stratify_y
        )

        # Step 2: 从 train_val 中分出 val 集
        # val_size 是占 train_val 的比例 (例如 0.15 / 0.80 ≈ 0.1875)
        val_ratio = self.val_size / (1.0 - self.test_size)
        stratify_tv = y_train_val if self.task_type == 'classification' and len(set(y_train_val)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=42,
            stratify=stratify_tv
        )

        # —— 转为 Tensor ——
        def _to_tensor(x_data, y_data, is_classification):
            x_t = torch.tensor(np.asarray(x_data, dtype='float32'), dtype=torch.float32)
            if is_classification:
                y_t = torch.tensor(np.asarray(y_data, dtype='int64'), dtype=torch.long)
            else:
                y_t = torch.tensor(np.asarray(y_data, dtype='float32'), dtype=torch.float32).reshape(-1, 1)
            return TensorDataset(x_t, y_t)

        train_ds = _to_tensor(X_train, y_train, self.task_type == 'classification')
        val_ds = _to_tensor(X_val, y_val, self.task_type == 'classification')
        test_ds = _to_tensor(X_test, y_test, self.task_type == 'classification')

        self._train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self._val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        self._test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        # 保存测试集 Tensor 用于最终评估
        self._X_test_tensor = torch.tensor(np.asarray(X_test, dtype='float32'), dtype=torch.float32)
        if self.task_type == 'classification':
            self._y_test_tensor = torch.tensor(np.asarray(y_test, dtype='int64'), dtype=torch.long)
        else:
            self._y_test_tensor = torch.tensor(np.asarray(y_test, dtype='float32'), dtype=torch.float32).reshape(-1, 1)

        self.callback.on_log(f'输入维度: {self._input_dim}, 输出维度: {self._output_dim}')
        self.callback.on_log(f'训练: {len(train_ds)} | 验证: {len(val_ds)} | 测试: {len(test_ds)}')
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
        # ReduceLROnPlateau: 监控 val_loss，停滞时学习率减半
        self._lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode='min', factor=self.lr_factor,
            patience=self.lr_patience
        )

        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        self.callback.on_log(f'参数: 总={total_params:,} 可训练={trainable_params:,} '
                           f'lr={self.learning_rate} dropout={self.dropout} wd={self.weight_decay}')

    def _eval_loader(self, loader) -> dict:
        """在一个 DataLoader 上评估模型 (train/val/test 通用)"""
        torch, _, _, _, _ = _ensure_torch()
        self._model.eval()
        total_loss = 0.0
        correct = total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self._device), batch_y.to(self._device)
                outputs = self._model(batch_x)
                loss = self._criterion(outputs, batch_y)
                total_loss += loss.item() * len(batch_x)

                if self.task_type == 'classification':
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == batch_y).sum().item()
                    total += len(batch_y)

        n = len(loader.dataset)
        result = {'loss': round(total_loss / n, 4)}
        if self.task_type == 'classification' and total > 0:
            result['accuracy'] = round(correct / total, 4)
        return result

    def train_epoch(self, epoch: int) -> dict:
        """训练一个 epoch，返回 train_metrics + val_metrics"""
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

        # 训练集指标
        n_train = len(self._train_loader.dataset)
        train_metrics = {'train_loss': round(total_loss / n_train, 4)}
        if self.task_type == 'classification' and total > 0:
            train_metrics['train_accuracy'] = round(correct / total, 4)

        # 验证集指标
        val_metrics = self._eval_loader(self._val_loader)
        # 加上 val_ 前缀，与 train_ 区分
        val_prefixed = {f'val_{k}': v for k, v in val_metrics.items()}

        # —— 早停 & LR 调度 (基于 val_loss) ——
        current_val_loss = val_metrics['loss']

        # ReduceLROnPlateau
        self._lr_scheduler.step(current_val_loss)
        current_lr = self._optimizer.param_groups[0]['lr']

        # Early Stopping
        if current_val_loss < self._best_val_loss - 1e-4:
            self._best_val_loss = current_val_loss
            self._patience_counter = 0
            # 保存最佳模型权重 (deep copy)
            self._best_model_state = copy.deepcopy(self._model.state_dict())
        else:
            self._patience_counter += 1
            if self._patience_counter >= self.early_stopping_patience:
                self._early_stop = True
                self._stopped_early = True
                self._best_val_metric = round(self._best_val_loss, 4)

        # 合并返回
        result = {**train_metrics, **val_prefixed}
        result['lr'] = current_lr  # 记录当前学习率
        return result

    def evaluate(self) -> dict:
        """最终评估: 恢复最佳模型 → 在测试集上计算最终指标"""
        torch, _, _, _, _ = _ensure_torch()
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                      f1_score, r2_score, mean_squared_error)

        # —— 恢复早停最佳模型 ——
        if self._best_model_state is not None:
            self._model.load_state_dict(self._best_model_state)
            self.callback.on_log(f'已恢复最佳模型 (val_loss={self._best_val_loss:.4f})')

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
            result['test_accuracy'] = round(float(accuracy_score(all_labels, all_preds)), 4)
            try:
                result['test_precision_weighted'] = round(float(precision_score(all_labels, all_preds, average='weighted', zero_division=0)), 4)
                result['test_recall_weighted'] = round(float(recall_score(all_labels, all_preds, average='weighted', zero_division=0)), 4)
                result['test_f1_weighted'] = round(float(f1_score(all_labels, all_preds, average='weighted', zero_division=0)), 4)
                result['test_precision_macro'] = round(float(precision_score(all_labels, all_preds, average='macro', zero_division=0)), 4)
                result['test_recall_macro'] = round(float(recall_score(all_labels, all_preds, average='macro', zero_division=0)), 4)
                result['test_f1_macro'] = round(float(f1_score(all_labels, all_preds, average='macro', zero_division=0)), 4)
            except Exception:
                pass
        else:
            all_preds_arr = np.asarray(all_preds, dtype='float32').reshape(-1, 1)
            all_labels_arr = np.asarray(all_labels, dtype='float32').reshape(-1, 1)
            if self._y_scaler is not None:
                all_preds_arr = self._y_scaler.inverse_transform(all_preds_arr)
                all_labels_arr = self._y_scaler.inverse_transform(all_labels_arr)
            result['test_mse'] = round(float(mean_squared_error(all_labels_arr, all_preds_arr)), 4)
            result['test_r2'] = round(float(r2_score(all_labels_arr, all_preds_arr)), 4)

        # 附加上下文信息
        if self._stopped_early:
            result['early_stopped'] = True
            result['best_val_loss'] = round(self._best_val_loss, 4)
        return result

    def save_model(self, path: str):
        torch, _, _, _, _ = _ensure_torch()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 保存 PyTorch 模型权重 (使用最佳模型)
        state = self._best_model_state if self._best_model_state is not None else self._model.state_dict()
        torch.save(state, path + '.pt')

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
            'val_size': self.val_size,
            'early_stopping_patience': self.early_stopping_patience,
        }
        with open(path + '_config.pkl', 'wb') as f:
            pickle.dump(config, f)

        self.callback.on_log(f'模型已保存到: {path}.pt + _config.pkl')

    # ============ 检查点 ============

    def save_checkpoint(self):
        """保存 PyTorch 训练快照: 模型权重 + 优化器 + 调度器 + epoch + 早停状态"""
        if self._model is None:
            return
        torch, _, _, _, _ = _ensure_torch()
        os.makedirs(self.output_dir, exist_ok=True)

        ckpt = {
            'model_state': self._model.state_dict(),
            'optimizer_state': self._optimizer.state_dict(),
            'scheduler_state': self._lr_scheduler.state_dict(),
            'epoch': self._current_epoch + 1,
            'best_val_loss': self._best_val_loss,
            'patience_counter': self._patience_counter,
            'best_model_state': self._best_model_state,
            'input_dim': self._input_dim,
            'output_dim': self._output_dim,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
            'task_type': self.task_type,
        }
        ckpt_path = os.path.join(self.output_dir, 'checkpoint.pt')
        torch.save(ckpt, ckpt_path)

    @staticmethod
    def load_checkpoint(output_dir: str) -> dict:
        ckpt_path = os.path.join(output_dir, 'checkpoint.pt')
        if not os.path.exists(ckpt_path):
            return {}
        torch, _, _, _, _ = _ensure_torch()
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        return {'epoch': ckpt.get('epoch', 0)}

    @staticmethod
    def has_checkpoint(output_dir: str) -> bool:
        return os.path.exists(os.path.join(output_dir, 'checkpoint.pt'))
