"""
============================================
HuggingFace Transformers NLP 训练器 v2
基于预训练模型 (BERT/DistilBERT) 的迁移学习微调

v2 改进 (2026-06-20):
  - 3-way 分割: train/val/test，val 用于早停
  - Early Stopping: 监控 val_loss，patience 轮无改善自动停止
  - Best model 持久化: 早停时保存最佳权重，最终评估恢复最佳模型
  - 每个 epoch 后报告 train + val 双指标
============================================
"""
import os
import json
import copy
import numpy as np
import pandas as pd

from app.executor.trainers.base import BaseTrainer

# 国内HF镜像 (huggingface.co 被墙时自动切换)
_HF_MIRROR = 'https://hf-mirror.com'

# 语言 → 预训练模型映射
LANGUAGE_MODEL_MAP = {
    'zh': 'bert-base-chinese',          # 中文: BERT-base (110M参数)
    'en': 'distilbert-base-uncased',    # 英文: DistilBERT (66M, 轻量)
    'multi': 'bert-base-multilingual-cased',  # 多语言
}

# 默认配置
DEFAULT_MODEL = 'bert-base-chinese'
MAX_LENGTH = 256        # 最大token长度
BATCH_SIZE = 16         # CPU友好batch
LEARNING_RATE = 2e-5    # BERT推荐学习率
EPOCHS = 3              # 微调3-5轮
WARMUP_STEPS = 100


def _ensure_hf_access():
    """确保 HuggingFace 可访问 (国内自动切镜像)"""
    # 如果已经设置过，直接返回
    if os.environ.get('HF_ENDPOINT'):
        return
    # 尝试直连，失败则切镜像
    try:
        import urllib.request
        req = urllib.request.Request('https://huggingface.co', method='HEAD')
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        os.environ['HF_ENDPOINT'] = _HF_MIRROR


def _download_model(model_name: str):
    """下载预训练模型 (含tokenizer + model)，自动重试镜像"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    _ensure_hf_access()

    # 先尝试从本地缓存加载
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True
        )
        # local_files_only 成功 → 已缓存
        return tokenizer
    except Exception:
        pass

    # 在线下载 (自动使用镜像)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        raise RuntimeError(
            f'无法下载模型 {model_name}。请手动下载到本地后使用 model_name 参数指定路径。\n'
            f'错误: {e}\n'
            f'或运行: export HF_ENDPOINT=https://hf-mirror.com'
        )
class TransformersNLPTrainer(BaseTrainer):
    """NLP Transformer 微调训练器

    使用 HuggingFace Transformers 库:
    - 自动检测文本语言 → 选择对应预训练模型
    - Tokenizer 分词 + DataLoader 批处理
    - 冻结底层 + 微调顶层 (节省GPU内存)
    - AdamW 优化器 + 线性学习率衰减
    - 保存模型 + tokenizer 以便后续推理

    适用场景:
    - NLP 文本分类 (情感分析/主题分类/意图识别等)
    - category='nlp' 的数据集
    - text_heavy=True 的数据集 (文本列占比高)
    """

    def __init__(self, job, dataset, hyperparams: dict = None):
        super().__init__(job, dataset, hyperparams)
        self.task_type = 'classification'
        self.batch_size = int(hyperparams.get('batch_size', BATCH_SIZE))
        self.learning_rate = float(hyperparams.get('learning_rate', LEARNING_RATE))
        self.max_length = int(hyperparams.get('max_length', MAX_LENGTH))
        self.total_epochs = int(hyperparams.get('epochs', EPOCHS))
        self.warmup_steps = int(hyperparams.get('warmup_steps', WARMUP_STEPS))
        self.model_name = hyperparams.get('model_name', DEFAULT_MODEL)
        self.test_size = float(hyperparams.get('test_size', 0.2))
        # v2: 验证集 + 早停
        self.val_size = float(hyperparams.get('val_size', 0.15))
        self.early_stopping_patience = int(hyperparams.get('early_stopping_patience', 10))

        self._model = None
        self._tokenizer = None
        self._train_loader = self._val_loader = self._test_loader = None
        self._optimizer = self._scheduler = None
        self._device = None
        self._id2label = {}
        self._label2id = {}
        self._text_column = None
        self._target_column = None
        self._X_test_texts = self._y_test_labels = None
        # v2: 早停追踪
        self._best_val_loss = float('inf')
        self._best_model_state = None
        self._patience_counter = 0
        self._stopped_early = False
        self._early_stop = False
        self._best_val_metric = None

    # ============ 数据加载 ============

    def load_data(self):
        import torch
        from sklearn.model_selection import train_test_split
        from transformers import AutoTokenizer

        _ensure_hf_access()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.callback.on_log(f'设备: {self._device}')
        self.callback.on_log(f'预训练模型: {self.model_name}')

        # 加载数据
        file_path = self.dataset.file_path
        fmt = self.dataset.file_format.lower()
        df = self._load_df(file_path, fmt)

        # 自动检测文本列和目标列
        text_col, target_col = self._detect_columns(df)
        self._text_column = text_col
        self._target_column = target_col

        # 清洗文本
        df[text_col] = df[text_col].astype(str).fillna('')
        texts = df[text_col].tolist()

        # 编码标签
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df[target_col].astype(str))
        self._label2id = {label: i for i, label in enumerate(le.classes_)}
        self._id2label = {i: label for label, i in self._label2id.items()}
        num_classes = len(le.classes_)

        self.callback.on_log(f'文本列: {text_col}, 目标列: {target_col}')
        self.callback.on_log(f'类别数: {num_classes}, 标签: {list(le.classes_)}')
        self.callback.on_log(f'样本数: {len(texts)}, 平均长度: {np.mean([len(t) for t in texts]):.0f} 字符')

        # —— 3-way 分割: train / val / test ——
        # Step 1: 分出 test 集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            texts, y, test_size=self.test_size, random_state=42,
            stratify=y if num_classes > 1 and min(np.bincount(y)) >= 2 else None
        )
        self._X_test_texts = X_test
        self._y_test_labels = y_test

        # Step 2: 从 train_val 中分出 val 集
        val_ratio = self.val_size / (1.0 - self.test_size)
        stratify_tv = y_train_val if num_classes > 1 and min(np.bincount(y_train_val)) >= 2 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=42,
            stratify=stratify_tv
        )

        self.callback.on_log(f'训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}')

        # Tokenize
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        train_enc = self._tokenizer(
            X_train, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        val_enc = self._tokenizer(
            X_val, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        test_enc = self._tokenizer(
            X_test, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )

        # DataLoader
        from torch.utils.data import TensorDataset, DataLoader
        train_ds = TensorDataset(
            train_enc['input_ids'], train_enc['attention_mask'],
            torch.tensor(y_train, dtype=torch.long)
        )
        val_ds = TensorDataset(
            val_enc['input_ids'], val_enc['attention_mask'],
            torch.tensor(y_val, dtype=torch.long)
        )
        test_ds = TensorDataset(
            test_enc['input_ids'], test_enc['attention_mask'],
            torch.tensor(y_test, dtype=torch.long)
        )
        self._train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self._val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        self._test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

    # ============ 模型构建 ============

    def build_model(self):
        import torch
        from transformers import AutoModelForSequenceClassification
        from transformers import get_linear_schedule_with_warmup

        # 确保模型可下载 (国内自动切镜像)
        _ensure_hf_access()

        num_classes = len(self._label2id)
        self.callback.on_log(f'正在加载预训练模型: {self.model_name} (num_labels={num_classes})')
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_classes,
            id2label=self._id2label,
            label2id=self._label2id,
        )
        self._model = self._model.to(self._device)

        total_params = sum(p.numel() for p in self._model.parameters())
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        self.callback.on_log(f'模型参数: {total_params:,} (可训练: {trainable:,})')

        # AdamW 优化器
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {'params': [p for n, p in self._model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self._model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        self._optimizer = torch.optim.AdamW(
            optimizer_grouped_params, lr=self.learning_rate
        )

        # 线性warmup+衰减调度器
        total_steps = len(self._train_loader) * self.total_epochs
        self._scheduler = get_linear_schedule_with_warmup(
            self._optimizer,
            num_warmup_steps=min(self.warmup_steps, total_steps // 5),
            num_training_steps=total_steps,
        )

    # ============ 训练 ============

    def train_epoch(self, epoch: int) -> dict:
        """训练一个 epoch + 验证，返回 train + val 指标"""
        import torch
        self._model.train()
        total_loss = 0.0
        correct = total = 0

        for batch in self._train_loader:
            input_ids, attention_mask, labels = [b.to(self._device) for b in batch]

            self._optimizer.zero_grad()
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimizer.step()
            self._scheduler.step()

            total_loss += loss.item() * len(input_ids)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        avg_loss = total_loss / total if total > 0 else 0
        acc = correct / total if total > 0 else 0
        lr = self._scheduler.get_last_lr()[0]

        # —— 验证集评估 ——
        self._model.eval()
        val_loss = 0.0
        val_correct = val_total = 0
        with torch.no_grad():
            for batch in self._val_loader:
                input_ids, attention_mask, labels = [b.to(self._device) for b in batch]
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item() * len(input_ids)
                preds = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_avg_loss = val_loss / val_total if val_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        # —— 早停 (基于 val_loss) ——
        if val_avg_loss < self._best_val_loss - 1e-4:
            self._best_val_loss = val_avg_loss
            self._patience_counter = 0
            self._best_model_state = copy.deepcopy(self._model.state_dict())
        else:
            self._patience_counter += 1
            if self._patience_counter >= self.early_stopping_patience:
                self._early_stop = True
                self._stopped_early = True
                self._best_val_metric = round(self._best_val_loss, 4)

        self.callback.on_log(
            f'Epoch {epoch+1}/{self.total_epochs} - loss={avg_loss:.4f}, '
            f'acc={acc:.4f}, val_loss={val_avg_loss:.4f}, val_acc={val_acc:.4f}, '
            f'lr={lr:.2e}'
        )
        return {
            'train_loss': round(avg_loss, 4),
            'train_accuracy': round(acc, 4),
            'val_loss': round(val_avg_loss, 4),
            'val_accuracy': round(val_acc, 4),
        }

    # ============ 评估 ============

    def evaluate(self) -> dict:
        """最终评估: 恢复最佳模型 → 在测试集上计算最终指标"""
        import torch
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # —— 恢复早停最佳模型 ——
        if self._best_model_state is not None:
            self._model.load_state_dict(self._best_model_state)
            self.callback.on_log(f'已恢复最佳模型 (val_loss={self._best_val_loss:.4f})')

        self._model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self._test_loader:
                input_ids, attention_mask, labels = [b.to(self._device) for b in batch]
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        result = {
            'test_accuracy': round(float(accuracy_score(all_labels, all_preds)), 4),
        }
        try:
            result['test_precision_weighted'] = round(float(precision_score(all_labels, all_preds, average='weighted', zero_division=0)), 4)
            result['test_recall_weighted'] = round(float(recall_score(all_labels, all_preds, average='weighted', zero_division=0)), 4)
            result['test_f1_weighted'] = round(float(f1_score(all_labels, all_preds, average='weighted', zero_division=0)), 4)
            result['test_precision_macro'] = round(float(precision_score(all_labels, all_preds, average='macro', zero_division=0)), 4)
            result['test_recall_macro'] = round(float(recall_score(all_labels, all_preds, average='macro', zero_division=0)), 4)
            result['test_f1_macro'] = round(float(f1_score(all_labels, all_preds, average='macro', zero_division=0)), 4)
        except Exception:
            pass

        if self._stopped_early:
            result['early_stopped'] = True
            result['best_val_loss'] = round(self._best_val_loss, 4)
        return result

    # ============ 保存 ============

    def save_model(self, path: str):
        # —— 恢复最佳模型后再保存 ——
        if self._best_model_state is not None:
            self._model.load_state_dict(self._best_model_state)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dir = path + '_nlp_model'
        self._model.save_pretrained(save_dir)
        self._tokenizer.save_pretrained(save_dir)

        # 保存元数据
        meta = {
            'model_name': self.model_name,
            'text_column': self._text_column,
            'target_column': self._target_column,
            'id2label': self._id2label,
            'label2id': self._label2id,
            'max_length': self.max_length,
            'task_type': 'classification',
        }
        with open(os.path.join(save_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self.callback.on_log(f'模型已保存到: {save_dir}/')

    # ============ 检查点 ============

    def save_checkpoint(self):
        """保存 Transformers 训练快照: 模型权重 + 优化器 + 调度器 + epoch + 早停状态"""
        if self._model is None:
            return
        import torch
        os.makedirs(self.output_dir, exist_ok=True)

        ckpt = {
            'model_state': self._model.state_dict(),
            'optimizer_state': self._optimizer.state_dict(),
            'scheduler_state': self._scheduler.state_dict(),
            'epoch': self._current_epoch + 1,
            'best_val_loss': self._best_val_loss,
            'patience_counter': self._patience_counter,
            'best_model_state': self._best_model_state,
        }
        ckpt_path = os.path.join(self.output_dir, 'checkpoint.pt')
        torch.save(ckpt, ckpt_path)

    @staticmethod
    def load_checkpoint(output_dir: str) -> dict:
        import torch
        ckpt_path = os.path.join(output_dir, 'checkpoint.pt')
        if not os.path.exists(ckpt_path):
            return {}
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        return {
            'epoch': ckpt.get('epoch', 0),
            'best_val_loss': ckpt.get('best_val_loss', float('inf')),
            'patience_counter': ckpt.get('patience_counter', 0),
            '_restore': ckpt,
        }

    def restore_checkpoint(self, ckpt: dict):
        """恢复 Transformers 模型权重 + 优化器 + 调度器 + 早停状态"""
        restore_data = ckpt.get('_restore')
        if not restore_data:
            return

        if self._model is not None and 'model_state' in restore_data:
            self._model.load_state_dict(restore_data['model_state'])

        if self._optimizer is not None and 'optimizer_state' in restore_data:
            self._optimizer.load_state_dict(restore_data['optimizer_state'])

        if self._scheduler is not None and 'scheduler_state' in restore_data:
            self._scheduler.load_state_dict(restore_data['scheduler_state'])

        self._best_val_loss = restore_data.get('best_val_loss', float('inf'))
        self._patience_counter = restore_data.get('patience_counter', 0)
        if 'best_model_state' in restore_data and restore_data['best_model_state'] is not None:
            self._best_model_state = restore_data['best_model_state']

    @staticmethod
    def has_checkpoint(output_dir: str) -> bool:
        return os.path.exists(os.path.join(output_dir, 'checkpoint.pt'))

    # ============ 工具方法 ============

    @staticmethod
    def _load_df(file_path: str, fmt: str) -> pd.DataFrame:
        if fmt == 'csv':
            return pd.read_csv(file_path)
        elif fmt in ('xlsx', 'xls'):
            return pd.read_excel(file_path)
        elif fmt == 'json':
            return pd.read_json(file_path)
        elif fmt == 'parquet':
            return pd.read_parquet(file_path)
        elif fmt == 'txt':
            return pd.read_csv(file_path, sep='\t')
        raise ValueError(f'不支持的文件格式: {fmt}')

    def _detect_columns(self, df: pd.DataFrame):
        """自动检测文本列和目标列"""
        hyper_target = self.hyperparams.get('target_column')
        hyper_text = self.hyperparams.get('text_column')

        target_col = hyper_target
        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        text_col = hyper_text
        if not text_col or text_col not in df.columns:
            # 自动检测: 找最长的字符串列
            str_cols = df.select_dtypes(include=['object']).columns
            if len(str_cols) == 0:
                col_info = ', '.join([f'{c}({d})' for c, d in df.dtypes.items()][:10])
                raise ValueError(
                    f'未检测到文本(object)列 — 该数据集所有列为数值类型。\n'
                    f'前10列: {col_info}\n'
                    f'BETR tokenizer 需要原始文本数据。如果数据是预提取的特征(如TF-IDF)，'
                    f'请将 algorithm 设为 "mlp" 使用 PyTorch/sklearn 训练器。'
                )
            # 选平均长度最长的object列
            text_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
            # 确保和目标列不同
            if text_col == target_col:
                others = [c for c in str_cols if c != target_col]
                if others:
                    text_col = max(others, key=lambda c: df[c].astype(str).str.len().mean())

        return text_col, target_col
