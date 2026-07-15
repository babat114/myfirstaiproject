"""
scikit-learn 训练器
支持分类、回归和聚类任务，涵盖常用算法
"""

import contextlib
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target

from app.executor.trainers.base import BaseTrainer

_nlp_logger = logging.getLogger(__name__)


# ===================================================================
# sklearn 算法注册表
# 格式: '算法简称' → ('模块路径', '类名')
# 分类器 — 预测离散类别标签
# ===================================================================
_CLASSIFIERS = {
    'random_forest': ('sklearn.ensemble', 'RandomForestClassifier'),  # 随机森林分类器 - 集成多棵决策树投票
    'logistic_regression': ('sklearn.linear_model', 'LogisticRegression'),  # 逻辑回归 - 线性分类器, 输出类别概率
    'svm': ('sklearn.svm', 'SVC'),  # 支持向量机 - 寻找最大间隔超平面
    'knn': ('sklearn.neighbors', 'KNeighborsClassifier'),  # K近邻 - 基于距离度量的懒惰学习
    'gradient_boosting': ('sklearn.ensemble', 'GradientBoostingClassifier'),  # 梯度提升树 - 逐步拟合残差的集成方法
    'decision_tree': ('sklearn.tree', 'DecisionTreeClassifier'),  # 决策树 - 树形规则分裂, 可解释性强
}

# 回归器 — 预测连续数值
_REGRESSORS = {
    'linear_regression': ('sklearn.linear_model', 'LinearRegression'),  # 线性回归 - 最小二乘法拟合
    'ridge': ('sklearn.linear_model', 'Ridge'),  # 岭回归 - L2正则化线性回归
    'random_forest_regressor': ('sklearn.ensemble', 'RandomForestRegressor'),  # 随机森林回归器
    'svr': ('sklearn.svm', 'SVR'),  # 支持向量回归 - epsilon不敏感损失
    'gradient_boosting_regressor': ('sklearn.ensemble', 'GradientBoostingRegressor'),  # 梯度提升回归器
    'knn_regressor': ('sklearn.neighbors', 'KNeighborsRegressor'),  # K近邻回归器 - 基于距离的回归
}

# 聚类器 — 无监督学习, 发现数据内在分组结构
_CLUSTERERS = {
    'kmeans': ('sklearn.cluster', 'KMeans'),  # K-Means — 基于质心的划分聚类
    'dbscan': ('sklearn.cluster', 'DBSCAN'),  # DBSCAN — 基于密度的空间聚类
    'agglomerative': ('sklearn.cluster', 'AgglomerativeClustering'),  # 层次聚类 — 自底向上合并
    'minibatch_kmeans': ('sklearn.cluster', 'MiniBatchKMeans'),  # MiniBatch K-Means — 小批量增量聚类
}


# 分类器 → 回归器算法映射 (当目标列为连续值时自动切换)
_CLS_TO_REG = {
    'random_forest': 'random_forest_regressor',
    'gradient_boosting': 'gradient_boosting_regressor',
    'knn': 'knn_regressor',
    'svm': 'svr',
    'logistic_regression': 'ridge',
    'decision_tree': 'random_forest_regressor',
}


def _import_model(module_path: str, class_name: str):
    """动态导入 sklearn 模型类"""
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class SklearnTrainer(BaseTrainer):
    """scikit-learn 训练器 — 支持分类、回归和聚类任务

    支持的算法:
        分类: 随机森林, 逻辑回归, SVM, KNN, 梯度提升, 决策树
        回归: 线性回归, 岭回归, 随机森林回归, SVR, 梯度提升回归, KNN回归
        聚类: K-Means, DBSCAN, 层次聚类, MiniBatch K-Means

    训练策略:
        - GradientBoosting: warm_start 增量添加树 → 真实渐进训练曲线
        - SGD/增量模型: partial_fit 分批训练
        - 聚类: 单轮无监督fit, 轮廓系数/DB指数/CH指数评估
        - 其他模型: 全量fit + 渐进子集评估 → 模拟收敛曲线
    """

    # 闭环模型 (无法增量训练) — 只应做 1 轮 fit，多轮 epoch 毫无意义
    _CLOSED_FORM_ALGOS = {
        'linear_regression',
        'ridge',
        'logistic_regression',
        'svm',
        'svr',
        'knn',
        'knn_regressor',
        'kmeans',
        'dbscan',
        'agglomerative',
        'minibatch_kmeans',
        'decision_tree',  # 决策树一次fit完全生长，多轮重复无意义
    }

    # 需要正则化防过拟合的算法默认参数
    # class_weight='balanced' 自动补偿类别不平衡 (对不支持该参数的算法跳过)
    _REGULARIZE_DEFAULTS = {
        'decision_tree': {'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 5, 'class_weight': 'balanced'},
        'random_forest': {'max_depth': 15, 'min_samples_leaf': 5, 'class_weight': 'balanced'},
        'random_forest_regressor': {'max_depth': 15, 'min_samples_leaf': 5},
        'gradient_boosting': {'max_depth': 5, 'min_samples_leaf': 10},
        'gradient_boosting_regressor': {'max_depth': 5, 'min_samples_leaf': 10},
        'svm': {'C': 0.5, 'class_weight': 'balanced'},
        'svr': {'C': 0.5},
        'knn': {'n_neighbors': 7},
        'logistic_regression': {'max_iter': 2000, 'class_weight': 'balanced'},
    }

    def __init__(self, job, dataset, hyperparams: dict = None):
        super().__init__(job, dataset, hyperparams)

        self.task_type = self.hyperparams.get('task_type', 'classification')
        self.algorithm = self.hyperparams.get('algorithm', 'random_forest')
        self.test_size = float(self.hyperparams.get('test_size', 0.2))
        self.random_state = int(self.hyperparams.get('random_state', 42))

        # 闭环模型强制单轮训练 — KNN/SVM/逻辑回归等一次 fit 即收敛, 不存在"多轮epoch"
        if self.algorithm in self._CLOSED_FORM_ALGOS:
            user_epochs = self.total_epochs
            if user_epochs > 1:
                self.callback.on_log(
                    f'[提示] {self.algorithm} 是闭环模型(一次训练即收敛)，'
                    f'已忽略 total_epochs={user_epochs}，固定为 1 轮。'
                    f'如需多轮训练请选择 PyTorch MLP 或其他迭代算法。'
                )
            self.total_epochs = 1

        self._model = None
        self._X_train = self._X_test = None
        self._y_train = self._y_test = None
        self._scaler = None
        self._label_encoders = {}
        self._vectorizer = None  # NLP: TfidfVectorizer
        self._class_labels = []  # 类别标签 (人类可读)

    # ═══════════════════════════════════════════════════════════════
    # 数据加载 (v2: 上帝方法拆分为 1 个主方法 + 8 个私有方法)
    # ═══════════════════════════════════════════════════════════════

    def load_data(self):
        """从 Dataset 文件加载数据并预处理。

        原始 397 行上帝方法已拆分为以下步骤 (M1 refactor):
        1. _read_data_file → 2. _prepare_clustering_data (无监督)
        → 3. _detect_nlp_features → 4. _impute_encode_features
        → 5. _encode_target_variable → 6. _split_train_test
        → 7. _apply_tfidf_to_splits → 8. _apply_scaler_and_balance
        """
        df = self._read_data_file()

        # ── 全行去重: 防止相同数据行落入训练/测试两侧导致数据泄露 ──
        before = len(df)
        df = df.drop_duplicates()
        if len(df) < before:
            self.callback.on_log(f'数据去重: {before} → {len(df)} 行 (移除 {before - len(df)} 条完全重复行)')

        # ── 聚类: 无监督学习, 不需要目标列 ──
        if self.task_type == 'clustering':
            self._prepare_clustering_data(df)
            return

        # ── 分类 / 回归: 监督学习, 需要目标列 ──
        target_col = self.hyperparams.get('target_column')
        if not target_col:
            target_col = df.columns[-1]

        if target_col not in df.columns:
            raise ValueError(f'目标列 "{target_col}" 不存在。可用列: {list(df.columns)}')

        self.callback.on_log(f'目标列: {target_col}, 特征数: {len(df.columns) - 1}')
        self.callback.on_log(f'数据形状: {df.shape}, 测试比例: {self.test_size}')

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # ── NLP 文本检测 + TF-IDF 预处理 ──
        nlp_texts, nlp_vectorizer_config, X = self._detect_nlp_features(X)

        # ── 保存类别标签 ──
        if self.task_type == 'classification':
            self._save_class_labels(y)

        # ── 先划分训练/测试集 (在 imputation/encoding 之前, 防止数据泄漏) ──
        self._split_train_test(X, y)

        # ── 仅在训练集上拟合 imputer/encoder, 变换训练集和测试集 ──
        self._fit_impute_encode_splits()

        # ── 目标变量编码: 仅在训练集上拟合 LabelEncoder ──
        self._fit_encode_target()

        # ── NLP: TF-IDF 仅在训练集上拟合, 然后变换训练集和测试集 ──
        if nlp_texts is not None and nlp_vectorizer_config is not None:
            self._apply_tfidf_to_splits(nlp_texts, nlp_vectorizer_config)

        # ── StandardScaler + 类别平衡 ──
        self._apply_scaler_and_balance()

        self.callback.on_log(f'训练集: {len(self._X_train)} 样本, 测试集: {len(self._X_test)} 样本')

    # ── 数据加载步骤 1: 读取文件 ──

    def _read_data_file(self) -> pd.DataFrame:
        """根据文件格式读取数据文件。"""
        file_path = self.dataset.file_path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'数据集文件不存在: {file_path}')

        fmt = self.dataset.file_format.lower()

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
        else:
            raise ValueError(f'不支持的文件格式: {fmt}')

    # ── 数据加载步骤 2: 聚类数据预处理 ──

    def _prepare_clustering_data(self, df: pd.DataFrame):
        """无监督聚类数据预处理 (无目标列, 不参与训练)。"""
        target_col = self.hyperparams.get('target_column')
        if target_col and target_col in df.columns:
            self._y_full = df[target_col].copy()
            X = df.drop(columns=[target_col])
            self.callback.on_log(f'目标列 (仅用于外部验证): {target_col}, 特征数: {len(X.columns)}')
        else:
            self._y_full = None
            X = df.copy()
            self.callback.on_log(f'无监督聚类, 特征数: {len(X.columns)}')

        self.callback.on_log(f'数据形状: {df.shape}, 测试比例: {self.test_size}')

        # 预处理: 缺失值填充
        num_cols = X.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy='mean')
            X[num_cols] = num_imputer.fit_transform(X[num_cols])

        cat_cols = X.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self._label_encoders[col] = le

        # 标准化
        num_cols_after = X.select_dtypes(include=[np.number]).columns
        if len(num_cols_after) > 0:
            self._scaler = StandardScaler()
            X[num_cols_after] = self._scaler.fit_transform(X[num_cols_after])

        # 划分训练/测试集 (无 y, 无分层)
        self._X_train, self._X_test = train_test_split(X, test_size=self.test_size, random_state=self.random_state)
        # 如果保留了 y, 对齐划分用于 ARI/NMI
        if self._y_full is not None:
            _, self._y_test = train_test_split(self._y_full, test_size=self.test_size, random_state=self.random_state)
            self._y_train = None
        else:
            self._y_train = self._y_test = None

        self.callback.on_log(f'训练集: {len(self._X_train)} 样本, 测试集: {len(self._X_test)} 样本')

    # ── 数据加载步骤 3: NLP 文本检测 ──

    def _detect_nlp_features(self, X: pd.DataFrame) -> tuple:
        """检测 NLP 文本列并准备 TF-IDF vectorizer 配置。

        Returns:
            (nlp_texts, vectorizer_config, X_clean) — 如果没有文本列, 前两个为 None
        """
        nlp_text_col = None
        ds_category = getattr(self.dataset, 'category', None)
        _nlp_logger.info(
            '[DEBUG] NLP check: dataset.category=%r, dataset.id=%s, X.columns=%s, task_type=%s',
            ds_category,
            getattr(self.dataset, 'id', '?'),
            list(X.columns)[:8],
            self.task_type,
        )

        if ds_category == 'nlp':
            for candidate in ['text', 'review', 'comment', 'content', 'sentence']:
                if candidate in X.columns:
                    nlp_text_col = candidate
                    _nlp_logger.info('[DEBUG] NLP text column found: %s', candidate)
                    break
            # 如果前几列是 tfidf_* 说明数据已被预处理, 不再做 NLP 处理
            if nlp_text_col is None and X.columns[0].startswith('tfidf_'):
                _nlp_logger.info('[DEBUG] NLP: features already TF-IDF, skipping')
            elif nlp_text_col is None:
                _nlp_logger.info('[DEBUG] NLP: no text column found in %s', list(X.columns))

        if nlp_text_col is None:
            return None, None, X

        # ── 读取 NLP 参数 (Batch B 优化) ──
        _nlp_mf = int(self.hyperparams.get('nlp_max_features', 2000))
        _nlp_min_df = int(self.hyperparams.get('nlp_min_df', 2))
        _nlp_max_df = float(self.hyperparams.get('nlp_max_df', 0.9))

        _nlp_logger.info(
            '[NLP] Detected text col "%s", will apply TfidfVectorizer '
            'AFTER split (combined jieba+char tokenizer, max_features=%d)',
            nlp_text_col,
            _nlp_mf,
        )
        self.callback.on_log(
            f'[NLP] 检测到文本列 "{nlp_text_col}", TfidfVectorizer '
            f'(jieba+char 合并分词, max_features={_nlp_mf}, min_df={_nlp_min_df}, '
            f'max_df={_nlp_max_df}, 仅在训练集上拟合)'
        )

        # 保存原始文本 (在 drop 之前)
        nlp_texts = X[nlp_text_col].fillna('').astype(str).tolist()

        # ── 自适应 max_features: 不超过训练样本数的一半 (防小数据集过拟合) ──
        _n_train_est = int(len(nlp_texts) * (1 - self.test_size))
        _adaptive_mf = max(100, min(_nlp_mf, _n_train_est // 2))
        if _adaptive_mf != _nlp_mf:
            self.callback.on_log(
                f'[NLP] 自适应 max_features: {_nlp_mf} → {_adaptive_mf} (训练集约 {_n_train_est} 样本)'
            )
            _nlp_mf = _adaptive_mf

        # ── 配置 vectorizer (使用共享 NLP 预处理模块) ──
        from app.utils.nlp_preprocessing import create_vectorizer_config

        nlp_vectorizer_config = create_vectorizer_config(_nlp_mf, _nlp_min_df, _nlp_max_df)

        # 从特征矩阵中移除文本列 (稍后添加 TF-IDF 特征)
        X = X.drop(columns=[nlp_text_col])

        return nlp_texts, nlp_vectorizer_config, X

    # ── 数据加载步骤 4: 缺失值填充 + 分类特征编码 ──

    def _save_class_labels(self, y: pd.Series):
        """保存类别标签 (人类可读)。"""
        try:
            unique_labels = y.dropna().unique()
            self._class_labels = [str(c) for c in sorted(unique_labels, key=str)]
            self.callback.on_log(f'[类别标签] {self._class_labels}')
        except Exception:
            self._class_labels = []

    def _fit_impute_encode_splits(self):
        """在划分后的训练集上拟合 imputer/encoder, 变换训练集和测试集。

        防止数据泄漏: imputer/encoder 仅在训练集上拟合。
        """
        X_train = self._X_train
        X_test = self._X_test

        # 处理缺失值 (非文本列) — 仅在训练集上拟合
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy='mean')
            X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
            test_num = [c for c in num_cols if c in X_test.columns]
            if test_num:
                X_test[test_num] = num_imputer.transform(X_test[test_num])

        cat_cols = X_train.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
            test_cat = [c for c in cat_cols if c in X_test.columns]
            if test_cat:
                X_test[test_cat] = cat_imputer.transform(X_test[test_cat])

        # 编码分类特征 — 仅在训练集上拟合 LabelEncoder
        for col in cat_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            self._label_encoders[col] = le
            if col in X_test.columns:
                try:
                    X_test[col] = le.transform(X_test[col].astype(str))
                except ValueError:
                    # 测试集中出现训练集未见的类别 — 映射到已知类
                    X_test[col] = (
                        X_test[col].astype(str).apply(lambda x, le=le: x if x in le.classes_ else le.classes_[0])
                    )
                    X_test[col] = le.transform(X_test[col].astype(str))

        self._X_train = X_train
        self._X_test = X_test

    # ── 数据加载步骤 5: 目标变量编码 (拆分后, 仅在训练集上拟合) ──

    def _fit_encode_target(self):
        """在划分后的训练集上拟合目标编码器, 变换训练集和测试集。

        含分类→回归自动纠错逻辑 (type_of_target 检查是纯检查, 无泄漏)。
        """
        if self.task_type != 'classification':
            return

        y_train = self._y_train
        y_test = self._y_test

        try:
            yt = type_of_target(y_train)
        except Exception:
            yt = 'unknown'

        if yt == 'continuous':
            reg_algo = _CLS_TO_REG.get(self.algorithm)
            if reg_algo:
                self.callback.on_log('[自动纠错] 目标列是连续值(float), 但任务类型是 classification。')
                self.callback.on_log(f'[自动纠错] 已自动切换: task_type → regression, algorithm → {reg_algo}')
                self.task_type = 'regression'
                self.algorithm = reg_algo
                if self.algorithm in self._CLOSED_FORM_ALGOS and self.total_epochs > 1:
                    self.total_epochs = 1
                    self.callback.on_log(f'[自动纠错] {reg_algo} 是闭环模型，epochs 固定为 1')
            else:
                self.callback.on_log(
                    f'[警告] 目标列是连续值，'
                    f'但算法 "{self.algorithm}" 无对应回归器。'
                    f'将强制编码为分类标签 (可能导致无意义结果)。'
                )
                le = LabelEncoder()
                self._y_train = le.fit_transform(y_train.astype(str))
                if y_test is not None:
                    try:
                        self._y_test = le.transform(y_test.astype(str))
                    except ValueError:
                        self._y_test = le.transform(
                            y_test.astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        )
                self._label_encoders['__target__'] = le
        elif y_train.dtype == 'object':
            le = LabelEncoder()
            self._y_train = le.fit_transform(y_train.astype(str))
            if y_test is not None:
                try:
                    self._y_test = le.transform(y_test.astype(str))
                except ValueError:
                    self._y_test = le.transform(
                        y_test.astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    )
            self._label_encoders['__target__'] = le

    # ── 数据加载步骤 6: 训练/测试集划分 ──

    def _split_train_test(self, X: pd.DataFrame, y: pd.Series):
        """划分训练/测试集, 带分类分层采样。

        在 TF-IDF 和 StandardScaler 之前调用, 防止数据泄漏。
        """
        stratify_y = None
        if self.task_type == 'classification':
            try:
                from collections import Counter

                class_counts = Counter(y)
                min_count = min(class_counts.values())
                if min_count >= 2:
                    stratify_y = y
                else:
                    self.callback.on_log(f'警告: 最少类别样本数={min_count}, 无法分层采样, 使用随机划分')
            except Exception:
                pass

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify_y
        )

    # ── 数据加载步骤 7: TF-IDF 变换 (仅在训练集上拟合) ──

    def _apply_tfidf_to_splits(self, nlp_texts: list, nlp_vectorizer_config: dict):
        """在训练集上拟合 TfidfVectorizer, 然后变换训练集和测试集。

        包含数据增强 (augment_factor > 1 时) 和自适应 max_features。
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        # 对齐文本索引到 train/test split
        _train_indices = self._X_train.index.tolist()
        _test_indices = self._X_test.index.tolist()
        _train_texts = [nlp_texts[i] for i in _train_indices]
        _test_texts = [nlp_texts[i] for i in _test_indices]

        # ── 数据增强 (仅增强训练集, 不碰测试集) ──
        _augment_factor = int(self.hyperparams.get('augment_factor', 0))
        if _augment_factor > 1 and len(_train_texts) < 5000:
            try:
                from app.utils.text_augment import augment_texts

                _n_train_orig = len(_train_texts)
                _y_train_list = list(self._y_train)
                _train_texts, _y_train_aug = augment_texts(
                    _train_texts, _y_train_list, factor=_augment_factor, seed=self.random_state
                )
                _n_aug = len(_train_texts) - _n_train_orig
                if _n_aug > 0:
                    # 扩展 X_train (空行, 因为文本列即将被 TF-IDF 替换)
                    _aug_indices = list(range(self._X_train.index.max() + 1, self._X_train.index.max() + 1 + _n_aug))
                    _aug_X = pd.DataFrame(
                        np.zeros((_n_aug, self._X_train.shape[1])), index=_aug_indices, columns=self._X_train.columns
                    )
                    self._X_train = pd.concat([self._X_train, _aug_X], axis=0)
                    self._y_train = np.concatenate([self._y_train, _y_train_aug[_n_train_orig:]])
                    # 更新 _train_indices 以包含新增的增强行索引
                    _train_indices = self._X_train.index.tolist()
                self.callback.on_log(
                    f'[增强] 文本增强完成: {_n_train_orig} -> {len(_train_texts)} '
                    f'(factor={_augment_factor}, +{_n_aug} samples)'
                )
            except ImportError:
                self.callback.on_log('[增强] text_augment 模块未找到, 跳过增强')
            except Exception as e:
                self.callback.on_log(f'[增强] 文本增强失败: {e}')

        self._vectorizer = TfidfVectorizer(**nlp_vectorizer_config)
        _tfidf_train = self._vectorizer.fit_transform(_train_texts)
        _tfidf_test = self._vectorizer.transform(_test_texts)

        _tfidf_cols = [f'tfidf_{i}' for i in range(_tfidf_train.shape[1])]
        _tfidf_train_df = pd.DataFrame.sparse.from_spmatrix(
            _tfidf_train,
            columns=_tfidf_cols,
            index=_train_indices,
        )
        _tfidf_test_df = pd.DataFrame.sparse.from_spmatrix(
            _tfidf_test,
            columns=_tfidf_cols,
            index=_test_indices,
        )

        self._X_train = pd.concat([self._X_train, _tfidf_train_df], axis=1)
        self._X_test = pd.concat([self._X_test, _tfidf_test_df], axis=1)

        self.callback.on_log(
            f'[NLP] TF-IDF 完成: {len(_train_texts)}+{len(_test_texts)} 文本 '
            f'→ {_tfidf_train.shape[1]} 维特征 (仅在训练集上拟合)'
        )
        _nlp_logger.info(
            '[NLP] TF-IDF done: %d+%d texts -> %d features (fit on train only)',
            len(_train_texts),
            len(_test_texts),
            _tfidf_train.shape[1],
        )

    # ── 数据加载步骤 8: StandardScaler + 类别平衡 ──

    def _apply_scaler_and_balance(self):
        """StandardScaler (仅在训练集上拟合) + 类别平衡 (SMOTE/undersample)。

        IMPORTANT: TF-IDF 特征列 (tfidf_*) 不参与 StandardScaler。
        TF-IDF 已被 TfidfVectorizer L2 归一化, 叠加 StandardScaler 会:
         - 破坏稀疏性 (0 -> 负值, 模型无法区分"词不存在"和"词频低")
         - 洗掉判别信号 (负面词和正面词区分度归零)
         - 导致模型对全部输入预测同一类别 (正向偏置)
        """
        # ═══════════════════════════════════════════════════════════════
        # StandardScaler: 仅在训练集上拟合, 跳过 tfidf_* 列
        # ═══════════════════════════════════════════════════════════════
        all_num_cols = self._X_train.select_dtypes(include=[np.number]).columns
        tfidf_cols = [c for c in all_num_cols if str(c).startswith('tfidf_')]
        non_tfidf_cols = [c for c in all_num_cols if c not in tfidf_cols]

        if tfidf_cols:
            n_tfidf = len(tfidf_cols)
            _nlp_logger.info(
                '[NLP] Skipping StandardScaler for %d tfidf_* columns (already L2-normalized by TfidfVectorizer)',
                n_tfidf,
            )
            self.callback.on_log(
                f'[NLP] 已跳过 StandardScaler for {n_tfidf} TF-IDF 特征列 (TfidfVectorizer 已做 L2 归一化)'
            )

        if len(non_tfidf_cols) > 0:
            self._scaler = StandardScaler()
            self._X_train[non_tfidf_cols] = self._scaler.fit_transform(self._X_train[non_tfidf_cols])
            # 测试集用训练集的 scaler 变换
            num_cols_test = [c for c in non_tfidf_cols if c in self._X_test.columns]
            if num_cols_test:
                self._X_test[num_cols_test] = self._scaler.transform(self._X_test[num_cols_test])
        else:
            # 无其他数值列 (纯 NLP 数据集), scaler 保持 None
            _nlp_logger.info('[NLP] No non-tfidf numeric columns, scaler is None')
            self._scaler = None

        # ── NLP 类别不平衡 → 重采样 (SMOTE / undersample) ──
        _balance = self.hyperparams.get('balance', None)
        if _balance and _balance != 'none' and self.task_type == 'classification':
            try:
                from collections import Counter

                from imblearn.over_sampling import SMOTE
                from imblearn.under_sampling import RandomUnderSampler

                before = Counter(self._y_train)

                # 保存原始列名 (SMOTE/undersample 返回 numpy array 会丢失列名)
                _orig_cols = list(self._X_train.columns)

                if _balance == 'smote':
                    sampler = SMOTE(random_state=self.random_state or 42)
                    self._X_train, self._y_train = sampler.fit_resample(self._X_train, self._y_train)
                elif _balance == 'undersample':
                    sampler = RandomUnderSampler(random_state=self.random_state or 42)
                    self._X_train, self._y_train = sampler.fit_resample(self._X_train, self._y_train)

                after = Counter(self._y_train)
                # 恢复 DataFrame + 原始列名 (SMOTE/undersample 返回 numpy array)
                self._X_train = pd.DataFrame(self._X_train, columns=_orig_cols)
                self._y_train = pd.Series(self._y_train)
                self.callback.on_log(f'[平衡] {_balance}: {dict(before)} → {dict(after)}')
            except ImportError:
                self.callback.on_log('[平衡] imbalanced-learn 未安装, 跳过重采样')
            except Exception as e:
                self.callback.on_log(f'[平衡] 重采样失败: {e}')

    # ═══════════════════════════════════════════════════════════════
    # 模型构建 + 训练 + 评估
    # ═══════════════════════════════════════════════════════════════

    def build_model(self):
        """构建 sklearn 模型"""
        if self.task_type == 'classification':
            model_info = _CLASSIFIERS.get(self.algorithm)
        elif self.task_type == 'clustering':
            model_info = _CLUSTERERS.get(self.algorithm)
        else:
            model_info = _REGRESSORS.get(self.algorithm)

        if not model_info:
            raise ValueError(f'不支持的算法: {self.algorithm} (任务类型: {self.task_type})')

        module_path, class_name = model_info
        model_cls = _import_model(module_path, class_name)

        # 获取算法特定参数, 过滤掉 None 值 (None 会覆盖 sklearn 默认值导致崩溃)
        algo_params = {k: v for k, v in self.hyperparams.get('algorithm_params', {}).items() if v is not None}

        # 应用正则化默认值 (防过拟合) — 用户显式指定则不覆盖
        if self.algorithm in self._REGULARIZE_DEFAULTS:
            for k, v in self._REGULARIZE_DEFAULTS[self.algorithm].items():
                if k not in algo_params:
                    algo_params[k] = v
            self.callback.on_log(
                f'[正则化] 已应用 {self.algorithm} 默认约束: '
                + ', '.join(f'{k}={v}' for k, v in self._REGULARIZE_DEFAULTS[self.algorithm].items())
            )

        # warm_start: epochs > 1 时启用增量训练 (否则每轮 fit() 从零重建)
        if (
            self.total_epochs > 1
            and module_path
            not in (
                'sklearn.svm',
                'sklearn.linear_model',
                'sklearn.neighbors',
                'sklearn.cluster',
            )
            and 'warm_start' not in algo_params
        ):
            algo_params['warm_start'] = True

        try:
            self._model = model_cls(random_state=self.random_state, **algo_params)
        except TypeError:
            # KNN, LinearRegression, SVR 等模型不接受 random_state
            self._model = model_cls(**algo_params)
        self.callback.on_log(f'模型: {class_name}')

    def train_epoch(self, epoch: int) -> dict:
        """sklearn epoch 训练 — 支持增量式训练以产生可视化训练曲线

        第一轮完整拟合后，后续 epoch 通过渐进评估和增量训练来模拟渐进提升:
        - 聚类: 单轮无监督fit (DBSCAN用fit_predict), 轮廓系数等评估
        - GradientBoosting/SGD: 使用 warm_start / partial_fit 进行真实增量训练
        - 其他模型: 在逐渐扩大的训练子集上评估, 模拟收敛曲线
        """
        # ---- 策略0: 聚类训练 (无监督, 单轮fit) ----
        if self.task_type == 'clustering':
            if epoch == 0:
                if hasattr(self._model, 'fit_predict'):
                    self._labels_train = self._model.fit_predict(self._X_train)
                else:
                    self._model.fit(self._X_train)
                    self._labels_train = self._model.labels_
            return self._compute_metrics(X=self._X_train, labels=self._labels_train, prefix='train_')

        # 计算渐进比例 (epoch 0 = 全量训练)
        progress = (epoch + 1) / max(self.total_epochs, 1)

        # ---- 策略1: warm_start 增量训练 (RF/GB 等, 需 warm_start=True) ----
        if getattr(self._model, 'warm_start', False) and hasattr(self._model, 'n_estimators'):
            # 保存原始 n_estimators 避免被修改后的值影响后续epoch计算
            if not hasattr(self, '_original_n_estimators'):
                self._original_n_estimators = self._model.n_estimators
            original_total = self._original_n_estimators
            trees_per_epoch = max(1, int(original_total / self.total_epochs))

            if epoch == 0:
                # 首次: 设置初始 n_estimators 然后 fit
                self._model.n_estimators = trees_per_epoch
                self._model.fit(self._X_train, self._y_train)
            else:
                # 增量添加树 — 使用原始 total，不是被修改后的值
                new_trees = min(original_total, self._model.n_estimators + trees_per_epoch)
                self._model.n_estimators = new_trees
                self._model.fit(self._X_train, self._y_train)

            # 在全量训练集上评估 (兼容 DataFrame 和 numpy array)
            y_pred = self._model.predict(self._safe_index(self._X_train))
            y_true = self._safe_index(self._y_train)
            return self._compute_metrics(y_true, y_pred, prefix='train_')

        # ---- 策略2: partial_fit 增量训练 (SGDClassifier 等) ----
        if hasattr(self._model, 'partial_fit'):
            if epoch == 0 and not hasattr(self._model, 'classes_'):
                self._model.partial_fit(self._X_train, self._y_train, classes=np.unique(self._y_train))
            else:
                # 每次用不同子集进行partial_fit
                frac = 0.3
                n_total = len(self._X_train)
                n_samples = max(100, int(n_total * frac))
                indices = np.random.RandomState(epoch).randint(0, n_total, n_samples)
                self._model.partial_fit(
                    self._safe_index(self._X_train, indices), self._safe_index(self._y_train, indices)
                )

            n_total = len(self._X_train)
            sample_size = min(n_total, int(n_total * progress))
            y_pred = self._model.predict(self._safe_index(self._X_train, slice(None, sample_size)))
            y_true = self._safe_index(self._y_train, slice(None, sample_size))
            return self._compute_metrics(y_true, y_pred, prefix='train_')

        # ---- 策略3: 标准拟合 (闭环模型一次 fit，全量训练集评估) ----
        if epoch == 0:
            self._model.fit(self._X_train, self._y_train)

        # 闭环模型 (total_epochs=1): 全量训练集评估，结果最准确
        if self.total_epochs == 1:
            y_pred = self._model.predict(self._X_train)
            return self._compute_metrics(self._y_train, y_pred, prefix='train_')

        # 多轮场景 (如用户强制设置): 每轮在随机子集上评估，模拟渐进曲线
        eval_frac = 0.1 + 0.9 * progress  # 10% → 100%
        n_total = len(self._X_train)
        sample_size = max(100, int(n_total * eval_frac))
        sample_size = min(sample_size, n_total)

        indices = np.random.RandomState(epoch * 7 + 13).choice(n_total, size=sample_size, replace=False)
        X_subset = self._safe_index(self._X_train, indices)
        y_subset = self._safe_index(self._y_train, indices)

        y_pred = self._model.predict(X_subset)
        return self._compute_metrics(y_subset, y_pred, prefix='train_')

    def evaluate(self) -> dict:
        """在测试集上评估"""
        if self.task_type == 'clustering':
            # 聚类: 无 ground truth, 用 X_test 评估聚类质量
            if hasattr(self._model, 'predict'):
                test_labels = self._model.predict(self._X_test)
            elif hasattr(self._model, 'fit_predict'):
                test_labels = self._model.fit_predict(self._X_test)
            else:
                self._model.fit(self._X_test)
                test_labels = self._model.labels_
            return self._compute_metrics(X=self._X_test, labels=test_labels, prefix='test_')
        else:
            y_pred = self._model.predict(self._X_test)
            return self._compute_metrics(self._y_test, y_pred, prefix='test_')

    def run_cross_validation(self, return_train_score: bool = False) -> dict:
        """使用 StratifiedKFold 交叉验证评估模型泛化能力。

        这是一个额外的评估层，不影响现有的单次 train/test split 训练流程。
        仅在训练完成后调用，用于获得更可靠的性能估计。

        Returns:
            dict with keys: cv_mean, cv_std, cv_scores, cv_folds, n_samples, error
            如果 CV 不可用 (样本太少/类别太少) 返回 error 说明原因
        """
        import numpy as np
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        cv_folds = int(self.hyperparams.get('cv_folds', 0))
        if cv_folds < 2:
            # Auto-compute: min(5, n_samples // 50), need at least 2
            n_total = (
                len(self._X_train) + len(self._X_test)
                if hasattr(self, '_X_test') and self._X_test is not None
                else len(self._X_train)
            )
            cv_folds = max(2, min(5, n_total // 50))
            self.callback.on_log(f'[CV] 自动设置 cv_folds={cv_folds} (n_samples={n_total})')

        try:
            # 使用全量 X, y (train+test) 做 CV, 获得更稳定的泛化估计
            X_full = np.vstack([self._X_train, self._X_test]) if hasattr(self, '_X_train') else self._X_train
            y_full = np.concatenate([self._y_train, self._y_test]) if hasattr(self, '_y_test') else self._y_train

            n_samples = len(y_full)
            if n_samples < 3 * cv_folds:
                return {
                    'error': f'样本量不足 ({n_samples} < {3 * cv_folds})',
                    'cv_mean': 0,
                    'cv_std': 0,
                    'cv_scores': [],
                    'cv_folds': cv_folds,
                    'n_samples': n_samples,
                }

            # 检查类别最少样本数
            from collections import Counter

            class_counts = Counter(y_full)
            min_count = min(class_counts.values())
            actual_folds = min(cv_folds, min_count, n_samples // 3)
            if actual_folds < 2:
                return {
                    'error': f'最少类别样本数={min_count}, 无法进行CV',
                    'cv_mean': 0,
                    'cv_std': 0,
                    'cv_scores': [],
                    'cv_folds': cv_folds,
                    'n_samples': n_samples,
                }

            skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(self._model, X_full, y_full, cv=skf, scoring='accuracy', n_jobs=1)

            return {
                'cv_mean': float(np.mean(scores)),
                'cv_std': float(np.std(scores)),
                'cv_scores': [float(s) for s in scores],
                'cv_folds': actual_folds,
                'n_samples': n_samples,
                'error': None,
            }
        except Exception as e:
            return {'error': str(e), 'cv_mean': 0, 'cv_std': 0, 'cv_scores': [], 'cv_folds': cv_folds, 'n_samples': 0}

    def save_model(self, path: str):
        """使用 pickle 保存模型 (含 NLP vectorizer + class_labels)"""
        full_path = path + '.pkl'
        bundle = {
            'model': self._model,
            'scaler': self._scaler,
            'label_encoders': self._label_encoders,
            'feature_names': list(self._X_train.columns),
            'task_type': self.task_type,
            'algorithm': self.algorithm,
        }
        # NLP: 保存 vectorizer 以便预测时复现特征转换
        if self._vectorizer is not None:
            bundle['vectorizer'] = self._vectorizer
        # 保存人类可读的类别标签
        if self._class_labels:
            bundle['class_labels'] = self._class_labels

        with open(full_path, 'wb') as f:
            pickle.dump(bundle, f)
        self.callback.on_log(f'模型已保存到: {full_path}')

    # ============ 检查点 ============

    def save_checkpoint(self):
        """保存 sklearn 训练快照 (仅 warm_start / partial_fit 模型)"""
        if self._model is None:
            return
        # closed-form 算法 (total_epochs=1) 不需要检查点
        if self.total_epochs <= 1:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        ckpt_path = os.path.join(self.output_dir, 'checkpoint.pkl')
        bundle = {
            'model': self._model,
            'epoch': self._current_epoch,
            'algorithm': self.algorithm,
            'task_type': self.task_type,
            'scaler': self._scaler,
        }
        if self._vectorizer is not None:
            bundle['vectorizer'] = self._vectorizer
        with open(ckpt_path, 'wb') as f:
            pickle.dump(bundle, f)

    @staticmethod
    def load_checkpoint(output_dir: str) -> dict:
        import pickle

        ckpt_path = os.path.join(output_dir, 'checkpoint.pkl')
        if not os.path.exists(ckpt_path):
            return {}
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        return {'epoch': ckpt.get('epoch', 0)}

    @staticmethod
    def has_checkpoint(output_dir: str) -> bool:
        return os.path.exists(os.path.join(output_dir, 'checkpoint.pkl'))

    # ============ 私有方法 ============

    @staticmethod
    def _safe_index(data, indices=None):
        """安全索引: 兼容 pandas DataFrame/Series 和 numpy array
        - indices=None: 返回全部数据
        - indices=int: 返回单个元素
        - indices=array: 返回索引子集
        """
        if indices is None:
            return data.iloc[:] if hasattr(data, 'iloc') else data[:]
        if hasattr(data, 'iloc'):
            return data.iloc[indices]
        return data[indices]

    def _compute_metrics(self, y_true=None, y_pred=None, prefix: str = '', X=None, labels=None) -> dict:
        """
        计算分类、回归和聚类任务的评估指标

        分类任务同时计算 weighted 和 macro 两种平均方式:
        - weighted: 按每个类别的样本数加权, 大类别影响更大
        - macro: 每个类别权重相同, 对小类别更敏感, 能暴露类别间差异

        聚类任务使用无监督指标:
        - silhouette_score: 轮廓系数 (-1到1, 越高越好)
        - davies_bouldin_score: Davies-Bouldin指数 (越低越好)
        - calinski_harabasz_score: Calinski-Harabasz指数 (越高越好)
        - inertia: KMeans惯性 (簇内平方和)

        通过对比两种平均方式，可以判断模型在不同类别上的表现是否均衡。
        """
        # ---- 聚类指标 (无监督) ----
        if X is not None and labels is not None:
            metrics = {}
            try:
                from sklearn.metrics import (
                    adjusted_rand_score,
                    calinski_harabasz_score,
                    davies_bouldin_score,
                    normalized_mutual_info_score,
                    silhouette_score,
                )

                unique_labels = set(labels)
                n_labels = len(unique_labels)
                # 至少需要2个簇且不是所有点在同一簇才能计算
                if n_labels >= 2 and n_labels < len(labels):
                    metrics[f'{prefix}silhouette_score'] = round(float(silhouette_score(X, labels)), 4)
                    metrics[f'{prefix}davies_bouldin_score'] = round(float(davies_bouldin_score(X, labels)), 4)
                    metrics[f'{prefix}calinski_harabasz_score'] = round(float(calinski_harabasz_score(X, labels)), 4)

                # KMeans 特有: 惯性 (簇内平方和)
                if hasattr(self._model, 'inertia_'):
                    metrics[f'{prefix}inertia'] = round(float(self._model.inertia_), 4)

                # 外部验证指标 (仅测试集, 且 ground truth 可用时)
                if self._y_test is not None and prefix == 'test_':
                    try:
                        metrics[f'{prefix}adjusted_rand_score'] = round(
                            float(adjusted_rand_score(self._y_test, labels)), 4
                        )
                        metrics[f'{prefix}normalized_mutual_info_score'] = round(
                            float(normalized_mutual_info_score(self._y_test, labels)), 4
                        )
                    except Exception:
                        pass
            except Exception:
                pass
            return metrics

        # ---- 分类 / 回归指标 (监督) ----
        metrics = {}
        if self.task_type == 'classification':
            # 基础准确率
            metrics[f'{prefix}accuracy'] = round(float(accuracy_score(y_true, y_pred)), 4)
            for name, func in [
                ('precision_weighted', lambda: precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                ('recall_weighted', lambda: recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                ('f1_weighted', lambda: f1_score(y_true, y_pred, average='weighted', zero_division=0)),
                ('precision_macro', lambda: precision_score(y_true, y_pred, average='macro', zero_division=0)),
                ('recall_macro', lambda: recall_score(y_true, y_pred, average='macro', zero_division=0)),
                ('f1_macro', lambda: f1_score(y_true, y_pred, average='macro', zero_division=0)),
            ]:
                full_name = f'{prefix}{name}'
                with contextlib.suppress(Exception):
                    metrics[full_name] = round(float(func()), 4)
        else:
            metrics[f'{prefix}mse'] = round(float(mean_squared_error(y_true, y_pred)), 4)
            metrics[f'{prefix}mae'] = round(float(mean_absolute_error(y_true, y_pred)), 4)
            metrics[f'{prefix}r2'] = round(float(r2_score(y_true, y_pred)), 4)

        return metrics
