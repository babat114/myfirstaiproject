"""
scikit-learn 训练器
支持分类、回归和聚类任务，涵盖常用算法
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.executor.trainers.base import BaseTrainer

# ===================================================================
# sklearn 算法注册表
# 格式: '算法简称' → ('模块路径', '类名')
# 分类器 — 预测离散类别标签
# ===================================================================
_CLASSIFIERS = {
    'random_forest': ('sklearn.ensemble', 'RandomForestClassifier'),        # 随机森林分类器 - 集成多棵决策树投票
    'logistic_regression': ('sklearn.linear_model', 'LogisticRegression'),  # 逻辑回归 - 线性分类器, 输出类别概率
    'svm': ('sklearn.svm', 'SVC'),                                          # 支持向量机 - 寻找最大间隔超平面
    'knn': ('sklearn.neighbors', 'KNeighborsClassifier'),                   # K近邻 - 基于距离度量的懒惰学习
    'gradient_boosting': ('sklearn.ensemble', 'GradientBoostingClassifier'),# 梯度提升树 - 逐步拟合残差的集成方法
    'decision_tree': ('sklearn.tree', 'DecisionTreeClassifier'),            # 决策树 - 树形规则分裂, 可解释性强
}

# 回归器 — 预测连续数值
_REGRESSORS = {
    'linear_regression': ('sklearn.linear_model', 'LinearRegression'),              # 线性回归 - 最小二乘法拟合
    'ridge': ('sklearn.linear_model', 'Ridge'),                                     # 岭回归 - L2正则化线性回归
    'random_forest_regressor': ('sklearn.ensemble', 'RandomForestRegressor'),       # 随机森林回归器
    'svr': ('sklearn.svm', 'SVR'),                                                  # 支持向量回归 - epsilon不敏感损失
    'gradient_boosting_regressor': ('sklearn.ensemble', 'GradientBoostingRegressor'),# 梯度提升回归器
    'knn_regressor': ('sklearn.neighbors', 'KNeighborsRegressor'),                  # K近邻回归器 - 基于距离的回归
}

# 聚类器 — 无监督学习, 发现数据内在分组结构
_CLUSTERERS = {
    'kmeans':              ('sklearn.cluster', 'KMeans'),                       # K-Means — 基于质心的划分聚类
    'dbscan':              ('sklearn.cluster', 'DBSCAN'),                       # DBSCAN — 基于密度的空间聚类
    'agglomerative':  ('sklearn.cluster', 'AgglomerativeClustering'),     # 层次聚类 — 自底向上合并
    'minibatch_kmeans':    ('sklearn.cluster', 'MiniBatchKMeans'),              # MiniBatch K-Means — 小批量增量聚类
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
        'linear_regression', 'ridge', 'logistic_regression',
        'svm', 'svr', 'knn', 'knn_regressor',
        'kmeans', 'dbscan', 'agglomerative', 'minibatch_kmeans',
        'decision_tree',  # 决策树一次fit完全生长，多轮重复无意义
    }

    # 需要正则化防过拟合的算法默认参数
    _REGULARIZE_DEFAULTS = {
        'decision_tree': {'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 5},
        'random_forest': {'max_depth': 15, 'min_samples_leaf': 5},
        'gradient_boosting': {'max_depth': 5, 'min_samples_leaf': 10},
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

    def load_data(self):
        """从 Dataset 文件加载数据并预处理"""
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
        elif fmt == 'txt':
            df = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(f'不支持的文件格式: {fmt}')

        # ================================================================
        # 聚类: 无监督学习, 不需要目标列
        # ================================================================
        if self.task_type == 'clustering':
            target_col = self.hyperparams.get('target_column')
            if target_col and target_col in df.columns:
                # 保留 y 用于 ARI/NMI 外部验证 (不参与训练)
                self._y_full = df[target_col].copy()
                X = df.drop(columns=[target_col])
                self.callback.on_log(f'目标列 (仅用于外部验证): {target_col}, 特征数: {len(X.columns)}')
            else:
                self._y_full = None
                X = df.copy()
                self.callback.on_log(f'无监督聚类, 特征数: {len(X.columns)}')

            self.callback.on_log(f'数据形状: {df.shape}, 测试比例: {self.test_size}')

            # 预处理: 缺失值填充 + 分类特征编码 + 标准化 (同监督学习)
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

            num_cols_after = X.select_dtypes(include=[np.number]).columns
            if len(num_cols_after) > 0:
                self._scaler = StandardScaler()
                X[num_cols_after] = self._scaler.fit_transform(X[num_cols_after])

            # 划分训练/测试集 (无 y, 无分层)
            self._X_train, self._X_test = train_test_split(
                X, test_size=self.test_size, random_state=self.random_state
            )
            # 如果保留了 y, 对齐划分用于 ARI/NMI
            if self._y_full is not None:
                _, self._y_test = train_test_split(
                    self._y_full, test_size=self.test_size, random_state=self.random_state
                )
                self._y_train = None  # 训练时不使用标签
            else:
                self._y_train = self._y_test = None

            self.callback.on_log(f'训练集: {len(self._X_train)} 样本, 测试集: {len(self._X_test)} 样本')
            return

        # ================================================================
        # 分类 / 回归: 监督学习, 需要目标列
        # ================================================================
        target_col = self.hyperparams.get('target_column')
        if not target_col:
            # 自动推断：优先用最后一列
            target_col = df.columns[-1]

        if target_col not in df.columns:
            raise ValueError(f'目标列 "{target_col}" 不存在。可用列: {list(df.columns)}')

        self.callback.on_log(f'目标列: {target_col}, 特征数: {len(df.columns) - 1}')
        self.callback.on_log(f'数据形状: {df.shape}, 测试比例: {self.test_size}')

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 处理缺失值
        num_cols = X.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy='mean')
            X[num_cols] = num_imputer.fit_transform(X[num_cols])

        cat_cols = X.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

        # 编码分类特征
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self._label_encoders[col] = le

        # 编码目标变量 (分类任务)
        if self.task_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            self._label_encoders['__target__'] = le

        # 标准化数值特征
        num_cols_after = X.select_dtypes(include=[np.number]).columns
        if len(num_cols_after) > 0:
            self._scaler = StandardScaler()
            X[num_cols_after] = self._scaler.fit_transform(X[num_cols_after])

        # 划分训练/测试集
        # 分类任务尝试分层采样，若某类别样本过少则退化为随机划分
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
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=stratify_y
        )

        self.callback.on_log(f'训练集: {len(self._X_train)} 样本, 测试集: {len(self._X_test)} 样本')

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

        # 获取算法特定参数
        algo_params = dict(self.hyperparams.get('algorithm_params', {}))

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
        if self.total_epochs > 1 and module_path not in (
            'sklearn.svm', 'sklearn.linear_model', 'sklearn.neighbors',
            'sklearn.cluster',
        ):
            if 'warm_start' not in algo_params:
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
            return self._compute_metrics(
                X=self._X_train, labels=self._labels_train, prefix='train_'
            )

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
                self._model.partial_fit(self._X_train, self._y_train,
                                        classes=np.unique(self._y_train))
            else:
                # 每次用不同子集进行partial_fit
                frac = 0.3
                n_total = len(self._X_train)
                n_samples = max(100, int(n_total * frac))
                indices = np.random.RandomState(epoch).randint(0, n_total, n_samples)
                self._model.partial_fit(
                    self._safe_index(self._X_train, indices),
                    self._safe_index(self._y_train, indices)
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

        indices = np.random.RandomState(epoch * 7 + 13).choice(
            n_total, size=sample_size, replace=False
        )
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
            return self._compute_metrics(
                X=self._X_test, labels=test_labels, prefix='test_'
            )
        else:
            y_pred = self._model.predict(self._X_test)
            return self._compute_metrics(self._y_test, y_pred, prefix='test_')

    def save_model(self, path: str):
        """使用 pickle 保存模型"""
        full_path = path + '.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump({
                'model': self._model,
                'scaler': self._scaler,
                'label_encoders': self._label_encoders,
                'feature_names': list(self._X_train.columns),
                'task_type': self.task_type,
                'algorithm': self.algorithm,
            }, f)
        self.callback.on_log(f'模型已保存到: {full_path}')

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

    def _compute_metrics(self, y_true=None, y_pred=None, prefix: str = '',
                         X=None, labels=None) -> dict:
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
                    silhouette_score, davies_bouldin_score, calinski_harabasz_score
                )
                from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

                unique_labels = set(labels)
                n_labels = len(unique_labels)
                # 至少需要2个簇且不是所有点在同一簇才能计算
                if n_labels >= 2 and n_labels < len(labels):
                    metrics[f'{prefix}silhouette_score'] = round(
                        float(silhouette_score(X, labels)), 4)
                    metrics[f'{prefix}davies_bouldin_score'] = round(
                        float(davies_bouldin_score(X, labels)), 4)
                    metrics[f'{prefix}calinski_harabasz_score'] = round(
                        float(calinski_harabasz_score(X, labels)), 4)

                # KMeans 特有: 惯性 (簇内平方和)
                if hasattr(self._model, 'inertia_'):
                    metrics[f'{prefix}inertia'] = round(float(self._model.inertia_), 4)

                # 外部验证指标 (仅测试集, 且 ground truth 可用时)
                if self._y_test is not None and prefix == 'test_':
                    try:
                        metrics[f'{prefix}adjusted_rand_score'] = round(
                            float(adjusted_rand_score(self._y_test, labels)), 4)
                        metrics[f'{prefix}normalized_mutual_info_score'] = round(
                            float(normalized_mutual_info_score(self._y_test, labels)), 4)
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
            try:
                # weighted 平均 — 按样本数加权
                metrics[f'{prefix}precision_weighted'] = round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)), 4)
                metrics[f'{prefix}recall_weighted'] = round(float(recall_score(y_true, y_pred, average='weighted', zero_division=0)), 4)
                metrics[f'{prefix}f1_weighted'] = round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4)
                # macro 平均 — 各类别等权, 暴露小类别表现
                metrics[f'{prefix}precision_macro'] = round(float(precision_score(y_true, y_pred, average='macro', zero_division=0)), 4)
                metrics[f'{prefix}recall_macro'] = round(float(recall_score(y_true, y_pred, average='macro', zero_division=0)), 4)
                metrics[f'{prefix}f1_macro'] = round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 4)
            except Exception:
                pass
        else:
            metrics[f'{prefix}mse'] = round(float(mean_squared_error(y_true, y_pred)), 4)
            metrics[f'{prefix}mae'] = round(float(mean_absolute_error(y_true, y_pred)), 4)
            metrics[f'{prefix}r2'] = round(float(r2_score(y_true, y_pred)), 4)

        return metrics
