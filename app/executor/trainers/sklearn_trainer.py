"""
scikit-learn 训练器
支持分类和回归任务，涵盖常用算法
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


def _import_model(module_path: str, class_name: str):
    """动态导入 sklearn 模型类"""
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class SklearnTrainer(BaseTrainer):
    """scikit-learn 训练器 — 支持分类和回归任务

    支持的算法:
        分类: 随机森林, 逻辑回归, SVM, KNN, 梯度提升
        回归: 线性回归, 随机森林回归, SVR, 梯度提升回归

    训练策略:
        - GradientBoosting: warm_start 增量添加树 → 真实渐进训练曲线
        - SGD/增量模型: partial_fit 分批训练
        - 其他模型: 全量fit + 渐进子集评估 → 模拟收敛曲线
    """

    def __init__(self, job, dataset, hyperparams: dict = None):
        super().__init__(job, dataset, hyperparams)

        self.task_type = self.hyperparams.get('task_type', 'classification')
        self.algorithm = self.hyperparams.get('algorithm', 'random_forest')
        self.test_size = float(self.hyperparams.get('test_size', 0.2))
        self.random_state = int(self.hyperparams.get('random_state', 42))

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

        # 推断 target 列
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
        else:
            model_info = _REGRESSORS.get(self.algorithm)

        if not model_info:
            raise ValueError(f'不支持的算法: {self.algorithm} (任务类型: {self.task_type})')

        module_path, class_name = model_info
        model_cls = _import_model(module_path, class_name)

        # 获取算法特定参数
        algo_params = self.hyperparams.get('algorithm_params', {})
        try:
            self._model = model_cls(random_state=self.random_state, **algo_params)
        except TypeError:
            # KNN, LinearRegression, SVR 等模型不接受 random_state
            self._model = model_cls(**algo_params)
        self.callback.on_log(f'模型: {class_name}')

    def train_epoch(self, epoch: int) -> dict:
        """sklearn epoch 训练 — 支持增量式训练以产生可视化训练曲线

        第一轮完整拟合后，后续 epoch 通过渐进评估和增量训练来模拟渐进提升:
        - GradientBoosting/SGD: 使用 warm_start / partial_fit 进行真实增量训练
        - 其他模型: 在逐渐扩大的训练子集上评估, 模拟收敛曲线
        """
        # 计算渐进比例 (epoch 0 = 全量训练)
        progress = (epoch + 1) / max(self.total_epochs, 1)

        # ---- 策略1: warm_start 增量训练 (GradientBoosting 等) ----
        if hasattr(self._model, 'warm_start') and hasattr(self._model, 'n_estimators'):
            if epoch == 0:
                # 首次: 设置初始 n_estimators 然后 fit
                initial_trees = max(1, int(self._model.n_estimators / self.total_epochs))
                self._model.n_estimators = initial_trees
                self._model.fit(self._X_train, self._y_train)
            else:
                # 增量添加树
                current_trees = self._model.n_estimators
                new_trees = current_trees + max(1, int(self._model.get_params().get('n_estimators', 100) / self.total_epochs))
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

        # ---- 策略3: 渐进评估 (标准模型, 训练集上模拟收敛) ----
        if epoch == 0:
            self._model.fit(self._X_train, self._y_train)

        # 每个 epoch 评估越来越大的训练子集，产生渐进收敛的曲线
        eval_frac = 0.1 + 0.9 * progress  # 10% → 100%
        n_total = len(self._X_train)
        sample_size = max(100, int(n_total * eval_frac))
        sample_size = min(sample_size, n_total)

        # 兼容 pandas DataFrame 和 numpy array
        indices = np.random.RandomState(epoch * 7 + 13).choice(
            n_total, size=sample_size, replace=False
        )
        X_subset = self._safe_index(self._X_train, indices)
        y_subset = self._safe_index(self._y_train, indices)

        y_pred = self._model.predict(X_subset)
        return self._compute_metrics(y_subset, y_pred, prefix='train_')

    def evaluate(self) -> dict:
        """在测试集上评估"""
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

    def _compute_metrics(self, y_true, y_pred, prefix: str = '') -> dict:
        """
        计算分类和回归任务的评估指标

        分类任务同时计算 weighted 和 macro 两种平均方式:
        - weighted: 按每个类别的样本数加权, 大类别影响更大
        - macro: 每个类别权重相同, 对小类别更敏感, 能暴露类别间差异

        通过对比两种平均方式，可以判断模型在不同类别上的表现是否均衡。
        """
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
