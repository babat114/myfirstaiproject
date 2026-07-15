"""
============================================
超参数自动调优服务 (v5 — 异步+实时进度)
支持 GridSearchCV 和 RandomizedSearchCV
覆盖分类 / 回归 / 聚类 三种任务类型

v5 新增:
  6. TuningProgressTracker — 线程安全进度追踪 + SSE推送
  7. run_grid_search_async / run_random_search_async — 后台线程执行
  8. 实时进度回调: 当前参数组合/当前CV分折/最佳分数/进度百分比
  9. 前端进度条 + 关键帧信息实时展示
============================================
"""

import contextlib
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd

from app import db, logger
from app._timezone import localnow
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.models.user import User


def _get_tuning_config(key: str, default):
    """安全获取超参数调优配置 (后台线程内无 Flask app context 时返回默认值)"""
    try:
        from flask import current_app

        return current_app.config.get(key, default)
    except RuntimeError:
        return default


# ===================================================================
# 实时进度追踪器 (v5 新增) — 线程安全的调优进度状态管理
# ===================================================================


class TuningProgressTracker:
    """线程安全的 GridSearchCV 调优进度追踪器 (单例)

    供后台调优线程写入, SSE端点读取, 无需数据库轮询.

    使用方式:
        tracker = TuningProgressTracker()
        tracker.init(tuning_id, total_steps, algorithm, task_type)
        tracker.update(step, params, score)              # 每个CV fold调用
        tracker.complete(tuning_id, result)               # 调优完成
        tracker.fail(tuning_id, error)                    # 调优失败
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._sessions = {}
                    cls._instance._lock = threading.Lock()
        return cls._instance

    def init(
        self, tuning_id: str, total_steps: int, algorithm: str, task_type: str, tuning_method: str = 'grid'
    ) -> dict:
        """初始化一个调优会话"""
        session = {
            'tuning_id': tuning_id,
            'status': 'running',  # running | completed | failed
            'algorithm': algorithm,
            'task_type': task_type,
            'tuning_method': tuning_method,
            'progress_percent': 0.0,
            'current_step': 0,
            'total_steps': total_steps,
            'current_params': {},
            'current_score': None,
            'best_score_so_far': None,
            'best_params_so_far': {},
            'elapsed_seconds': 0.0,
            'log_lines': [],
            'result': None,
            'error': None,
            'started_at': time.time(),
        }
        with self._lock:
            self._sessions[tuning_id] = session
        logger.info(f'TuningProgress: [{tuning_id}] 初始化 {algorithm}/{task_type}, 共 {total_steps} 个评估步骤')
        return session

    def update(self, tuning_id: str, step: int, params: dict = None, score: float = None, total: int = None):
        """更新进度 (每个 CV fold 评估完成后调用)

        Args:
            total: 可选, 动态更新 total_steps (聚类手动搜索时步数与初始化时不同)
        """
        with self._lock:
            s = self._sessions.get(tuning_id)
            if not s:
                return
            s['current_step'] = step
            # 允许动态更新总步数 (如聚类从 CV步数 改为 参数组合数)
            if total is not None:
                s['total_steps'] = total
            if s['total_steps'] > 0:
                # v7: 允许进度自然到达100%, 不再人工卡99%
                s['progress_percent'] = round(min(100.0, (step / s['total_steps']) * 100), 1)
            if params:
                s['current_params'] = params
            if score is not None:
                s['current_score'] = round(float(score), 4)
                if s['best_score_so_far'] is None or score > s['best_score_so_far']:
                    s['best_score_so_far'] = round(float(score), 4)
                    if params:
                        s['best_params_so_far'] = dict(params)
            s['elapsed_seconds'] = round(time.time() - s['started_at'], 1)

    def add_log(self, tuning_id: str, message: str):
        """追加日志行"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        line = f'[{timestamp}] {message}'
        with self._lock:
            s = self._sessions.get(tuning_id)
            if s:
                s['log_lines'].append(line)
                # 保留最近 50 行
                if len(s['log_lines']) > 50:
                    s['log_lines'] = s['log_lines'][-50:]

    def complete(self, tuning_id: str, result: dict):
        """调优完成"""
        with self._lock:
            s = self._sessions.get(tuning_id)
            if s:
                s['status'] = 'completed'
                s['progress_percent'] = 100.0
                s['result'] = result
                s['elapsed_seconds'] = round(time.time() - s['started_at'], 1)
                s['best_score_so_far'] = result.get('best_score', s['best_score_so_far'])
                s['best_params_so_far'] = result.get('best_params', s['best_params_so_far'])
        logger.info(f'TuningProgress: [{tuning_id}] 完成, best={s.get("best_score_so_far")}')

    def fail(self, tuning_id: str, error: str):
        """调优失败"""
        with self._lock:
            s = self._sessions.get(tuning_id)
            if s:
                s['status'] = 'failed'
                s['error'] = error
                s['elapsed_seconds'] = round(time.time() - s['started_at'], 1)
        logger.error(f'TuningProgress: [{tuning_id}] 失败: {error}')

    def get(self, tuning_id: str) -> dict | None:
        """获取当前进度 (SSE/轮询用) — 返回只读快照"""
        with self._lock:
            s = self._sessions.get(tuning_id)
            if not s:
                return None
            return dict(s)  # 浅拷贝避免并发修改问题

    def cleanup(self, tuning_id: str):
        """清理已完成/失败的会话 (保留5分钟后由get调用时惰性清理)"""
        with self._lock:
            s = self._sessions.get(tuning_id)
            if s and s['status'] in ('completed', 'failed'):
                age = time.time() - s['started_at']
                if age > _get_tuning_config('TUNING_SESSION_CLEANUP_TTL', 3600):  # 1小时后清理
                    del self._sessions[tuning_id]


# 全局单例
_tuning_tracker: TuningProgressTracker | None = None


def get_tuning_tracker() -> TuningProgressTracker:
    global _tuning_tracker
    if _tuning_tracker is None:
        _tuning_tracker = TuningProgressTracker()
    return _tuning_tracker


# ===================================================================
# 预定义超参数搜索空间 (v4: 覆盖分类+回归+聚类)
# ===================================================================
SEARCH_SPACES = {
    # ---- 分类器 ----
    'random_forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.7, 0.8, 1.0],
    },
    'logistic_regression': {
        'C': [0.01, 0.1, 0.5, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 3000, 5000],
    },
    'svm': {
        'C': [0.1, 1.0, 10.0, 100.0],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'degree': [2, 3, 4],
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2],
    },
    'decision_tree': {
        'max_depth': [None, 5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None],
    },
    # ---- 回归器 ----
    'linear_regression': {
        'fit_intercept': [True, False],
        'positive': [True, False],
    },
    'random_forest_regressor': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'gradient_boosting_regressor': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
    },
    'svr': {
        'C': [0.1, 1.0, 10.0, 100.0],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'epsilon': [0.01, 0.05, 0.1, 0.2],
        'degree': [2, 3],
    },
    'ridge': {
        'alpha': [0.01, 0.1, 0.5, 1.0, 10.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag'],
        'fit_intercept': [True, False],
    },
    'knn_regressor': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2],
    },
    # ---- 聚类器 (v4 新增) ----
    'kmeans': {
        'n_clusters': [2, 3, 4, 5, 7, 10, 15],
        'init': ['k-means++', 'random'],
        'max_iter': [100, 300, 500],
        'algorithm': ['lloyd', 'elkan'],
        'n_init': [10, 20],
    },
    'dbscan': {
        'eps': [0.1, 0.3, 0.5, 0.7, 1.0, 1.5],
        'min_samples': [3, 5, 10, 15, 20],
        'metric': ['euclidean', 'manhattan'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
    },
    'agglomerative': {
        'n_clusters': [2, 3, 5, 7, 10, 15],
        'linkage': ['ward', 'complete', 'average', 'single'],
        'metric': ['euclidean', 'manhattan', 'cosine'],
    },
    'minibatch_kmeans': {
        'n_clusters': [2, 3, 5, 7, 10, 15],
        'init': ['k-means++', 'random'],
        'max_iter': [100, 300],
        'batch_size': [100, 256, 512],
        'n_init': [3, 10],
    },
}

# sklearn MLP 搜索空间 (用于代理搜索)
MLP_SEARCH_SPACE = {
    'hidden_layer_sizes': [(64, 32), (128, 64, 32), (256, 128, 64)],
    'learning_rate_init': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64],
    'alpha': [1e-5, 1e-4, 1e-3],
    'early_stopping': [True, False],
    'validation_fraction': [0.1, 0.2],
    'max_iter': [200, 500],
}

# PyTorch MLP 搜索空间
PYTORCH_SEARCH_SPACE = {
    'hidden_layers': [[64, 32], [128, 64, 32], [256, 128, 64], [128, 128, 64, 32]],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
    'batch_size': [16, 32, 64, 128],
    'dropout': [0.1, 0.2, 0.3, 0.5],
    'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3],
}

# ── 自动模式: task_type → 算法列表 (AutoML 智能算法选择) ──
# 每种任务类型的快速扫描算法, 包含对应的推荐框架
AUTO_ALGORITHMS = {
    'classification': [
        {'algo': 'random_forest', 'framework': 'sklearn', 'label': 'Random Forest'},
        {'algo': 'gradient_boosting', 'framework': 'sklearn', 'label': 'Gradient Boosting'},
        {'algo': 'logistic_regression', 'framework': 'sklearn', 'label': 'Logistic Regression'},
        {'algo': 'svm', 'framework': 'sklearn', 'label': 'SVM'},
        {'algo': 'knn', 'framework': 'sklearn', 'label': 'KNN'},
        {'algo': 'decision_tree', 'framework': 'sklearn', 'label': 'Decision Tree'},
        {'algo': 'mlp', 'framework': 'pytorch', 'label': 'PyTorch MLP'},
    ],
    'regression': [
        {'algo': 'random_forest_regressor', 'framework': 'sklearn', 'label': 'Random Forest'},
        {'algo': 'gradient_boosting_regressor', 'framework': 'sklearn', 'label': 'Gradient Boosting'},
        {'algo': 'linear_regression', 'framework': 'sklearn', 'label': 'Linear Regression'},
        {'algo': 'ridge', 'framework': 'sklearn', 'label': 'Ridge'},
        {'algo': 'svr', 'framework': 'sklearn', 'label': 'SVR'},
        {'algo': 'knn_regressor', 'framework': 'sklearn', 'label': 'KNN'},
        {'algo': 'mlp', 'framework': 'pytorch', 'label': 'PyTorch MLP'},
    ],
    'clustering': [
        {'algo': 'kmeans', 'framework': 'sklearn', 'label': 'K-Means'},
        {'algo': 'minibatch_kmeans', 'framework': 'sklearn', 'label': 'MiniBatch K-Means'},
        {'algo': 'agglomerative', 'framework': 'sklearn', 'label': '层次聚类'},
        {'algo': 'dbscan', 'framework': 'sklearn', 'label': 'DBSCAN'},
    ],
}

# ---- 聚类算法集合 (v4 新增) ----
_CLUSTERING_ALGOS = {'kmeans', 'dbscan', 'agglomerative', 'minibatch_kmeans'}

# ---- 算法名映射: classification <-> regression ----
_CLS_TO_REG = {
    'random_forest': 'random_forest_regressor',
    'gradient_boosting': 'gradient_boosting_regressor',
    'knn': 'knn_regressor',
    'svm': 'svr',
    'decision_tree': 'random_forest_regressor',
    'logistic_regression': 'ridge',
    'mlp': 'mlp',
}
_REG_TO_CLS = {
    'random_forest_regressor': 'random_forest',
    'gradient_boosting_regressor': 'gradient_boosting',
    'svr': 'svm',
    'knn_regressor': 'knn',
    'linear_regression': 'logistic_regression',
    'ridge': 'logistic_regression',
    'mlp': 'mlp',
}


# ===================================================================
# 数据加载与验证 (v4: 增强防御)
# ===================================================================


def _load_dataset_file(dataset: Dataset) -> pd.DataFrame | None:
    """加载数据集文件"""
    from app.utils.data_io import load_dataframe

    return load_dataframe(dataset.file_path, dataset.file_format.lower())


def _validate_not_empty(df: pd.DataFrame) -> None:
    """验证DataFrame非空"""
    if df is None or len(df) == 0:
        raise ValueError('数据集为空, 无法进行超参数搜索。')


def _validate_min_samples(n_samples: int, cv: int) -> None:
    """验证样本数足够支持CV分折"""
    if n_samples < 10:
        raise ValueError(f'样本数过少 ({n_samples}), 至少需要10个样本。请选择更简单的验证策略或收集更多数据。')
    min_required = cv * 2
    if n_samples < min_required:
        raise ValueError(
            f'样本数 ({n_samples}) 不足以支持 {cv}-折交叉验证 '
            f'(至少需要 {min_required})。请减少CV折数或使用简单train/test分割。'
        )


def _detect_and_remove_inf(y) -> tuple:
    """检测并移除inf值, 返回 (cleaned_y, inf_count)

    同时支持 numpy array 和 pandas Series.
    保留原始 dtype (分类任务整数标签不转为 float64).
    """
    from pandas.api.types import is_integer_dtype

    is_int_like = hasattr(y, 'dtype') and is_integer_dtype(y)
    y_arr = np.asarray(y, dtype=np.float64)
    inf_mask = ~np.isfinite(y_arr)
    n_inf = inf_mask.sum()
    if n_inf > 0:
        logger.warning(f'目标列包含 {n_inf} 个 inf/nan 值, 已移除对应行')
        # 对原始数据和 numpy 数组同步过滤
        if hasattr(y, 'iloc'):
            return y.loc[~inf_mask], n_inf
        return y_arr[~inf_mask], n_inf
    # 无 inf: 返回原始类型以保持一致性
    if hasattr(y, 'iloc'):
        return y, 0
    if is_int_like:
        return y, 0
    return y_arr, 0


# ===================================================================
# 数据准备 — 分类 / 回归 / 聚类 三条路径
# ===================================================================


def _prepare_xy(df: pd.DataFrame, target_column: str):
    """从DataFrame中分离X和y, 处理基础问题(缺失列)"""
    if not target_column or target_column not in df.columns:
        target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y_raw = df[target_column]
    return X, y_raw, target_column


def _impute_and_encode_X(X: pd.DataFrame) -> np.ndarray:
    """对X做缺失值填充 + 分类编码 + 标准化, 返回 float64 numpy array"""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    X = X.copy()

    # 数值列: 均值填充
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X[num_cols] = SimpleImputer(strategy='mean').fit_transform(X[num_cols])

    # 类别列: 填'__missing__'后 LabelEncoder
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        X[col] = X[col].fillna('__missing__')
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # 标准化 → float64
    X_scaled = StandardScaler().fit_transform(X).astype(np.float64)
    return X_scaled


def _prepare_y_for_classification(y_raw) -> np.ndarray:
    """将任意目标列强制转换为 int64 分类标签

    核心策略: 全部转str → LabelEncoder → int64
    sklearn 的 type_of_target 对 int64 数组必然返回 'binary' 或 'multiclass'.

    重要: y_raw 不应包含 NaN/inf (调用方需先行清理).
    """
    from sklearn.preprocessing import LabelEncoder

    s = pd.Series(y_raw) if isinstance(y_raw, np.ndarray) else y_raw

    # 处理inf: 转为特殊字符串 '__inf__' / '__ninf__'
    s = s.copy()
    if hasattr(s, 'replace'):
        s.replace([np.inf, -np.inf], ['__inf__', '__ninf__'], inplace=True)
    else:
        # numpy array 直接替换
        s = pd.Series(s)
        s.replace([np.inf, -np.inf], ['__inf__', '__ninf__'], inplace=True)

    # 全部转 str → LabelEncoder → int64
    y_str = s.astype(str).values
    y_int = LabelEncoder().fit_transform(y_str).astype(np.int64)
    return y_int


def _prepare_y_for_regression(y_raw) -> np.ndarray:
    """将目标列转换为 float64 (回归)

    重要: y_raw 不应包含 NaN/inf (调用方需先行清理).
    """
    if isinstance(y_raw, np.ndarray):
        return y_raw.astype(np.float64)
    elif hasattr(y_raw, 'values'):
        return y_raw.values.astype(np.float64)
    else:
        return np.array(y_raw, dtype=np.float64)


def _prepare_data_robust(df, target_column, task_type):
    """完整的数据准备流程 — 带多层防御验证

    支持三种任务类型:
      - classification: 强制LabelEncoder编码y, 保证type_of_target正确
      - regression:     强制float64编码y
      - clustering:     只准备X, 不处理y (无监督学习)

    Returns:
        (X, y_or_None, task_type, scoring)
        - 聚类返回 y=None
        - 分类/回归的 task_type/scoring 已被修正为一致
    """
    from sklearn.utils.multiclass import type_of_target

    _validate_not_empty(df)

    # ================================================================
    # 聚类路径: 无监督学习, 不需要y
    # ================================================================
    if task_type == 'clustering':
        X = df.drop(columns=[target_column]) if target_column and target_column in df.columns else df.copy()
        X_scaled = _impute_and_encode_X(X)
        return X_scaled, None, 'clustering', 'silhouette'

    # ================================================================
    # 分类 / 回归路径
    # ================================================================

    # ---- Step 0: 确保目标列存在 ----
    if not target_column or target_column not in df.columns:
        target_column = df.columns[-1]
        logger.warning(f'目标列未指定或不存在, 使用最后一列: {target_column}')

    # ---- Step 1: 移除目标列的 NaN / inf (保证 X/y 行数一致) ----
    nan_mask = df[target_column].isna()
    inf_mask = ~df[target_column].apply(lambda x: np.isfinite(x) if isinstance(x, (int, float)) else True)
    bad_mask = nan_mask | inf_mask
    n_bad = bad_mask.sum()
    if n_bad > 0:
        df = df.loc[~bad_mask].copy()
        logger.warning(f'_prepare_data_robust: 目标列 "{target_column}" 包含 {n_bad} 个无效值 (NaN/inf), 已移除对应行')
    _validate_not_empty(df)

    # ---- Step 2: 分离 X / y ----
    X, y_raw, target_column = _prepare_xy(df, target_column)
    len(X)

    # ---- Step 3: 编码 X ----
    X_scaled = _impute_and_encode_X(X)

    # ---- Step 4: 根据 task_type 编码 y ----
    if task_type == 'classification':
        y_encoded = _prepare_y_for_classification(y_raw)
    else:
        y_encoded = _prepare_y_for_regression(y_raw)

    # ---- Step 5: 防御验证 — 确保 y type 与 task_type 一致 ----
    actual_target_type = type_of_target(y_encoded)

    # 情况A: 声明分类但y是连续 → 尝试强制编码, 仍不行为回归
    if task_type == 'classification' and actual_target_type == 'continuous':
        logger.warning(
            f'DEFENSE: y type_of_target={actual_target_type} 但 task_type=classification, 强制 LabelEncoder 重新编码'
        )
        # 回到原始y_raw重新编码 (不用已损坏的y_encoded)
        y_encoded = _prepare_y_for_classification(y_raw)
        actual_target_type = type_of_target(y_encoded)

        if actual_target_type == 'continuous':
            # 仍然连续 → 数据本质是连续的, 切换为回归
            logger.warning('DEFENSE: 编码后仍为continuous, 自动切换为 regression')
            y_encoded = _prepare_y_for_regression(y_raw)
            return X_scaled, y_encoded, 'regression', 'neg_mean_squared_error'

        return X_scaled, y_encoded, 'classification', 'accuracy'

    # 情况B: 声明回归但y是离散 → 切换为分类
    if task_type == 'regression' and actual_target_type in ('binary', 'multiclass'):
        logger.warning(
            f'DEFENSE: y type_of_target={actual_target_type} 但 task_type=regression, 自动切换为 classification'
        )
        return X_scaled, y_encoded, 'classification', 'accuracy'

    # 情况C: 一切正常 — 确保 scoring 匹配
    scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'

    return X_scaled, y_encoded, task_type, scoring


# ===================================================================
# 模型工厂 — 根据 task_type + algorithm 创建模型 (v4: +聚类)
# ===================================================================


def _create_model(algorithm: str, task_type: str, is_mlp: bool = False, random_state: int = None):
    """根据算法名和任务类型创建 sklearn 模型

    支持: classification / regression / clustering

    Args:
        random_state: 随机种子 (None=真随机, int=可复现)
    """
    from app.executor.trainers.sklearn_trainer import _CLASSIFIERS, _CLUSTERERS, _REGRESSORS, _import_model

    # MLP 特殊处理
    if is_mlp:
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        kwargs = dict(activation='relu', solver='adam', early_stopping=False, max_iter=500)
        if random_state is not None:
            kwargs['random_state'] = random_state
        if task_type == 'classification':
            return MLPClassifier(**kwargs)
        else:
            return MLPRegressor(**kwargs)

    # 聚类
    if task_type == 'clustering':
        model_info = _CLUSTERERS.get(algorithm)
        if not model_info:
            raise ValueError(f'不支持的聚类算法: "{algorithm}"。可用聚类算法: {list(_CLUSTERERS.keys())}')
        model_cls = _import_model(*model_info)
        try:
            return model_cls(random_state=random_state) if random_state is not None else model_cls()
        except TypeError:
            return model_cls()

    # 分类 / 回归
    model_map = _CLASSIFIERS if task_type == 'classification' else _REGRESSORS
    model_info = model_map.get(algorithm)

    if not model_info:
        alt_map = _REGRESSORS if task_type == 'classification' else _CLASSIFIERS
        alt_info = alt_map.get(algorithm)
        if alt_info:
            alt_task = 'regression' if task_type == 'classification' else 'classification'
            raise ValueError(
                f'算法 "{algorithm}" 不支持 {task_type} 任务。请使用 {alt_task} 任务类型, 或切换到对应算法。'
            )
        if algorithm in _CLUSTERING_ALGOS:
            raise ValueError(f'算法 "{algorithm}" 是聚类算法, 不支持 {task_type} 任务。请将任务类型设为 clustering。')
        raise ValueError(
            f'不支持的算法: "{algorithm}"。'
            f'可用分类算法: {list(_CLASSIFIERS.keys())}。'
            f'可用回归算法: {list(_REGRESSORS.keys())}。'
            f'可用聚类算法: {list(_CLUSTERERS.keys())}。'
        )

    model_cls = _import_model(*model_info)
    try:
        return model_cls(random_state=random_state) if random_state is not None else model_cls()
    except TypeError:
        return model_cls()


# ===================================================================
# 聚类专用评分器 (v4 新增)
# ===================================================================


def _clustering_scorer(estimator, X, y_true=None):
    """聚类专用评分函数 — silhouette_score

    作为 GridSearchCV 的 scoring 参数直接传入 (不使用 make_scorer,
    因为 sklearn 在 y=None 时不传递 y_true 参数).

    支持所有 sklearn 聚类器:
      - KMeans / MiniBatchKMeans / AgglomerativeClustering → predict
      - DBSCAN → labels_ (fit_predict)
    """
    from sklearn.metrics import silhouette_score

    # 获取聚类标签
    if hasattr(estimator, 'predict'):
        labels = estimator.predict(X)
    elif hasattr(estimator, 'labels_'):
        labels = estimator.labels_
    else:
        try:
            labels = estimator.fit_predict(X)
        except Exception:
            return -1.0

    unique_labels = set(labels)
    n_labels = len(unique_labels)

    # 需要至少2个簇, 且不能全部点在同一个簇
    if n_labels < 2 or n_labels >= len(labels):
        return -1.0

    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return -1.0


# ===================================================================
# error_score 安全值 (v9 — 防止回归 scoring 下非法组合得高分)
# ===================================================================


def _error_score_for_scoring(scoring) -> float:
    """根据 scoring 方向返回安全的 error_score

    防止 error_score=0 在负向指标 (neg_mean_squared_error 等) 中
    被错误地当作"最佳分数"，导致非法参数组合排名第一。

    - 正向指标 (accuracy, f1, precision, recall, silhouette, r2): 0 = 最低/中性
    - 负向指标 (neg_*): float('-inf') — 确保失败组合不排到前面
    """
    if isinstance(scoring, str):
        if scoring.startswith('neg_'):
            return float('-inf')
        # r2 可能是负数 (比随机还差), 但 0 通常已是最差合理值
        return 0.0
    # callable scorer (如 _clustering_scorer): 返回范围 [-1, 1], 0 中性
    return 0.0


# ===================================================================
# 进度评分器工厂 (v6) — 包装原始评分器 + 进度回调, n_jobs=1 安全
# ===================================================================


def _make_progress_scorer(actual_scoring, total_steps: int, progress_callback: Callable):
    """创建带进度追踪的评分器 (闭包, n_jobs=1 无序列化问题)

    关键: 将评分委托给原始评分指标, 确保 GridSearchCV 使用正确的 metric.
    同时在每次评分后调用 progress_callback(step, total, params=None, score=float).

    Args:
        actual_scoring: str 评分名 (如 'accuracy') 或 callable (如 _clustering_scorer)
        total_steps: 总评估步数
        progress_callback: callable(step, total, params, score)

    Returns:
        callable: scorer(estimator, X, y_true) -> float
    """
    from sklearn.metrics import get_scorer

    # —— 步骤1: 获取基础评分器 ——
    if isinstance(actual_scoring, str):
        base_scorer = get_scorer(actual_scoring)
    elif callable(actual_scoring):
        base_scorer = actual_scoring
    else:
        base_scorer = get_scorer('accuracy')

    # —— 步骤2: 使用可变容器追踪计数 (闭包捕获) ——
    counter = [0]

    # —— 步骤3: 创建进度评分器 ——
    def _progress_scorer(estimator, X, y_true=None):
        """评分 + 进度更新 (委托给原始评分器, 不重新计算分数)

        签名兼容性:
          - 监督学习 (分类/回归): sklearn 调用 scorer(estimator, X, y_true) — 3个位置参数
          - 无监督学习 (聚类):   sklearn 调用 scorer(estimator, X) — 仅2个参数, y_true=None
        """
        counter[0] += 1
        try:
            # 委托给原始评分器 — 确保 GridSearchCV 使用正确的 metric
            score_val = float(base_scorer(estimator, X, y_true))
        except Exception:
            # 回退: 某些 scorer (如 clustering) 不接受 y_true 参数
            try:
                if y_true is None:
                    score_val = float(base_scorer(estimator, X))
                else:
                    score_val = float(base_scorer(estimator, X, y_true))
            except Exception:
                score_val = 0.0

        # 更新进度
        with contextlib.suppress(Exception):
            progress_callback(counter[0], total_steps, None, score_val)

        return score_val

    return _progress_scorer


# ===================================================================
# 聚类手动参数搜索 (v7) — 无监督学习不需要交叉验证
# ===================================================================


def _manual_clustering_search(
    base_model, param_grid, X, progress_callback: Callable = None, max_combos: int = None, random_state: int = None
) -> dict:
    """手动遍历参数组合 — 聚类在全量数据上fit+evaluate, 无需CV

    为什么不用 GridSearchCV?
      - 聚类是无监督学习, 没有 ground-truth y
      - CV 把数据拆成 train/test fold, 但 train 上找到的簇和 test 上的簇没有对应关系
      - 结果: silhouette_score 在 CV 下既低又不稳定
      - 正确做法: 全量数据 fit → 全量数据 evaluate

    对每个参数组合:
      1. 创建新的模型实例 (fit 前设置参数, 避免 KMeans fit() 不接受 kwargs)
      2. fit(X) — 全量数据
      3. 计算 silhouette_score(X, labels) — 全量数据
      4. 调用 progress_callback (如果提供)

    Args:
        max_combos: 最大参数组合数 (采样限制, 用于AutoML快速扫描)

    Returns:
        dict: 兼容 GridSearchCV cv_results_ 的格式, 可直接传入 _build_result
    """
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import ParameterGrid

    param_list = list(ParameterGrid(param_grid))
    n_combinations = len(param_list)

    # AutoML 快速扫描: 采样限制组合数
    if max_combos and n_combinations > max_combos:
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        indices = rng.choice(n_combinations, max_combos, replace=False)
        param_list = [param_list[i] for i in indices]
        n_combinations = max_combos
        logger.info(f'聚类参数采样: {len(list(ParameterGrid(param_grid)))} → {max_combos} 组合')

    if n_combinations == 0:
        return {
            'success': False,
            'error': '参数网格为空, 无法进行聚类搜索。',
        }

    results = []
    best_score = -2.0  # silhouette 范围 [-1, 1], -2 确保任何有效值都能覆盖
    best_params = None

    # ── 性能优化: silhouette_score 是 O(n²), 大数据集时对子样本评分 ──
    # fit 用全量数据 (KMeans 是 O(n·k·d), 很快)
    # score 用子样本 (避免 O(n²) 爆炸)
    MAX_SCORE_SAMPLES = _get_tuning_config('CLUSTERING_MAX_SCORE_SAMPLES', 3000)
    n_samples = len(X)
    if n_samples > MAX_SCORE_SAMPLES:
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        score_indices = rng.choice(n_samples, MAX_SCORE_SAMPLES, replace=False)
        X_score = X[score_indices]
        logger.info(
            f'聚类评分子采样: {n_samples} → {MAX_SCORE_SAMPLES} 样本 '
            f'(silhouette_score 加速 {(n_samples / MAX_SCORE_SAMPLES) ** 2:.0f}x)'
        )
    else:
        X_score = X

    for i, params in enumerate(param_list):
        step = i + 1
        score = -1.0
        current_params = dict(params)

        try:
            # 创建新模型 — 在构造时设置参数 (避免 KMeans.fit() 不接受额外 kwargs)
            model_cls = type(base_model)
            try:
                model = (
                    model_cls(**current_params, random_state=random_state)
                    if random_state is not None
                    else model_cls(**current_params)
                )
            except TypeError:
                model = model_cls(**current_params)

            # 全量数据 fit
            model.fit(X)

            # 获取聚类标签 — 用全量数据 predict
            if hasattr(model, 'predict'):
                labels_all = model.predict(X)
            elif hasattr(model, 'labels_'):
                labels_all = model.labels_
            else:
                try:
                    labels_all = model.fit_predict(X)
                except Exception:
                    score = -1.0
                    raise ValueError('无法获取聚类标签')

            # silhouette_score — 用于评分的子样本 (避免 O(n²) 在全量数据上爆炸)
            labels = labels_all[score_indices] if n_samples > MAX_SCORE_SAMPLES else labels_all

            unique_labels = set(labels)
            n_unique = len(unique_labels)

            score = float(silhouette_score(X_score, labels)) if n_unique >= 2 and n_unique < len(labels) else -1.0

        except Exception as e:
            logger.warning(f'聚类参数组合 #{step} 失败 {current_params}: {e}')
            score = -1.0

        # 记录结果
        results.append(
            {
                'params': current_params,
                'mean_score': round(score, 4),
                'std_score': 0.0,
                'rank': step,  # 临时, 后面重排
            }
        )

        # 追踪最佳
        if score > best_score:
            best_score = score
            best_params = current_params

        # 进度回调 — 每完成一个参数组合就推送
        if progress_callback:
            with contextlib.suppress(Exception):
                progress_callback(step, n_combinations, current_params, score)

    # 按分数降序排列 (最佳排第1)
    results.sort(key=lambda x: x['mean_score'], reverse=True)
    for rank_idx, r in enumerate(results):
        r['rank'] = rank_idx + 1

    if best_params is None:
        best_params = param_list[0] if param_list else {}
        best_score = -1.0

    logger.info(f'聚类手动搜索完成: {n_combinations} 参数组合, best_score={best_score:.4f}, best_params={best_params}')

    return {
        'success': True,
        'best_params': best_params,
        'best_score': round(best_score, 4),
        'cv_results': results[:20],
        'n_combinations': n_combinations,
        'manual_search': True,  # 标记: 非 GridSearchCV 结果
    }


# ===================================================================
# GridSearchCV / RandomizedSearchCV 安全包装
# ===================================================================


def _safe_grid_fit(
    base_model,
    param_grid,
    X,
    y,
    scoring,
    cv,
    n_jobs,
    verbose,
    is_clustering=False,
    progress_callback: Callable = None,
    random_state: int = None,
):
    """安全执行 GridSearchCV.fit() — 内置多重回退策略 + 进度回调

    Args:
        is_clustering: 如果是聚类, 跳过 CV (使用手动全量搜索), 直接返回结果 dict
        progress_callback: callable(step, total, params, score) — 每个CV fold评估后调用
    """
    from sklearn.model_selection import GridSearchCV, ParameterGrid
    from sklearn.utils.multiclass import type_of_target

    n_samples = len(X)

    # ── 聚类路径: 不使用 GridSearchCV ──
    # CV 对无监督学习没有意义: train fold 上找到的簇无法映射到 test fold
    # 改用手动参数遍历: 全量数据 fit → 全量数据 silhouette_score
    if is_clustering:
        logger.info(f'聚类任务检测到: 使用手动参数搜索 (无CV), {len(list(ParameterGrid(param_grid)))} 个参数组合')
        result = _manual_clustering_search(
            base_model,
            param_grid,
            X,
            progress_callback=progress_callback,
            random_state=random_state,
        )
        # 返回 (result_dict, effective_cv=1) — 聚类无CV
        return result, 1

    # 计算有效CV折数
    effective_cv = max(2, min(cv, n_samples // 3, 10))
    if n_samples < effective_cv * 2:
        effective_cv = max(2, n_samples // 2)
    effective_cv = max(2, effective_cv)

    # ── 防御: 检查每个类别是否有足够样本支持分层CV ──
    # 如果某类别样本数 < effective_cv，StratifiedKFold 会报错:
    # "n_splits=N cannot be greater than the number of members in each class."
    if not is_clustering and y is not None:
        try:
            from collections import Counter

            y_flat = np.asarray(y).ravel()
            class_counts = Counter(y_flat)
            min_class_count = min(class_counts.values()) if class_counts else n_samples
            if effective_cv > min_class_count:
                old_cv = effective_cv
                effective_cv = max(2, min_class_count)
                logger.warning(
                    f'CV 折数自动调整: {old_cv} → {effective_cv} '
                    f'(最小类别样本数={min_class_count}, 原CV折数超过此数会导致分层采样失败)'
                )
                if effective_cv < 2:
                    effective_cv = 2
        except Exception:
            pass

    # 计算总评估步数 (= 参数组合数 * CV折数)
    param_list = list(ParameterGrid(param_grid))
    n_combinations = len(param_list)
    total_steps = n_combinations * effective_cv

    # ---- 最终防御: 验证 type_of_target (仅监督学习) ----
    classification_scorers = {
        'accuracy',
        'precision',
        'recall',
        'f1',
        'roc_auc',
        'average_precision',
        'precision_macro',
        'recall_macro',
        'f1_macro',
        'precision_weighted',
        'recall_weighted',
        'f1_weighted',
    }
    regression_scorers = {
        'neg_mean_squared_error',
        'neg_mean_absolute_error',
        'r2',
        'explained_variance',
        'neg_root_mean_squared_error',
        'max_error',
        'neg_median_absolute_error',
    }

    if not is_clustering and y is not None:
        yt = type_of_target(y)
        if yt == 'continuous' and scoring in classification_scorers:
            raise ValueError(
                f'REFUSED: classification scorer "{scoring}" '
                f'with continuous target (type_of_target={yt}). '
                f'Use regression instead.'
            )
        if yt in ('binary', 'multiclass') and scoring in regression_scorers:
            raise ValueError(
                f'REFUSED: regression scorer "{scoring}" '
                f'with classification target (type_of_target={yt}). '
                f'Use classification instead.'
            )

    # ---- 选择评分器 ----
    actual_scoring = scoring
    fit_y = y

    # ---- 进度包装: n_jobs=1 + 委托式评分器 (v6 修复) ----
    if progress_callback is not None:
        effective_n_jobs = 1
        scoring_for_grid = _make_progress_scorer(actual_scoring, total_steps, progress_callback)
    else:
        effective_n_jobs = n_jobs
        scoring_for_grid = actual_scoring

    # 根据 scoring 方向确定合法的 error_score:
    #   - 正向指标 (accuracy, f1, silhouette): 0 = 最低分
    #   - 负向指标 (neg_*): float('-inf') 避免失败组合得最高分
    error_score = _error_score_for_scoring(scoring)
    grid = GridSearchCV(
        base_model,
        param_grid,
        scoring=scoring_for_grid,
        cv=effective_cv,
        n_jobs=effective_n_jobs,
        verbose=verbose,
        error_score=error_score,
    )

    grid.fit(X, fit_y)
    return grid, effective_cv


def _build_result(grid_or_dict, search_time, scoring, task_type, effective_cv, is_mlp):
    """从拟合好的 GridSearchCV 或手动搜索结果提取结构化结果

    Args:
        grid_or_dict: GridSearchCV 对象 或 手动搜索返回的 dict (带 manual_search=True 标记)
    """
    # ── 手动搜索结果 (聚类全量搜索 或 错误dict) — 直接添加元数据返回 ──
    if isinstance(grid_or_dict, dict):
        result = dict(grid_or_dict)
        result['search_time'] = search_time
        result['scoring'] = scoring
        result['task_type'] = task_type
        result['cv_folds'] = effective_cv
        return result

    # ── GridSearchCV 结果 — 提取 cv_results_ ──
    grid = grid_or_dict
    cv_results = []
    keys = list(grid.cv_results_.keys())
    param_keys = [k for k in keys if k.startswith('param_')]

    for i in range(len(grid.cv_results_['mean_test_score'])):
        param_dict = {k.replace('param_', ''): grid.cv_results_[k][i] for k in param_keys}
        cv_results.append(
            {
                'params': param_dict,
                'mean_score': round(float(grid.cv_results_['mean_test_score'][i]), 4),
                'std_score': round(float(grid.cv_results_['std_test_score'][i]), 4),
                'rank': int(grid.cv_results_['rank_test_score'][i]),
            }
        )
    cv_results.sort(key=lambda x: x['rank'])

    best_params = dict(grid.best_params_)
    if is_mlp:
        best_params = _map_mlp_params(best_params)
        for r in cv_results:
            r['params'] = _map_mlp_params(r['params'])

    return {
        'success': True,
        'best_params': best_params,
        'best_score': round(float(grid.best_score_), 4),
        'cv_results': cv_results[:20],
        'search_time': search_time,
        'n_combinations': len(grid.cv_results_['mean_test_score']),
        'scoring': scoring,
        'task_type': task_type,
        'cv_folds': effective_cv,
    }


def _map_mlp_params(params: dict) -> dict:
    """sklearn MLP 参数名 → PyTorch MLP 参数名"""
    mapping = {
        'hidden_layer_sizes': 'hidden_layers',
        'learning_rate_init': 'learning_rate',
        'alpha': 'weight_decay',
        'validation_fraction': 'val_size',
    }
    mapped = {}
    for k, v in params.items():
        new_key = mapping.get(k, k)
        if k == 'hidden_layer_sizes' and isinstance(v, tuple):
            mapped[new_key] = list(v)
        else:
            mapped[new_key] = v
    return mapped


# ===================================================================
# HyperparameterTuningService (v4 重写)
# ===================================================================


class HyperparameterTuningService:
    """超参数自动调优服务 — 支持分类/回归/聚类 + GridSearchCV/RandomizedSearchCV"""

    @staticmethod
    def get_search_space(algorithm: str, framework: str = 'sklearn') -> dict:
        """获取指定算法的搜索空间 (框架感知: sklearn MLP vs PyTorch MLP)"""
        if algorithm == 'mlp':
            return PYTORCH_SEARCH_SPACE if framework == 'pytorch' else MLP_SEARCH_SPACE
        return SEARCH_SPACES.get(algorithm, {})

    # ------------------------------------------------------------------
    # GridSearchCV
    # ------------------------------------------------------------------

    @staticmethod
    def run_grid_search(
        dataset: Dataset,
        algorithm: str,
        task_type: str,
        target_column: str,
        scoring: str = 'accuracy',
        cv: int = 5,
        n_jobs: int = 2,
        verbose: int = 1,
        progress_callback: Callable = None,
        random_state: int = None,
    ) -> dict:
        """运行 GridSearchCV 超参数搜索

        策略: 最多尝试 3 次 (分类 → 回归 → 聚类),
        每次自动修正 task_type/scoring/algorithm 的不一致.

        Args:
            dataset: 数据集对象
            algorithm: 算法名 (random_forest, kmeans, svm, ...)
            task_type: classification | regression | clustering
            target_column: 目标列名 (聚类可空)
            scoring: 初始评分指标 (会被防御层修正)
            cv: 交叉验证折数
            n_jobs: 并行线程数
            verbose: 详细程度
        """
        is_mlp = algorithm == 'mlp'
        is_clustering = task_type == 'clustering' or algorithm in _CLUSTERING_ALGOS

        # ---- 确定搜索空间 ----
        if is_mlp:
            param_grid = MLP_SEARCH_SPACE
        elif is_clustering and task_type != 'clustering':
            # 聚类算法但声明了非聚类task_type → 自动修正
            logger.warning(f'算法 "{algorithm}" 是聚类算法, 自动将 task_type 从 "{task_type}" 修正为 "clustering"')
            task_type = 'clustering'
            is_clustering = True
            param_grid = SEARCH_SPACES.get(algorithm, {})
        else:
            param_grid = SEARCH_SPACES.get(algorithm, {})

        if not param_grid:
            # 尝试从映射表查找对等算法
            alt_algo = None
            if algorithm in _CLS_TO_REG:
                alt_algo = _CLS_TO_REG[algorithm]
            elif algorithm in _REG_TO_CLS:
                alt_algo = _REG_TO_CLS[algorithm]
            if alt_algo and alt_algo in SEARCH_SPACES:
                param_grid = SEARCH_SPACES[alt_algo]
                logger.info(f'算法 "{algorithm}" 无独立搜索空间, 使用 "{alt_algo}" 的搜索空间')

        if not param_grid:
            supported = sorted(SEARCH_SPACES.keys())
            return {
                'success': False,
                'error': (
                    f'算法 "{algorithm}" 无预定义搜索空间。'
                    f'支持的算法: {supported}。'
                    f'如需要, 可在 SEARCH_SPACES 中添加 "{algorithm}" 的搜索空间。'
                ),
            }

        # ---- 构建备选方案 ----
        attempts = [(task_type, scoring, algorithm)]
        if not is_clustering:
            if task_type == 'classification':
                reg_algo = _CLS_TO_REG.get(algorithm, algorithm)
                attempts.append(('regression', 'neg_mean_squared_error', reg_algo))
            else:
                cls_algo = _REG_TO_CLS.get(algorithm, algorithm)
                attempts.append(('classification', 'accuracy', cls_algo))

        last_error = None

        for attempt_idx, (try_task, _try_scoring, try_algo) in enumerate(attempts):
            try:
                # ---- 加载数据 ----
                df = _load_dataset_file(dataset)
                if df is None:
                    return {'success': False, 'error': '无法加载数据集文件。请检查文件路径和格式。'}

                # ---- 准备数据 (带防御验证) ----
                X, y, resolved_task, resolved_scoring = _prepare_data_robust(df, target_column, try_task)
                final_task = resolved_task
                final_scoring = resolved_scoring

                # _prepare_data_robust 可能修正 task_type → 同步 algorithm
                if final_task != try_task:
                    if final_task == 'regression':
                        try_algo = _CLS_TO_REG.get(try_algo, try_algo)
                    elif final_task == 'classification':
                        try_algo = _REG_TO_CLS.get(try_algo, try_algo)
                    elif final_task == 'clustering':
                        try_algo = try_algo  # 保持原算法

                # 再次检查 search space (可能在修正后变化)
                final_is_clustering = final_task == 'clustering'
                final_param_grid = SEARCH_SPACES.get(try_algo, param_grid) if final_is_clustering else param_grid

                # ---- 验证样本数 ----
                n_samples = len(X)
                _validate_min_samples(n_samples, cv)

                # ---- 创建模型 ----
                base_model = _create_model(try_algo, final_task, is_mlp, random_state=random_state)

                # ---- 安全拟合 ----
                start_time = time.time()
                grid, effective_cv = _safe_grid_fit(
                    base_model,
                    final_param_grid,
                    X,
                    y,
                    final_scoring,
                    cv,
                    n_jobs,
                    verbose,
                    is_clustering=final_is_clustering,
                    progress_callback=progress_callback,
                    random_state=random_state,
                )
                search_time = round(time.time() - start_time, 2)

                # ---- 构建结果 ----
                result = _build_result(grid, search_time, final_scoring, final_task, effective_cv, is_mlp)
                logger.info(
                    f'GridSearchCV 完成 (attempt {attempt_idx + 1}): '
                    f'{try_algo}/{final_task}, '
                    f'best={result["best_score"]:.4f}, '
                    f'{search_time}s, '
                    f'{result.get("n_combinations", "?")}组合'
                )
                return result

            except Exception as e:
                last_error = str(e)
                err_lower = last_error.lower()

                # 判断是否应该尝试下一个备选方案
                should_retry = any(
                    kw in err_lower
                    for kw in [
                        'continuous',
                        'multiclass',
                        'mixed',
                        'classification metrics',
                        'refused',
                        '不支持',
                        'not supported',
                    ]
                )

                # 聚类算法不需要尝试回归备选 (本身就是无监督)
                if is_clustering:
                    should_retry = False

                if should_retry and attempt_idx < len(attempts) - 1:
                    logger.warning(
                        f'GridSearchCV attempt {attempt_idx + 1} 失败: {last_error}. '
                        f'自动重试备选方案 ({attempts[attempt_idx + 1][0]})...'
                    )
                    continue

                logger.error(f'GridSearchCV 最终失败: {last_error}', exc_info=True)
                return {'success': False, 'error': f'搜索失败: {last_error}'}

        return {'success': False, 'error': f'搜索失败 (所有方案已用尽): {last_error}'}

    # ------------------------------------------------------------------
    # RandomizedSearchCV
    # ------------------------------------------------------------------

    @staticmethod
    def run_random_search(
        dataset: Dataset,
        algorithm: str,
        task_type: str,
        target_column: str,
        n_iter: int = 30,
        scoring: str = 'accuracy',
        cv: int = 5,
        n_jobs: int = 2,
        progress_callback: Callable = None,
        random_state: int = None,
    ) -> dict:
        """运行 RandomizedSearchCV — 与 GridSearchCV 相同的多层防御策略"""
        from scipy.stats import loguniform, randint
        from sklearn.model_selection import RandomizedSearchCV

        is_mlp = algorithm == 'mlp'
        is_clustering = task_type == 'clustering' or algorithm in _CLUSTERING_ALGOS

        # ---- 搜索空间 ----
        if is_mlp:
            param_distributions = MLP_SEARCH_SPACE
        elif is_clustering and task_type != 'clustering':
            task_type = 'clustering'
            is_clustering = True
            param_distributions = SEARCH_SPACES.get(algorithm, {})
        else:
            param_distributions = SEARCH_SPACES.get(algorithm, {})

        if not param_distributions:
            supported = sorted(SEARCH_SPACES.keys())
            return {
                'success': False,
                'error': (f'算法 "{algorithm}" 无预定义搜索空间。支持的算法: {supported}。'),
            }

        # ---- 转换为 scipy 分布 (仅对数值型参数) ----
        distributions = {}
        for param, values in param_distributions.items():
            if not isinstance(values, list) or len(values) <= 3:
                distributions[param] = values
                continue

            # 过滤 None
            clean_vals = [v for v in values if v is not None]
            if not clean_vals:
                distributions[param] = values
                continue

            int_vals = [v for v in clean_vals if isinstance(v, int)]
            float_vals = [v for v in clean_vals if isinstance(v, float)]

            if len(int_vals) == len(clean_vals):
                distributions[param] = randint(min(int_vals), max(int_vals) + 1)
            elif len(float_vals) == len(clean_vals):
                min_f, max_f = min(float_vals), max(float_vals)
                if min_f <= 0:
                    distributions[param] = values  # loguniform 不支持非正
                else:
                    distributions[param] = loguniform(min_f, max_f)
            else:
                distributions[param] = values

        # ---- 备选方案 ----
        attempts = [(task_type, scoring, algorithm)]
        if not is_clustering:
            if task_type == 'classification':
                reg_algo = _CLS_TO_REG.get(algorithm, algorithm)
                attempts.append(('regression', 'neg_mean_squared_error', reg_algo))
            else:
                cls_algo = _REG_TO_CLS.get(algorithm, algorithm)
                attempts.append(('classification', 'accuracy', cls_algo))

        last_error = None

        for attempt_idx, (try_task, _try_scoring, try_algo) in enumerate(attempts):
            try:
                df = _load_dataset_file(dataset)
                if df is None:
                    return {'success': False, 'error': '无法加载数据集文件。'}

                X, y, resolved_task, resolved_scoring = _prepare_data_robust(df, target_column, try_task)
                final_task = resolved_task
                final_scoring = resolved_scoring
                final_is_clustering = final_task == 'clustering'

                if final_task != try_task:
                    if final_task == 'regression':
                        try_algo = _CLS_TO_REG.get(try_algo, try_algo)
                    elif final_task == 'classification':
                        try_algo = _REG_TO_CLS.get(try_algo, try_algo)

                n_samples = len(X)
                _validate_min_samples(n_samples, cv)

                if final_is_clustering:
                    effective_cv = max(2, min(cv, n_samples // 5, 5))
                else:
                    effective_cv = max(2, min(cv, n_samples // 3, 10))

                if n_samples < effective_cv * 2:
                    effective_cv = max(2, n_samples // 2)

                # ── 防御: 检查每个类别是否有足够样本支持分层CV ──
                if not final_is_clustering and y is not None:
                    try:
                        from collections import Counter

                        y_flat = np.asarray(y).ravel()
                        class_counts = Counter(y_flat)
                        min_class_count = min(class_counts.values()) if class_counts else n_samples
                        if effective_cv > min_class_count:
                            old_cv = effective_cv
                            effective_cv = max(2, min_class_count)
                            logger.warning(
                                f'RandomSearch CV 折数自动调整: {old_cv} → {effective_cv} '
                                f'(最小类别样本数={min_class_count})'
                            )
                            if effective_cv < 2:
                                effective_cv = 2
                    except Exception:
                        pass

                # 最终防御: 验证 scoring 与 y type 一致
                from sklearn.utils.multiclass import type_of_target

                if not final_is_clustering and y is not None:
                    yt = type_of_target(y)
                    if yt == 'continuous' and final_scoring == 'accuracy':
                        raise ValueError('REFUSED: accuracy scorer with continuous target')
                    if yt in ('binary', 'multiclass') and final_scoring == 'neg_mean_squared_error':
                        raise ValueError('REFUSED: regression scorer with classification target')

                base_model = _create_model(try_algo, final_task, is_mlp, random_state=random_state)

                start_time = time.time()

                # ── 聚类路径: 使用手动全量搜索 (CV对无监督无意义) ──
                if final_is_clustering:
                    param_grid = SEARCH_SPACES.get(try_algo, param_distributions)
                    if not param_grid:
                        return {'success': False, 'error': f'聚类算法 "{try_algo}" 无搜索空间'}
                    result = _manual_clustering_search(
                        base_model,
                        param_grid,
                        X,
                        progress_callback=progress_callback,
                        max_combos=n_iter,
                        random_state=random_state,
                    )
                    search_time = round(time.time() - start_time, 2)
                    result = _build_result(result, search_time, final_scoring, final_task, effective_cv, is_mlp)
                else:
                    # ── 监督学习路径 ──
                    actual_scoring = _clustering_scorer if final_is_clustering else final_scoring

                    # 进度回调 → n_jobs=1 + 委托式评分器
                    if progress_callback is not None:
                        from sklearn.model_selection import ParameterGrid

                        param_list = list(ParameterGrid(param_distributions))
                        len(param_list) if param_list else 0
                        total_steps_r = min(n_iter, 100) * effective_cv
                        scoring_for_rnd = _make_progress_scorer(actual_scoring, total_steps_r, progress_callback)
                        effective_n_jobs_rnd = 1
                    else:
                        scoring_for_rnd = actual_scoring
                        effective_n_jobs_rnd = n_jobs

                    search = RandomizedSearchCV(
                        base_model,
                        distributions,
                        n_iter=min(n_iter, 100),
                        scoring=scoring_for_rnd,
                        cv=effective_cv,
                        n_jobs=effective_n_jobs_rnd,
                        random_state=random_state,
                        error_score=_error_score_for_scoring(final_scoring),
                    )
                    search.fit(X, y)
                    search_time = round(time.time() - start_time, 2)

                    result = _build_result(search, search_time, final_scoring, final_task, effective_cv, is_mlp)
                best_log_score = result.get('best_score', 0) if final_is_clustering else search.best_score_
                logger.info(
                    f'RandomizedSearchCV 完成 (attempt {attempt_idx + 1}): '
                    f'{try_algo}/{final_task}, best={best_log_score:.4f}'
                )
                return result

            except Exception as e:
                last_error = str(e)
                err_lower = last_error.lower()
                should_retry = any(
                    kw in err_lower
                    for kw in [
                        'continuous',
                        'multiclass',
                        'mixed',
                        'classification metrics',
                        'refused',
                        '不支持',
                    ]
                )

                if is_clustering:
                    should_retry = False

                if should_retry and attempt_idx < len(attempts) - 1:
                    logger.warning(f'RandomizedSearchCV attempt {attempt_idx + 1} 失败: {last_error}. 自动重试...')
                    continue

                logger.error(f'RandomizedSearchCV 最终失败: {last_error}', exc_info=True)
                return {'success': False, 'error': f'搜索失败: {last_error}'}

        return {'success': False, 'error': f'搜索失败 (所有方案已用尽): {last_error}'}

    # ------------------------------------------------------------------
    # 一站式调优+训练
    # ------------------------------------------------------------------

    @staticmethod
    def create_tuned_training(
        user: User,
        dataset: Dataset,
        algorithm: str,
        task_type: str,
        target_column: str,
        tuning_method: str = 'grid',
        n_iter: int = 30,
        cv: int = 5,
        epochs: int = 0,
        random_state: int = None,
    ) -> tuple[TrainingJob | None, dict | None, str | None]:
        """运行超参数搜索并创建使用最佳参数的训练任务

        当 algorithm='auto' 时, 自动遍历该任务类型的所有适用算法,
        对每种算法执行快速 RandomSearch, 选出全局最优。
        """
        from app.services.training_service import TrainingService

        # ── AutoML 模式: 遍历所有适用算法 ──
        if algorithm == 'auto':
            algo_list = AUTO_ALGORITHMS.get(task_type, [])
            if not algo_list:
                return None, None, f'AutoML 不支持任务类型 "{task_type}"。可选: classification, regression, clustering'

            best_overall_score = None
            best_overall_algo = None
            best_overall_params = None
            best_overall_framework = 'sklearn'
            best_overall_task_type = task_type
            algo_results = []

            QUICK_N_ITER = min(n_iter, _get_tuning_config('AUTO_ML_QUICK_N_ITER', 15))

            for algo_info in algo_list:
                algo_name = algo_info['algo']
                algo_label = algo_info['label']
                algo_fw = algo_info['framework']

                logger.info(f'AutoML: 正在搜索 {algo_label} ({algo_name})...')
                try:
                    result = HyperparameterTuningService.run_random_search(
                        dataset=dataset,
                        algorithm=algo_name,
                        task_type=task_type,
                        target_column=target_column,
                        n_iter=QUICK_N_ITER,
                        scoring='accuracy'
                        if task_type == 'classification'
                        else 'neg_mean_squared_error'
                        if task_type == 'regression'
                        else 'silhouette',
                        cv=cv,
                        n_jobs=2,
                        random_state=random_state,
                    )
                except Exception as e:
                    logger.warning(f'AutoML: {algo_label} ({algo_name}) 搜索异常: {e}', exc_info=True)
                    continue

                if not result.get('success'):
                    logger.warning(
                        f'AutoML: {algo_label} ({algo_name}) 搜索返回失败: {result.get("error", "未知错误")}'
                    )

                if result.get('success'):
                    score = result['best_score']
                    algo_results.append(
                        {
                            'algo': algo_name,
                            'label': algo_label,
                            'framework': algo_fw,
                            'best_score': score,
                            'best_params': result['best_params'],
                        }
                    )
                    if best_overall_score is None or score > best_overall_score:
                        best_overall_score = score
                        best_overall_algo = algo_name
                        best_overall_params = result['best_params']
                        best_overall_framework = algo_fw
                        best_overall_task_type = result.get('task_type', task_type)

            if best_overall_algo is None:
                return None, None, 'AutoML: 所有算法均搜索失败，无法找到有效模型。'

            # 排序算法排名
            algo_results.sort(key=lambda x: x['best_score'], reverse=True)

            # 使用最优算法创建训练任务
            algorithm = best_overall_algo
            best_params = best_overall_params
            effective_task_type = best_overall_task_type
            framework = best_overall_framework
            is_mlp = algorithm == 'mlp'

            # 构建 tuning_result
            tuning_result = {
                'success': True,
                'best_params': best_params,
                'best_score': best_overall_score,
                'task_type': effective_task_type,
                'auto_mode': True,
                'algo_results': algo_results,
                'total_algos_tried': len(algo_results),
                'cv_results': algo_results[:5],
            }

            actual_epochs = epochs if epochs > 0 else (20 if is_mlp else 1)

            hyperparams = {
                'task_type': effective_task_type,
                'algorithm': algorithm,
                'target_column': target_column,
                'test_size': 0.2,
                'algorithm_params': best_params,
                'tuned': True,
                'tuning_method': 'auto',
                'best_cv_score': best_overall_score,
                'tuning_cv_folds': cv,
            }
            if is_mlp:
                for k in ('hidden_layers', 'learning_rate', 'batch_size', 'weight_decay', 'val_size'):
                    if k in best_params:
                        hyperparams[k] = best_params[k]

            job_name = f'AutoML-{algorithm}-Tuned-{localnow().strftime("%H%M")}'

            job, error = TrainingService.create_job(
                user=user,
                name=job_name,
                dataset_id=dataset.id,
                description=f'AutoML 自动调优训练 — 从 {len(algo_results)} 种算法中选出最优: '
                f'{algorithm} (CV分数: {best_overall_score:.4f})',
                framework=framework,
                total_epochs=actual_epochs,
                hyperparameters=hyperparams,
                ml_task_type=effective_task_type,
                algorithm=algorithm,
                target_column=target_column,
            )

            if error:
                return None, tuning_result, error

            if job.model:
                hp = job.model.hyperparameters_dict
                hp['tuning_result'] = {
                    'best_params': best_params,
                    'best_cv_score': best_overall_score,
                    'cv_results_top5': algo_results[:5],
                    'auto_mode': True,
                    'all_algo_results': algo_results,
                }
                job.model.set_hyperparameters(hp)
                db.session.commit()

            logger.info(
                f'AutoML 调优完成: 最优算法={algorithm}, 分数={best_overall_score:.4f}, 共尝试{len(algo_results)}种算法'
            )
            return job, tuning_result, None

        # ── 单算法调优 (原有逻辑) ──
        if tuning_method == 'grid':
            tuning_result = HyperparameterTuningService.run_grid_search(
                dataset,
                algorithm,
                task_type,
                target_column,
                cv=cv,
                random_state=random_state,
            )
        else:
            tuning_result = HyperparameterTuningService.run_random_search(
                dataset,
                algorithm,
                task_type,
                target_column,
                n_iter=n_iter,
                cv=cv,
                random_state=random_state,
            )

        if not tuning_result.get('success'):
            return None, tuning_result, tuning_result.get('error')

        best_params = tuning_result['best_params']
        effective_task_type = tuning_result.get('task_type', task_type)

        is_mlp = algorithm == 'mlp'
        framework = 'pytorch' if is_mlp else 'sklearn'
        actual_epochs = epochs if epochs > 0 else (20 if is_mlp else 1)

        hyperparams = {
            'task_type': effective_task_type,
            'algorithm': algorithm,
            'target_column': target_column,
            'test_size': 0.2,
            'algorithm_params': best_params,
            'tuned': True,
            'tuning_method': tuning_method,
            'best_cv_score': tuning_result['best_score'],
            'tuning_cv_folds': cv,
        }
        if is_mlp:
            for k in ('hidden_layers', 'learning_rate', 'batch_size', 'weight_decay', 'val_size'):
                if k in best_params:
                    hyperparams[k] = best_params[k]

        job_name = f'{algorithm}-Tuned-{localnow().strftime("%H%M")}'

        job, error = TrainingService.create_job(
            user=user,
            name=job_name,
            dataset_id=dataset.id,
            description=f'超参数调优训练 ({tuning_method}), 最佳CV分数: {tuning_result["best_score"]:.4f}',
            framework=framework,
            total_epochs=actual_epochs,
            hyperparameters=hyperparams,
            ml_task_type=effective_task_type,
            algorithm=algorithm,
            target_column=target_column,
        )

        if error:
            return None, tuning_result, error

        if job.model:
            hp = job.model.hyperparameters_dict
            hp['tuning_result'] = {
                'best_params': best_params,
                'best_cv_score': tuning_result['best_score'],
                'cv_results_top5': tuning_result.get('cv_results', [])[:5],
                'search_time': tuning_result.get('search_time'),
            }
            job.model.set_hyperparameters(hp)
            db.session.commit()

        logger.info(f'调优训练任务创建: {job_name}, 最佳参数: {best_params}')
        return job, tuning_result, None

    # ------------------------------------------------------------------
    # 异步 GridSearchCV (v5 新增) — 后台线程 + 实时进度推送
    # ------------------------------------------------------------------

    @staticmethod
    def run_grid_search_async(
        dataset: Dataset,
        algorithm: str,
        task_type: str,
        target_column: str,
        scoring: str = 'accuracy',
        cv: int = 5,
        n_jobs: int = 2,
        random_state: int = None,
    ) -> str:
        """在后台线程启动 GridSearchCV, 立即返回 tuning_id

        Returns:
            tuning_id (str): UUID, 用于 SSE 端点订阅进度
        """
        tuning_id = str(uuid.uuid4())[:8]
        tracker = get_tuning_tracker()

        # 预计算总步数
        param_grid = MLP_SEARCH_SPACE if algorithm == 'mlp' else SEARCH_SPACES.get(algorithm, {})
        if not param_grid:
            # 尝试映射
            alt = _CLS_TO_REG.get(algorithm) or _REG_TO_CLS.get(algorithm)
            if alt and alt in SEARCH_SPACES:
                param_grid = SEARCH_SPACES[alt]

        from sklearn.model_selection import ParameterGrid

        n_combos = len(list(ParameterGrid(param_grid))) if param_grid else 0
        # 聚类不需要CV — 总步数 = 参数组合数 (不是 n_combos * cv)
        if task_type == 'clustering' or algorithm in _CLUSTERING_ALGOS:
            total_steps = n_combos
        else:
            total_steps = n_combos * max(2, min(cv, 5))

        tracker.init(tuning_id, total_steps, algorithm, task_type, 'grid')
        tracker.add_log(tuning_id, f'启动 GridSearchCV: {algorithm}/{task_type}, 参数组合={n_combos}, CV={cv}')

        def _bg_run():
            try:

                def progress_cb(step, total, params, score):
                    tracker.update(tuning_id, step, params, score, total=total)

                result = HyperparameterTuningService.run_grid_search(
                    dataset,
                    algorithm,
                    task_type,
                    target_column,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=n_jobs,
                    progress_callback=progress_cb,
                    random_state=random_state,
                )
                if result.get('success'):
                    tracker.add_log(
                        tuning_id, f'最佳分数: {result["best_score"]:.4f}, 最佳参数: {result["best_params"]}'
                    )
                    tracker.complete(tuning_id, result)
                else:
                    tracker.add_log(tuning_id, f'搜索失败: {result.get("error")}')
                    tracker.fail(tuning_id, result.get('error', '未知错误'))
            except Exception as e:
                tracker.add_log(tuning_id, f'异常: {str(e)}')
                tracker.fail(tuning_id, str(e))
            finally:
                db.session.remove()

        thread = threading.Thread(target=_bg_run, daemon=True, name=f'tuning-{tuning_id}')
        thread.start()
        logger.info(f'异步 GridSearchCV 已启动: tuning_id={tuning_id}')
        return tuning_id

    @staticmethod
    def run_random_search_async(
        dataset: Dataset,
        algorithm: str,
        task_type: str,
        target_column: str,
        n_iter: int = 30,
        scoring: str = 'accuracy',
        cv: int = 5,
        n_jobs: int = 2,
        random_state: int = None,
    ) -> str:
        """在后台线程启动 RandomizedSearchCV, 立即返回 tuning_id"""
        tuning_id = str(uuid.uuid4())[:8]
        tracker = get_tuning_tracker()

        param_grid = MLP_SEARCH_SPACE if algorithm == 'mlp' else SEARCH_SPACES.get(algorithm, {})
        if not param_grid:
            alt = _CLS_TO_REG.get(algorithm) or _REG_TO_CLS.get(algorithm)
            if alt and alt in SEARCH_SPACES:
                param_grid = SEARCH_SPACES[alt]

        effective_n = min(n_iter, 100)
        if task_type == 'clustering' or algorithm in _CLUSTERING_ALGOS:
            total_steps = effective_n
        else:
            total_steps = effective_n * max(2, min(cv, 5))

        tracker.init(tuning_id, total_steps, algorithm, task_type, 'random')
        tracker.add_log(tuning_id, f'启动 RandomizedSearchCV: {algorithm}/{task_type}, n_iter={effective_n}')

        def _bg_run():
            try:

                def progress_cb(step, total, params, score):
                    tracker.update(tuning_id, step, params, score, total=total)

                result = HyperparameterTuningService.run_random_search(
                    dataset,
                    algorithm,
                    task_type,
                    target_column,
                    n_iter=n_iter,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    progress_callback=progress_cb,
                )
                if result.get('success'):
                    tracker.add_log(tuning_id, f'最佳分数: {result["best_score"]:.4f}')
                    tracker.complete(tuning_id, result)
                else:
                    tracker.fail(tuning_id, result.get('error', '未知错误'))
            except Exception as e:
                tracker.fail(tuning_id, str(e))
            finally:
                db.session.remove()

        thread = threading.Thread(target=_bg_run, daemon=True, name=f'tuning-{tuning_id}')
        thread.start()
        logger.info(f'异步 RandomizedSearchCV 已启动: tuning_id={tuning_id}')
        return tuning_id

    # ------------------------------------------------------------------
    # AutoML 自动算法选择 (v8 新增) — 遍历所有算法找最优
    # ------------------------------------------------------------------

    @staticmethod
    def run_auto_tuning_async(
        dataset: Dataset,
        task_type: str,
        target_column: str = None,
        cv: int = 3,
        n_jobs: int = 2,
        random_state: int = None,
    ) -> str:
        """AutoML 模式: 自动遍历任务类型的所有适用算法, 找到最优组合

        对每种算法执行快速 RandomSearch, 实时展示:
          - 当前正在搜索的算法名
          - 每种算法的 best score
          - 全局 best score + best algo + best params

        Returns:
            tuning_id (str): SSE 订阅用 UUID
        """
        import time as _time

        tuning_id = str(uuid.uuid4())[:8]
        tracker = get_tuning_tracker()

        # 获取该 task_type 的所有适用算法
        algo_list = AUTO_ALGORITHMS.get(task_type, [])
        if not algo_list:
            tracker.init(tuning_id, 0, 'auto', task_type, 'auto')
            tracker.fail(tuning_id, f'任务类型 "{task_type}" 不支持自动模式')
            return tuning_id

        # 计算总步数: 每种算法的快速搜索迭代数
        QUICK_N_ITER = _get_tuning_config('AUTO_ML_QUICK_N_ITER', 15)
        total_steps = len(algo_list) * QUICK_N_ITER
        tracker.init(tuning_id, total_steps, 'auto', task_type, 'auto')
        tracker.add_log(tuning_id, f'🤖 AutoML 启动: {task_type}, {len(algo_list)} 种算法, 每种 {QUICK_N_ITER} 组参数')

        def _bg_run():
            overall_best_score = None
            overall_best_algo = None
            overall_best_params = None
            overall_best_framework = 'sklearn'
            algo_results = []  # [{algo, label, framework, best_score, best_params, search_time}]
            global_step = 0

            try:
                for idx, algo_info in enumerate(algo_list):
                    algo = algo_info['algo']
                    label = algo_info['label']
                    framework = algo_info['framework']

                    tracker.add_log(tuning_id, f'▶ [{idx + 1}/{len(algo_list)}] {label} ({algo}) 开始搜索...')

                    # 内部 progress 回调 — 更新全局步数 + 当前算法信息
                    def algo_progress_cb(step, total, params, score, _label=label, _algo=algo):
                        nonlocal global_step
                        global_step += 1
                        tracker.update(
                            tuning_id,
                            global_step,
                            params=params,
                            score=score,
                            total=total_steps,
                        )
                        # 在 best_params 中注入当前算法信息, 前端可展示
                        s = tracker.get(tuning_id)
                        if s and s.get('best_params_so_far') is not None:
                            bp = dict(s['best_params_so_far'])
                            bp['_current_algo'] = f'{_label} ({_algo})'
                            with tracker._lock:
                                if tuning_id in tracker._sessions:
                                    tracker._sessions[tuning_id]['best_params_so_far'] = bp

                    # 快速 RandomSearch (每算法)
                    t0 = _time.time()
                    try:
                        param_grid = SEARCH_SPACES.get(algo, {})
                        if not param_grid:
                            # 尝试映射
                            alt = _CLS_TO_REG.get(algo) or _REG_TO_CLS.get(algo)
                            if alt and alt in SEARCH_SPACES:
                                param_grid = SEARCH_SPACES[alt]

                        if param_grid:
                            result = HyperparameterTuningService.run_random_search(
                                dataset=dataset,
                                algorithm=algo,
                                task_type=task_type,
                                target_column=target_column,
                                n_iter=QUICK_N_ITER,
                                scoring='accuracy'
                                if task_type == 'classification'
                                else 'neg_mean_squared_error'
                                if task_type == 'regression'
                                else 'silhouette',
                                cv=cv,
                                n_jobs=n_jobs,
                                progress_callback=algo_progress_cb,
                                random_state=random_state,
                            )
                        else:
                            result = {'success': False, 'error': '无搜索空间'}
                    except Exception as e:
                        result = {'success': False, 'error': str(e)}

                    search_t = round(_time.time() - t0, 1)

                    if result.get('success'):
                        score = result['best_score']
                        params = result['best_params']
                        algo_results.append(
                            {
                                'algo': algo,
                                'label': label,
                                'framework': framework,
                                'best_score': score,
                                'best_params': params,
                                'search_time': search_t,
                            }
                        )

                        tracker.add_log(tuning_id, f'  ✓ {label}: best={score:.4f} ({search_t}s)')

                        # 更新全局最优
                        if overall_best_score is None or score > overall_best_score:
                            overall_best_score = score
                            overall_best_algo = algo
                            overall_best_params = params
                            overall_best_framework = framework
                    else:
                        err_detail = f'{label}: {result.get("error", "未知错误")}'
                        tracker.add_log(tuning_id, f'  ✗ {err_detail}')
                        # 收集错误详情，供最终汇总使用
                        with tracker._lock:
                            if tuning_id in tracker._sessions:
                                errs = tracker._sessions[tuning_id].setdefault('_algo_errors', [])
                                errs.append(err_detail)

                # ── 全部算法搜索完毕, 汇总结果 ──
                algo_results.sort(key=lambda x: x['best_score'], reverse=True)

                if overall_best_algo is None:
                    # 所有算法均失败 — 报告详细错误
                    error_details = tracker._sessions[tuning_id].get('_algo_errors', [])
                    error_msg = '所有 ' + str(len(algo_list)) + ' 种算法均搜索失败。'
                    if error_details:
                        error_msg += ' 错误详情: ' + '; '.join(error_details[:3])
                    tracker.add_log(tuning_id, f'✗ {error_msg}')
                    tracker.fail(tuning_id, error_msg)
                    return

                final_result = {
                    'success': True,
                    'task_type': task_type,
                    'best_algo': overall_best_algo,
                    'best_framework': overall_best_framework,
                    'best_score': overall_best_score,
                    'best_params': overall_best_params or {},
                    'algo_results': algo_results,
                    'total_algos_tried': len(algo_list),
                    'total_search_time': round(_time.time() - tracker._sessions[tuning_id]['started_at'], 1),
                    'auto_mode': True,
                }

                tracker.add_log(tuning_id, f'🏆 最优: {overall_best_algo} score={overall_best_score:.4f}')
                tracker.add_log(
                    tuning_id,
                    '📊 算法排名: ' + ', '.join(f'{r["label"]}({r["best_score"]:.4f})' for r in algo_results[:5]),
                )

                # 注入最终算法的 best_params 给前端展示
                with tracker._lock:
                    if tuning_id in tracker._sessions:
                        tracker._sessions[tuning_id]['best_params_so_far'] = overall_best_params or {}
                        tracker._sessions[tuning_id]['_auto_algo_results'] = algo_results

                tracker.complete(tuning_id, final_result)

            except Exception as e:
                logger.error(f'AutoML 失败: {e}', exc_info=True)
                tracker.add_log(tuning_id, f'异常: {str(e)}')
                tracker.fail(tuning_id, str(e))
            finally:
                db.session.remove()

        thread = threading.Thread(target=_bg_run, daemon=True, name=f'autotune-{tuning_id}')
        thread.start()
        logger.info(f'AutoML 已启动: tuning_id={tuning_id}, {len(algo_list)} algorithms, {task_type}')
        return tuning_id
