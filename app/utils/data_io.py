"""
============================================
共享数据 I/O 工具
统一的数据集文件加载和预处理函数
消除跨 5+ 文件的重复文件读取逻辑
============================================
"""
import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def load_dataframe(file_path: str, file_format: str = None,
                   nrows: int = None) -> Optional[pd.DataFrame]:
    """根据文件格式加载 DataFrame

    支持的格式: csv, xlsx, xls, json, parquet, txt (TSV), npy

    Args:
        file_path: 文件路径
        file_format: 明确指定的格式 (可选, 默认从文件扩展名推断)
        nrows: 最多读取的行数 (可选)

    Returns:
        DataFrame 或 None (加载失败时)
    """
    if not file_path or not os.path.exists(file_path):
        return None

    if file_format is None:
        file_format = os.path.splitext(file_path)[1].lower().lstrip('.')

    fmt = file_format.lower()

    try:
        if fmt == 'csv':
            return pd.read_csv(file_path, nrows=nrows)
        elif fmt in ('xlsx', 'xls'):
            return pd.read_excel(file_path, nrows=nrows)
        elif fmt == 'json':
            return pd.read_json(file_path)
        elif fmt == 'parquet':
            return pd.read_parquet(file_path)
        elif fmt == 'txt':
            return pd.read_csv(file_path, sep='\t', nrows=nrows or 1000)
        elif fmt == 'npy':
            arr = np.load(file_path)
            return pd.DataFrame(arr)
    except Exception:
        pass

    # 回退: 尝试用不同编码加载 CSV
    try:
        return pd.read_csv(file_path, nrows=nrows, encoding='latin-1')
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 目标列类型自动判定 — 基于 sklearn 的 type_of_target (权威判定)
# ---------------------------------------------------------------------------

def detect_target_type(y: np.ndarray) -> str:
    """使用 sklearn 的 type_of_target 判定目标列类型

    Returns:
        'classification' | 'regression' | 'multilabel' | 'unknown'
    """
    from sklearn.utils.multiclass import type_of_target

    # 过滤掉 NaN
    y_clean = y[~pd.isna(y)] if hasattr(y, '__len__') else y

    if len(y_clean) == 0:
        return 'unknown'

    try:
        t = type_of_target(y_clean)
    except Exception:
        return 'unknown'

    # sklearn 的 type_of_target 返回值映射
    if t in ('binary', 'multiclass'):
        return 'classification'
    elif t == 'continuous':
        return 'regression'
    elif t in ('multilabel-indicator', 'multiclass-multioutput'):
        return 'multilabel'
    else:
        return 'unknown'


def _safe_label_encode(y) -> np.ndarray:
    """安全编码: 将任意类型的目标值转换为 0..n-1 的整数标签

    策略:
      1. 强制转为字符串, 保证一致性
      2. LabelEncoder 编码为 0..n_classes-1 的整数
      3. 所有值都会变成 Python int, 不会残留 float
    """
    from sklearn.preprocessing import LabelEncoder

    if isinstance(y, np.ndarray):
        y_series = pd.Series(y)
    else:
        y_series = y

    # 转为字符串 — 处理 float / int / object 混排
    y_str = y_series.astype(str).values

    # LabelEncoder 永远返回 int64 的 numpy 数组
    encoded = LabelEncoder().fit_transform(y_str)

    return encoded.astype(np.int64)


def _safe_float_encode(y) -> np.ndarray:
    """安全转换为 float64 — 用于回归任务"""
    if isinstance(y, np.ndarray):
        arr = y.astype(np.float64)
    elif hasattr(y, 'values'):
        arr = y.values.astype(np.float64)
    else:
        arr = np.array(y, dtype=np.float64)
    return arr


def preprocess_data(X: pd.DataFrame, y: pd.Series, task_type: str = 'classification'
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """通用数据预处理: 缺失值填充 + 分类编码 + 标准化

    关键修复 (GridSearchCV "mixed multiclass and continuous targets"):
      - 分类任务: 强制 LabelEncoder 编码, 无论 y 的 dtype 是 float/int/object
      - 回归任务: 强制 float64 转换
      - NaN 在编码前先行移除

    Args:
        X: 特征 DataFrame
        y: 目标 Series 或 array
        task_type: 'classification' 或 'regression'

    Returns:
        (X_scaled, y_encoded) — 两个 float64 numpy 数组 (y 在分类任务中为 int64)
    """
    from sklearn.preprocessing import StandardScaler

    # ------ 1. 处理 X 的缺失值 ------
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        from sklearn.impute import SimpleImputer
        X[num_cols] = SimpleImputer(strategy='mean').fit_transform(X[num_cols])

    cat_cols = X.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        from sklearn.preprocessing import LabelEncoder
        for col in cat_cols:
            X[col] = X[col].fillna('missing')
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # ------ 2. 移除 y 中的 NaN (必须在编码前做) ------
    if isinstance(y, pd.Series) and y.isna().any():
        nan_count = y.isna().sum()
        valid_idx = y.notna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        import logging
        logging.getLogger('app').warning(
            f'preprocess_data: 目标列包含 {nan_count} 个 NaN, 已移除对应行'
        )

    # ------ 3. 编码目标变量 ------
    if task_type == 'classification':
        # 强制编码 — 所有类型统一处理, 杜绝 float 残留
        y = _safe_label_encode(y)
    else:
        # 回归任务: 保证 float64
        y = _safe_float_encode(y)

    # ------ 4. 标准化 X ------
    X_scaled = StandardScaler().fit_transform(X).astype(np.float64)

    return X_scaled, y
