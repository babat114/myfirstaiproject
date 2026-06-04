"""
============================================
超参数自动调优服务
支持 GridSearchCV 和 RandomizedSearchCV
============================================
"""
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict
from app import db, logger
from app.models.training_job import TrainingJob
from app.models.dataset import Dataset
from app.models.model_record import ModelRecord
from app.models.user import User


# ===================================================================
# 预定义超参数搜索空间
# 每种算法对应一组可搜索的超参数，用于 GridSearchCV / RandomizedSearchCV
# ===================================================================
SEARCH_SPACES = {
    # ---- 分类器搜索空间 ----
    'random_forest': {
        'n_estimators': [50, 100, 200, 300],       # 树的数量 — 越多越稳定但越慢
        'max_depth': [None, 10, 20, 30, 50],        # 最大深度 — None=不限制, 深层树容易过拟合
        'min_samples_split': [2, 5, 10],             # 内部节点最小样本数 — 越大越防过拟合
        'min_samples_leaf': [1, 2, 4],               # 叶节点最小样本数
        'max_features': ['sqrt', 'log2', None],      # 每次分裂考虑的特征比例
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],              # 提升轮数 — 贪心拟合残差
        'learning_rate': [0.01, 0.05, 0.1, 0.2],    # 学习率 — 小学习率需要更多轮, 但泛化更好
        'max_depth': [3, 5, 7, 10],                  # 树深度 — 梯度提升通常用浅树 (3-5)
        'min_samples_split': [2, 5, 10],
        'subsample': [0.7, 0.8, 1.0],                # 随机抽样比例 — <1.0 引入随机性防过拟合
    },
    'logistic_regression': {
        'C': [0.01, 0.1, 0.5, 1.0, 10.0],           # 正则化强度的倒数 — C越小正则化越强
        'penalty': ['l1', 'l2'],                      # L1(稀疏特征选择) / L2(均匀权重收缩)
        'solver': ['liblinear', 'saga'],              # 优化求解器
        'max_iter': [1000, 3000, 5000],               # 最大迭代次数
    },
    'svm': {
        'C': [0.1, 1.0, 10.0, 100.0],                # 软间隔惩罚系数 — C越大越不允许误分类
        'kernel': ['linear', 'rbf', 'poly'],          # 核函数 — linear(线性), rbf(高斯), poly(多项式)
        'gamma': ['scale', 'auto', 0.01, 0.1],        # 核系数 — 单个样本的影响范围
        'degree': [2, 3, 4],                          # 多项式核的次数
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],         # 邻居数 — K越小越容易过拟合
        'weights': ['uniform', 'distance'],            # 权重 — uniform(等权) / distance(距离反比)
        'metric': ['euclidean', 'manhattan', 'minkowski'],  # 距离度量
        'p': [1, 2],                                   # Minkowski距离的p参数 (1=曼哈顿, 2=欧氏)
    },
    # ---- 回归器搜索空间 ----
    'linear_regression': {
        'fit_intercept': [True, False],               # 是否拟合截距项
        'positive': [True, False],                     # 是否强制系数为正
    },
    'random_forest_regressor': {
        'n_estimators': [50, 100, 200, 300],          # 树的数量
        'max_depth': [None, 10, 20, 30],              # 最大深度
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'gradient_boosting_regressor': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],           # 学习率
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
    },
}

# PyTorch MLP 的搜索空间 (与 sklearn 搜索空间结构不同)
PYTORCH_SEARCH_SPACE = {
    'hidden_layers': [
        [64, 32],                                     # 2层: 简单结构, 适合小数据集
        [128, 64, 32],                                # 3层: 中等深度, 最常用
        [256, 128, 64],                               # 3层宽网络: 更多参数容量
        [128, 128, 64, 32],                           # 4层: 深层网络, 适合复杂数据
    ],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],  # AdamW学习率
    'batch_size': [16, 32, 64, 128],                           # 批量大小
    'dropout': [0.1, 0.2, 0.3, 0.5],                           # Dropout比率
    'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3],                  # L2权重衰减系数
}


class HyperparameterTuningService:
    """超参数自动调优服务"""

    @staticmethod
    def get_search_space(algorithm: str, framework: str = 'sklearn') -> dict:
        """获取指定算法的搜索空间"""
        if framework == 'pytorch':
            return PYTORCH_SEARCH_SPACE
        return SEARCH_SPACES.get(algorithm, {})

    @staticmethod
    def run_grid_search(dataset: Dataset, algorithm: str, task_type: str,
                        target_column: str, scoring: str = 'accuracy',
                        cv: int = 5, n_jobs: int = -1,
                        verbose: int = 1) -> Dict:
        """
        运行 GridSearchCV

        Returns:
            {
                'success': bool,
                'best_params': dict,
                'best_score': float,
                'cv_results': [...],
                'search_time': float,
            }
        """
        from sklearn.model_selection import GridSearchCV
        from app.executor.trainers.sklearn_trainer import _CLASSIFIERS, _REGRESSORS, _import_model

        param_grid = SEARCH_SPACES.get(algorithm, {})
        if not param_grid:
            return {'success': False, 'error': f'算法 "{algorithm}" 无预定义搜索空间。'}

        # 加载数据
        try:
            df = _load_dataset(dataset)
            if df is None:
                return {'success': False, 'error': '无法加载数据集。'}

            if target_column not in df.columns:
                target_column = df.columns[-1]

            X, y = _preprocess(df, target_column, task_type)
        except Exception as e:
            return {'success': False, 'error': f'数据预处理失败: {str(e)}'}

        # 选择模型
        model_map = _CLASSIFIERS if task_type == 'classification' else _REGRESSORS
        model_info = model_map.get(algorithm)
        if not model_info:
            return {'success': False, 'error': f'不支持的算法: {algorithm}'}

        model_cls = _import_model(*model_info)
        try:
            base_model = model_cls(random_state=42)
        except TypeError:
            base_model = model_cls()

        # 根据任务调整 scoring
        if scoring == 'accuracy' and task_type == 'regression':
            scoring = 'neg_mean_squared_error'

        start_time = time.time()
        try:
            grid = GridSearchCV(
                base_model, param_grid,
                scoring=scoring,
                cv=min(cv, len(X) // 10),  # 确保每个fold有足够样本
                n_jobs=n_jobs,
                verbose=verbose,
                error_score='raise',
            )
            grid.fit(X, y)

            search_time = round(time.time() - start_time, 2)

            # 提取 CV 结果
            cv_results = []
            for i in range(len(grid.cv_results_['mean_test_score'])):
                cv_results.append({
                    'params': dict(zip(
                        [k.replace('param_', '') for k in grid.cv_results_.keys() if k.startswith('param_')],
                        [grid.cv_results_[k][i] for k in grid.cv_results_.keys() if k.startswith('param_')]
                    )),
                    'mean_score': round(float(grid.cv_results_['mean_test_score'][i]), 4),
                    'std_score': round(float(grid.cv_results_['std_test_score'][i]), 4),
                    'rank': int(grid.cv_results_['rank_test_score'][i]),
                })
            # 按 rank 排序
            cv_results.sort(key=lambda x: x['rank'])

            logger.info(f'GridSearchCV 完成: {algorithm}, 最佳分数: {grid.best_score_:.4f}, 耗时: {search_time}s')
            return {
                'success': True,
                'best_params': grid.best_params_,
                'best_score': round(float(grid.best_score_), 4),
                'cv_results': cv_results[:20],  # 最多返回前20
                'search_time': search_time,
                'n_combinations': len(grid.cv_results_['mean_test_score']),
                'scoring': scoring,
                'cv_folds': cv,
            }

        except Exception as e:
            logger.error(f'GridSearchCV 失败: {e}', exc_info=True)
            return {'success': False, 'error': f'搜索失败: {str(e)}'}

    @staticmethod
    def run_random_search(dataset: Dataset, algorithm: str, task_type: str,
                          target_column: str, n_iter: int = 30,
                          scoring: str = 'accuracy', cv: int = 5,
                          n_jobs: int = -1) -> Dict:
        """
        运行 RandomizedSearchCV

        Returns: 同上格式
        """
        from sklearn.model_selection import RandomizedSearchCV
        from app.executor.trainers.sklearn_trainer import _CLASSIFIERS, _REGRESSORS, _import_model
        from scipy.stats import randint, uniform, loguniform

        param_distributions = SEARCH_SPACES.get(algorithm, {})
        if not param_distributions:
            return {'success': False, 'error': f'算法 "{algorithm}" 无预定义搜索空间。'}

        # 将列表分布转换为 scipy 分布 (更高效的随机搜索)
        distributions = {}
        for param, values in param_distributions.items():
            if isinstance(values, list) and len(values) > 3:
                if all(isinstance(v, int) or v is None for v in values if v is not None):
                    # 整数参数 — 使用 randint
                    int_vals = [v for v in values if v is not None]
                    if int_vals:
                        distributions[param] = randint(min(int_vals), max(int_vals) + 1)
                elif all(isinstance(v, float) for v in values if v is not None):
                    # 浮点参数 — 使用 loguniform
                    float_vals = [v for v in values if v is not None]
                    if float_vals:
                        distributions[param] = loguniform(min(float_vals), max(float_vals))
                else:
                    distributions[param] = values
            else:
                distributions[param] = values

        # 加载数据
        try:
            df = _load_dataset(dataset)
            if df is None:
                return {'success': False, 'error': '无法加载数据集。'}

            if target_column not in df.columns:
                target_column = df.columns[-1]

            X, y = _preprocess(df, target_column, task_type)
        except Exception as e:
            return {'success': False, 'error': f'数据预处理失败: {str(e)}'}

        model_map = _CLASSIFIERS if task_type == 'classification' else _REGRESSORS
        model_info = model_map.get(algorithm)
        if not model_info:
            return {'success': False, 'error': f'不支持的算法: {algorithm}'}

        model_cls = _import_model(*model_info)
        try:
            base_model = model_cls(random_state=42)
        except TypeError:
            base_model = model_cls()

        if scoring == 'accuracy' and task_type == 'regression':
            scoring = 'neg_mean_squared_error'

        start_time = time.time()
        try:
            search = RandomizedSearchCV(
                base_model, distributions,
                n_iter=min(n_iter, 100),
                scoring=scoring,
                cv=min(cv, len(X) // 10),
                n_jobs=n_jobs,
                random_state=42,
                error_score='raise',
            )
            search.fit(X, y)
            search_time = round(time.time() - start_time, 2)

            cv_results = []
            for i in range(len(search.cv_results_['mean_test_score'])):
                cv_results.append({
                    'params': dict(zip(
                        [k.replace('param_', '') for k in search.cv_results_.keys() if k.startswith('param_')],
                        [search.cv_results_[k][i] for k in search.cv_results_.keys() if k.startswith('param_')]
                    )),
                    'mean_score': round(float(search.cv_results_['mean_test_score'][i]), 4),
                    'std_score': round(float(search.cv_results_['std_test_score'][i]), 4),
                    'rank': int(search.cv_results_['rank_test_score'][i]),
                })
            cv_results.sort(key=lambda x: x['rank'])

            logger.info(f'RandomizedSearchCV 完成: {algorithm}, 最佳: {search.best_score_:.4f}, {search_time}s')
            return {
                'success': True,
                'best_params': search.best_params_,
                'best_score': round(float(search.best_score_), 4),
                'cv_results': cv_results[:20],
                'search_time': search_time,
                'n_combinations': len(search.cv_results_['mean_test_score']),
                'scoring': scoring,
                'cv_folds': cv,
            }

        except Exception as e:
            logger.error(f'RandomizedSearchCV 失败: {e}', exc_info=True)
            return {'success': False, 'error': f'搜索失败: {str(e)}'}

    @staticmethod
    def create_tuned_training(user: User, dataset: Dataset, algorithm: str,
                              task_type: str, target_column: str,
                              tuning_method: str = 'grid',
                              n_iter: int = 30, cv: int = 5,
                              epochs: int = 0) -> Tuple[Optional[TrainingJob], Optional[Dict], Optional[str]]:
        """
        运行超参数搜索并创建使用最佳参数的训练任务

        Args:
            user: 当前用户
            dataset: 数据集
            algorithm: 算法名称
            task_type: 分类/回归
            target_column: 目标列
            tuning_method: 'grid' 或 'random'
            n_iter: 随机搜索迭代次数
            cv: 交叉验证折数
            epochs: 训练epoch数 (sklearn=0, PyTorch>0)

        Returns:
            (TrainingJob, tuning_result, error_message)
        """
        # 运行超参数搜索
        if tuning_method == 'grid':
            tuning_result = HyperparameterTuningService.run_grid_search(
                dataset, algorithm, task_type, target_column, cv=cv
            )
        else:
            tuning_result = HyperparameterTuningService.run_random_search(
                dataset, algorithm, task_type, target_column, n_iter=n_iter, cv=cv
            )

        if not tuning_result.get('success'):
            return None, tuning_result, tuning_result.get('error')

        best_params = tuning_result['best_params']

        # 创建使用最佳参数的训练任务
        from app.services.training_service import TrainingService

        hyperparams = {
            'task_type': task_type,
            'algorithm': algorithm,
            'target_column': target_column,
            'test_size': 0.2,
            'algorithm_params': best_params,  # 最佳参数
            'tuned': True,
            'tuning_method': tuning_method,
            'best_cv_score': tuning_result['best_score'],
            'tuning_cv_folds': cv,
        }

        job_name = f'{algorithm}-Tuned-{datetime.now(timezone.utc).strftime("%H%M")}'

        job, error = TrainingService.create_job(
            user=user,
            name=job_name,
            dataset_id=dataset.id,
            description=f'超参数调优训练 ({tuning_method}), 最佳CV分数: {tuning_result["best_score"]:.4f}',
            framework='sklearn',
            total_epochs=epochs,
            hyperparameters=hyperparams,
            ml_task_type=task_type,
            algorithm=algorithm,
            target_column=target_column,
        )

        if error:
            return None, tuning_result, error

        # 将调优结果保存到模型的 hyperparameters 中
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


# ============ 辅助函数 ============

def _load_dataset(dataset: Dataset) -> pd.DataFrame | None:
    """加载数据集文件"""
    file_path = dataset.file_path
    if not file_path or not os.path.exists(file_path):
        return None

    fmt = dataset.file_format.lower()
    if fmt == 'csv':
        return pd.read_csv(file_path)
    elif fmt in ('xlsx', 'xls'):
        return pd.read_excel(file_path)
    elif fmt == 'json':
        return pd.read_json(file_path)
    elif fmt == 'parquet':
        return pd.read_parquet(file_path)
    return None


def _preprocess(df: pd.DataFrame, target_col: str, task_type: str):
    """预处理数据用于超参数搜索"""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 处理缺失值
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X[num_cols] = SimpleImputer(strategy='mean').fit_transform(X[num_cols])
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X[col] = X[col].fillna('missing')
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # 编码目标变量
    if task_type == 'classification' and y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))

    # 标准化
    X_scaled = StandardScaler().fit_transform(X)

    return X_scaled, y
