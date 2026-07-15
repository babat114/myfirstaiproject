"""
============================================
数据集分析与推荐服务
分析数据集特征，推荐最佳模型类型、算法和框架
============================================
"""

import os

import numpy as np
import pandas as pd

from app import logger


class DatasetAnalyzer:
    """数据集特征分析器 — 深度分析数据集内容并提取关键特征

    分析维度:
        - 基本统计: 样本数、特征数、文件大小
        - 数据类型分布: 数值型/分类型/文本型比例
        - 目标变量特征: 类别数、缺失率、是否连续
        - 数据质量: 缺失值比例、重复行比例、异常值
        - 特征相关性: 高相关特征对数量
        - 类平衡度: 分类任务中各类别比例
    """

    @staticmethod
    def analyze(file_path: str, target_col: str = None, file_format: str = 'csv') -> dict:
        """完整分析数据集，返回所有特征指标"""
        try:
            df = DatasetAnalyzer._load(file_path, file_format)
            if df is None or df.empty:
                return {'error': '无法加载数据集'}

            result = {
                'file_size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
                'n_samples': len(df),
                'n_features': len(df.columns),
                'column_names': list(df.columns),
            }

            # 数据类型分布
            type_dist = DatasetAnalyzer._analyze_types(df)
            result.update(type_dist)

            # 缺失值
            result['missing_rate'] = round(df.isnull().mean().mean(), 4)

            # 目标变量分析
            if target_col and target_col in df.columns:
                result.update(DatasetAnalyzer._analyze_target(df, target_col))
            else:
                # 尝试推测最后一列
                result.update(DatasetAnalyzer._analyze_target(df, df.columns[-1]))

            # 文本特征检测
            text_cols = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() > len(df) * 0.1]
            result['text_column_count'] = len(text_cols)
            result['text_heavy'] = len(text_cols) >= max(1, len(df.columns) * 0.2)
            result['text_columns'] = text_cols

            # 特征相关性 (仅数值列)
            num_df = df.select_dtypes(include=[np.number])
            if num_df.shape[1] >= 2:
                corr = num_df.corr().abs()
                high_corr = ((corr > 0.85) & (corr < 1.0)).sum().sum() // 2
                result['high_corr_pairs'] = int(high_corr)
                result['avg_correlation'] = round(
                    (corr.sum().sum() - num_df.shape[1]) / max(1, num_df.shape[1] * (num_df.shape[1] - 1)), 4
                )
            else:
                result['high_corr_pairs'] = 0
                result['avg_correlation'] = 0

            # 维度判断
            result['wide_data'] = result['n_features'] > result['n_samples']
            result['high_dim'] = result['n_features'] > 100

            logger.info(
                f'数据集分析完成: {result.get("n_samples")}样本, '
                f'{result.get("n_features")}特征, '
                f'类别数={result.get("n_classes", "N/A")}'
            )
            return result

        except Exception as e:
            logger.error(f'数据集分析失败: {e}')
            return {'error': str(e)}

    @staticmethod
    def _load(df_path: str, fmt: str) -> pd.DataFrame | None:
        from app.utils.data_io import load_dataframe

        # 加载最多 100000 行以准确分析数据集特征，避免因采样不足导致推荐偏差
        return load_dataframe(df_path, fmt, nrows=100000)

    @staticmethod
    def _analyze_types(df) -> dict:
        numeric = df.select_dtypes(include=[np.number]).columns
        categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns
        n_num, n_cat = len(numeric), len(categorical)
        total = max(1, n_num + n_cat)
        return {
            'numeric_cols': n_num,
            'categorical_cols': n_cat,
            'numeric_ratio': round(n_num / total, 3),
            'categorical_ratio': round(n_cat / total, 3),
            'mostly_numeric': n_num / total > 0.7,
            'mostly_categorical': n_cat / total > 0.5,
            'mixed_types': n_num > 0 and n_cat > 0 and abs(n_num - n_cat) / total < 0.4,
        }

    @staticmethod
    def _analyze_target(df, target_col) -> dict:
        y = df[target_col]
        result = {}
        if y.dtype in ('object', 'category', 'bool') or y.nunique() <= 30:
            result['target_type'] = 'categorical'
            result['n_classes'] = y.nunique()
            vc = y.value_counts()
            result['class_balance'] = round(vc.min() / max(1, vc.max()), 4)
            result['imbalanced'] = result['class_balance'] < 0.2
            result['binary'] = result['n_classes'] == 2
            result['multi_class'] = result['n_classes'] > 2
        else:
            result['target_type'] = 'continuous'
            result['n_classes'] = 0
            result['target_mean'] = round(float(y.mean()), 4)
            result['target_std'] = round(float(y.std()), 4)
            result['target_skew'] = round(float(y.skew()), 4) if len(y) > 1 else 0
            result['binary'] = False
            result['multi_class'] = False
            result['class_balance'] = 1.0
            result['imbalanced'] = False
        return result


class DatasetRecommendationService:
    """数据集推荐引擎 — 根据数据集特征推荐最佳模型配置

    推荐逻辑 (基于ML最佳实践):
        - 数值+分类目标(2类) → LogisticRegression/RandomForest/SVM/GradientBoosting
        - 数值+分类目标(多类) → RandomForest/GradientBoosting/KNN/MLP
        - 数值+连续目标 → Ridge/RandomForestRegressor/GradientBoostingRegressor/MLP
        - 文本特征多 → NLP (TF-IDF + classifier)
        - 高维稀疏 → SVM (rbf) / MLP
        - 类别不平衡 → 集成方法 (RandomForest/GradientBoosting)
        - 样本>10K → 考虑PyTorch/TensorFlow深度学习
        - 少量样本 → 简单模型 (LogisticRegression/Ridge)
    """

    @staticmethod
    def recommend(
        file_path: str, target_col: str = None, file_format: str = 'csv', known_n_samples: int = None
    ) -> dict:
        """分析数据集并返回推荐结果

        Args:
            file_path: 数据集文件路径
            target_col: 目标列名
            file_format: 文件格式
            known_n_samples: 已知的实际样本数 (优先于分析器采样数, 来自 DB row_count)
        """
        analysis = DatasetAnalyzer.analyze(file_path, target_col, file_format)
        if 'error' in analysis:
            return {'error': analysis['error']}

        # 使用已知的实际样本数 (避免采样偏差)
        if known_n_samples and known_n_samples > 0:
            analysis['n_samples'] = known_n_samples

        # 基础信息
        analysis['n_samples']
        analysis['n_features']
        target_type = analysis.get('target_type', 'categorical')
        n_classes = analysis.get('n_classes', 0)
        text_heavy = analysis.get('text_heavy', False)
        analysis.get('imbalanced', False)
        analysis.get('mostly_numeric', True)
        analysis.get('high_dim', False)
        analysis.get('wide_data', False)
        analysis.get('missing_rate', 0)

        # 确定最佳模型类型
        if text_heavy:
            primary_type = 'nlp'
            reason = f'检测到{analysis.get("text_column_count", 0)}个文本列，适合NLP处理'
            types = [
                {'model_type': 'nlp', 'confidence': 0.85, 'reason': reason},
                {'model_type': 'classification', 'confidence': 0.40, 'reason': '可尝试将文本特征编码后分类'},
            ]
        elif target_type == 'continuous':
            primary_type = 'regression'
            types = [
                {
                    'model_type': 'regression',
                    'confidence': 0.90,
                    'reason': f'目标变量为连续值 (std={analysis.get("target_std", "?"):.2f})',
                },
                {'model_type': 'clustering', 'confidence': 0.30, 'reason': '无监督替代方案'},
            ]
        elif target_type == 'categorical':
            if n_classes == 2:
                primary_type = 'classification'
                types = [{'model_type': 'classification', 'confidence': 0.92, 'reason': f'二分类任务 ({n_classes}类)'}]
            else:
                primary_type = 'classification'
                types = [{'model_type': 'classification', 'confidence': 0.90, 'reason': f'多分类任务 ({n_classes}类)'}]
        else:
            primary_type = 'other'
            types = [{'model_type': 'other', 'confidence': 0.50, 'reason': '自动检测'}]

        # 推荐具体算法
        algorithms = DatasetRecommendationService._recommend_algorithms(analysis, target_type, n_classes, text_heavy)

        # 推荐框架
        frameworks = DatasetRecommendationService._recommend_frameworks(analysis, text_heavy)

        # 推荐超参数预设
        param_presets = DatasetRecommendationService._recommend_params(analysis)

        return {
            'analysis': analysis,
            'recommended_types': types,
            'recommended_algorithms': algorithms,
            'recommended_frameworks': frameworks,
            'param_presets': param_presets,
            'primary_type': primary_type,
            'summary': DatasetRecommendationService._generate_summary(analysis, primary_type),
        }

    @staticmethod
    def _recommend_algorithms(analysis, target_type, n_classes, text_heavy) -> list:
        """根据分析结果推荐算法 (按推荐度排序)"""
        n_samples = analysis['n_samples']
        n_features = analysis['n_features']
        imbalanced = analysis.get('imbalanced', False)
        high_dim = analysis.get('high_dim', False)

        algorithms = []

        if text_heavy:
            algorithms = [
                {
                    'algorithm': 'transformer_bert',
                    'display': 'BERT Transformer 微调',
                    'confidence': 0.95,
                    'reason': '预训练迁移学习，NLP首选方案',
                },
                {
                    'algorithm': 'tfidf_logistic',
                    'display': 'TF-IDF + LogisticRegression',
                    'confidence': 0.70,
                    'reason': '轻量文本分类，快速baseline',
                },
                {
                    'algorithm': 'tfidf_svm',
                    'display': 'TF-IDF + SVM',
                    'confidence': 0.65,
                    'reason': '高维稀疏文本数据SVM效果好',
                },
            ]
        elif high_dim and n_features > 200:
            # 视觉embedding特征 → PyTorch深度学习
            algorithms = [
                {
                    'algorithm': 'mlp',
                    'display': 'PyTorch MLP (宽网络)',
                    'confidence': 0.88,
                    'reason': f'{n_features}维视觉特征适合深度MLP',
                },
                {
                    'algorithm': 'random_forest',
                    'display': 'Random Forest',
                    'confidence': 0.55,
                    'reason': '传统方法baseline',
                },
            ]
        elif target_type == 'continuous':
            if n_samples > 5000:
                algorithms = [
                    {
                        'algorithm': 'gradient_boosting_regressor',
                        'display': 'Gradient Boosting Regressor',
                        'confidence': 0.88,
                        'reason': '大数据集回归首选',
                    },
                    {
                        'algorithm': 'random_forest_regressor',
                        'display': 'Random Forest Regressor',
                        'confidence': 0.82,
                        'reason': '鲁棒性好，不易过拟合',
                    },
                    {
                        'algorithm': 'ridge',
                        'display': 'Ridge Regression',
                        'confidence': 0.70,
                        'reason': '线性基线，快速训练',
                    },
                    {
                        'algorithm': 'mlp',
                        'display': 'PyTorch MLP Regressor',
                        'confidence': 0.65,
                        'reason': f'{n_features}个特征适合MLP',
                    },
                ]
            else:
                algorithms = [
                    {
                        'algorithm': 'ridge',
                        'display': 'Ridge Regression',
                        'confidence': 0.85,
                        'reason': '小样本首选线性模型',
                    },
                    {
                        'algorithm': 'random_forest_regressor',
                        'display': 'Random Forest Regressor',
                        'confidence': 0.75,
                        'reason': '集成方法，控制过拟合',
                    },
                    {
                        'algorithm': 'knn_regressor',
                        'display': 'KNN Regressor',
                        'confidence': 0.60,
                        'reason': '简单非参数方法',
                    },
                ]
        elif target_type == 'categorical':
            if n_samples > 5000 and not imbalanced:
                algorithms = [
                    {
                        'algorithm': 'random_forest',
                        'display': 'Random Forest',
                        'confidence': 0.90,
                        'reason': '综合性能最佳',
                    },
                    {
                        'algorithm': 'gradient_boosting',
                        'display': 'Gradient Boosting',
                        'confidence': 0.87,
                        'reason': '高准确率，大数据集表现好',
                    },
                    {
                        'algorithm': 'mlp',
                        'display': 'PyTorch MLP',
                        'confidence': 0.72,
                        'reason': f'{n_features}维特征深度学习',
                    },
                ]
            elif imbalanced:
                algorithms = [
                    {
                        'algorithm': 'random_forest',
                        'display': 'Random Forest (class_weight=balanced)',
                        'confidence': 0.88,
                        'reason': '处理不平衡数据',
                    },
                    {
                        'algorithm': 'gradient_boosting',
                        'display': 'Gradient Boosting',
                        'confidence': 0.82,
                        'reason': '集成方法对不平衡鲁棒',
                    },
                ]
            elif n_samples < 1000:
                algorithms = [
                    {'algorithm': 'svm', 'display': 'SVM (RBF Kernel)', 'confidence': 0.82, 'reason': '小样本高精度'},
                    {
                        'algorithm': 'logistic_regression',
                        'display': 'Logistic Regression',
                        'confidence': 0.78,
                        'reason': '简单可解释',
                    },
                ]
            else:
                algorithms = [
                    {
                        'algorithm': 'random_forest',
                        'display': 'Random Forest',
                        'confidence': 0.88,
                        'reason': '通用分类首选',
                    },
                    {'algorithm': 'svm', 'display': 'SVM', 'confidence': 0.75, 'reason': '高维数据效果好'},
                    {'algorithm': 'knn', 'display': 'KNN', 'confidence': 0.65, 'reason': '简单直观'},
                ]
        else:
            algorithms = [
                {'algorithm': 'random_forest', 'display': 'Random Forest', 'confidence': 0.60, 'reason': '通用算法'},
                {'algorithm': 'mlp', 'display': 'PyTorch MLP', 'confidence': 0.50, 'reason': '深度学习'},
            ]

        return algorithms[:5]  # Top 5

    @staticmethod
    def _recommend_frameworks(analysis, text_heavy) -> list:
        n_samples = analysis['n_samples']
        analysis['n_features']
        high_dim = analysis.get('high_dim', False)

        frameworks = [
            {'framework': 'sklearn', 'confidence': 0.90, 'reason': '通用首选，API简单'},
        ]

        if n_samples > 5000 or high_dim or text_heavy:
            frameworks.append({'framework': 'pytorch', 'confidence': 0.75, 'reason': '大数据/高维/文本场景深度学习'})
            frameworks.append({'framework': 'tensorflow', 'confidence': 0.65, 'reason': 'Keras Sequential快速原型'})
        else:
            frameworks.append({'framework': 'pytorch', 'confidence': 0.50, 'reason': '可尝试MLP网络'})

        return frameworks

    @staticmethod
    def _recommend_params(analysis) -> dict:
        """根据数据集特征推荐超参数预设"""
        n_samples = analysis['n_samples']
        n_features = analysis['n_features']
        analysis.get('n_classes', 0)
        analysis.get('target_type', 'categorical')
        imbalanced = analysis.get('imbalanced', False)

        params = {}

        # Epochs (深度学习相关)
        if n_samples > 10000:
            params['epochs'] = 5
        elif n_samples > 5000:
            params['epochs'] = 10
        else:
            params['epochs'] = 20

        # Test size
        if n_samples > 20000:
            params['test_size'] = 0.1
        elif n_samples > 5000:
            params['test_size'] = 0.2
        else:
            params['test_size'] = 0.25

        # Batch size (深度学习)
        if n_samples > 20000:
            params['batch_size'] = 128
        else:
            params['batch_size'] = 64

        # Hidden layers (深度学习)
        if n_features > 50:
            params['hidden_layers'] = [256, 128, 64]
        elif n_features > 20:
            params['hidden_layers'] = [128, 64, 32]
        else:
            params['hidden_layers'] = [64, 32]

        # Learning rate
        params['learning_rate'] = 0.001

        # Dropout
        if n_samples < 5000:
            params['dropout'] = 0.4
        else:
            params['dropout'] = 0.3

        # 类别不平衡
        if imbalanced:
            params['class_weight'] = 'balanced'

        return params

    @staticmethod
    def _generate_summary(analysis, primary_type) -> str:
        """生成可读的推荐摘要"""
        n_samples = analysis['n_samples']
        n_features = analysis['n_features']
        target_type = analysis.get('target_type', '?')
        n_classes = analysis.get('n_classes', 0)
        missing = analysis.get('missing_rate', 0)
        imbalanced = analysis.get('imbalanced', False)

        type_cn = {
            'classification': '分类',
            'regression': '回归',
            'nlp': '自然语言处理',
            'clustering': '聚类',
            'computer_vision': '计算机视觉',
            'other': '其他',
        }

        parts = [
            f'检测到 {n_samples:,} 个样本、{n_features} 个特征的数据集',
            f'目标类型: {target_type}' + (f' ({n_classes}类)' if target_type == 'categorical' else ''),
            f'推荐模型类型: {type_cn.get(primary_type, primary_type)}',
        ]
        if missing > 0.05:
            parts.append(f'⚠ 缺失值较多 ({missing:.1%})，建议先做数据清洗')
        if imbalanced:
            parts.append('⚠ 类别不平衡，建议使用balanced权重或重采样')
        if analysis.get('high_corr_pairs', 0) > 3:
            parts.append('💡 存在多重共线性，可考虑降维')

        return '。'.join(parts) + '。'
