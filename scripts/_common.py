"""
共享脚本基础设施
提供统一的路径设置、argparse 参数、Flask app 初始化

Usage:
    from scripts._common import PROJECT_ROOT, create_base_parser, app_context

    if __name__ == '__main__':
        parser = create_base_parser('脚本描述')
        parser.add_argument('--custom-arg', help='自定义参数')
        args = parser.parse_args()

        with app_context():
            # ... 在 Flask 应用上下文中执行
"""
import os
import sys
import argparse
import logging
from contextlib import contextmanager

# ═══════════════════════════════════════════════════════════════
# 路径设置 — 确保项目根目录在 sys.path 中
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'uploads', 'models')


# ═══════════════════════════════════════════════════════════════
# argparse 工具
# ═══════════════════════════════════════════════════════════════

def create_base_parser(description: str = '', **kwargs) -> argparse.ArgumentParser:
    """创建带有标准共享参数的 argparse parser。

    所有脚本应该使用这个工厂函数，确保 --dry-run 和 --verbose 一致性。
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        **kwargs,
    )
    parser.add_argument(
        '--dry-run', action='store_true', default=False,
        help='仅打印计划, 不执行实际操作',
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true', default=False,
        help='详细输出 (DEBUG 级别日志)',
    )
    return parser


def add_training_args(parser: argparse.ArgumentParser):
    """添加训练相关的共享参数组。"""
    group = parser.add_argument_group('Training options')
    group.add_argument('--task-type', choices=['classification', 'regression', 'clustering'],
                       help='任务类型')
    group.add_argument('--algorithm', help='算法名称')
    group.add_argument('--target', dest='target_column', help='目标列名')
    group.add_argument('--test-size', type=float, default=0.2,
                       help='测试集比例 (default: 0.2)')
    group.add_argument('--epochs', type=int, default=50,
                       help='训练轮数 (default: 50)')


def add_nlp_args(parser: argparse.ArgumentParser):
    """添加 NLP 相关的共享参数组。"""
    group = parser.add_argument_group('NLP options')
    group.add_argument('--max-features', type=int, default=2000,
                       help='TF-IDF 最大特征数 (default: 2000)')
    group.add_argument('--min-df', type=int, default=2,
                       help='TF-IDF 最小文档频率 (default: 2)')
    group.add_argument('--max-df', type=float, default=0.9,
                       help='TF-IDF 最大文档频率 (default: 0.9)')
    group.add_argument('--balance', choices=['smote', 'undersample', 'none'],
                       help='类别平衡策略')
    group.add_argument('--augment', dest='augment_factor', type=int,
                       help='文本数据增强倍数')
    group.add_argument('--cv-folds', type=int,
                       help='交叉验证折数')


def add_model_filter_args(parser: argparse.ArgumentParser):
    """添加模型筛选参数组。"""
    group = parser.add_argument_group('Model filter')
    group.add_argument('--dataset', help='按数据集名筛选')
    group.add_argument('--algo', help='按算法筛选')
    group.add_argument('--framework', choices=['sklearn', 'pytorch', 'keras', 'transformers'],
                       help='按框架筛选')
    group.add_argument('--limit', type=int, help='限制处理数量')


def setup_verbose(args):
    """根据 --verbose 标志配置日志级别。"""
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S',
        )


# ═══════════════════════════════════════════════════════════════
# Flask 应用工具
# ═══════════════════════════════════════════════════════════════

@contextmanager
def app_context(config=None):
    """Flask 应用上下文管理器 — 自动处理 create_app 和上下文进出。

    Usage:
        with app_context() as app:
            # 在 Flask 应用上下文中执行
            from app import db
            ...
    """
    from app import create_app
    app = create_app(config)
    with app.app_context():
        yield app


# ═══════════════════════════════════════════════════════════════
# 批量训练共享配置 (从 batch_train.py / batch_train_v2.py 提取)
# ═══════════════════════════════════════════════════════════════

# 标准 8 数据集训练配置
STANDARD_JOBS = [
    {'file': 'iris_classification.csv',          'name': 'Iris-随机森林',     'task': 'classification', 'algo': 'random_forest',              'target': 'species'},
    {'file': 'wine_classification.csv',          'name': 'Wine-SVM分类',      'task': 'classification', 'algo': 'svm',                       'target': 'wine_class'},
    {'file': 'breast_cancer_classification.csv', 'name': '乳腺癌-逻辑回归',     'task': 'classification', 'algo': 'logistic_regression',       'target': 'diagnosis'},
    {'file': 'digits_classification.csv',        'name': 'Digits-随机森林',    'task': 'classification', 'algo': 'random_forest',              'target': 'digit'},
    {'file': 'synthetic_binary_classification.csv', 'name': '合成二分类-KNN',  'task': 'classification', 'algo': 'knn',                       'target': 'label'},
    {'file': 'diabetes_regression.csv',          'name': '糖尿病-线性回归',     'task': 'regression',     'algo': 'linear_regression',          'target': 'disease_progression'},
    {'file': 'housing_regression.csv',           'name': '房价-随机森林回归',   'task': 'regression',     'algo': 'random_forest_regressor',    'target': 'median_house_value'},
    {'file': 'california_regression.csv',        'name': '加州房价-梯度提升',   'task': 'regression',     'algo': 'gradient_boosting_regressor','target': 'median_house_value'},
]
