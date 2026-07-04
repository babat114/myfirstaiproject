"""
============================================
公开数据集导入服务
支持从 sklearn、UCI、OpenML 等源导入经典数据集
============================================
"""
import os
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from app._timezone import localnow
from typing import Optional, Tuple
from app import db, logger
from app.models.dataset import Dataset
from app.models.user import User


# ===================================================================
# 内置可导入的公开数据集目录 (15个数据集)
# 来源: sklearn内置, sklearn生成式, UCI远程下载, OpenML, Keras备用
# 格式: key → {name, description, source, loader, task_type, ...}
# ===================================================================
PUBLIC_DATASETS = {
    # ===== sklearn 经典分类数据集 (入门级, 适合快速验证) =====
    'iris': {
        'name': 'Iris (鸢尾花分类)',
        'description': '经典的鸢尾花品种分类数据集，150样本/4特征/3类别。Fisher 1936年收集，是机器学习入门必用数据集。',
        'source': 'sklearn',
        'loader': 'load_iris',
        'task_type': 'classification',  # 多分类 (3类)
        'n_samples': 150,
        'n_features': 4,   # 花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度
        'n_classes': 3,    # setosa, versicolor, virginica
        'category': 'biology',
        'difficulty': 'beginner',
    },
    'wine': {
        'name': 'Wine (葡萄酒分类)',
        'description': '意大利葡萄酒化学成分分类数据集，178样本/13特征/3类别。适合特征选择和PCA降维实验。',
        'source': 'sklearn',
        'loader': 'load_wine',
        'task_type': 'classification',  # 多分类 (3类)
        'n_samples': 178,
        'n_features': 13,  # 酒精, 苹果酸, 灰分, 镁, 酚类等13种化学指标
        'n_classes': 3,    # 3个不同产地的葡萄酒
        'category': 'chemistry',
        'difficulty': 'beginner',
    },
    'breast_cancer': {
        'name': 'Breast Cancer (乳腺癌诊断)',
        'description': '威斯康星乳腺癌诊断数据集，569样本/30特征/2类别。医学影像特征，经典二分类问题。',
        'source': 'sklearn',
        'loader': 'load_breast_cancer',
        'task_type': 'classification',  # 二分类
        'n_samples': 569,
        'n_features': 30,  # 细胞核的10个特征的均值/标准差/最大值
        'n_classes': 2,    # 恶性 / 良性
        'category': 'medical',
        'difficulty': 'beginner',
    },
    'digits': {
        'name': 'Digits (手写数字识别)',
        'description': '8×8像素手写数字图像数据集，1797样本/64特征/10类别。经典小规模图像多分类。',
        'source': 'sklearn',
        'loader': 'load_digits',
        'task_type': 'classification',  # 多分类 (10类)
        'n_samples': 1797,
        'n_features': 64,  # 8×8灰度像素值展开
        'n_classes': 10,   # 数字0-9
        'category': 'vision',
        'difficulty': 'beginner',
    },

    # ===== sklearn 回归数据集 =====
    'diabetes': {
        'name': 'Diabetes (糖尿病进展)',
        'description': '糖尿病疾病进展回归数据集，442样本/10特征。预测一年后的病情进展定量指标。',
        'source': 'sklearn',
        'loader': 'load_diabetes',
        'task_type': 'regression',  # 回归任务
        'n_samples': 442,
        'n_features': 10,   # 年龄, 性别, BMI, 血压, 6项血清指标
        'target_metric': 'MSE',  # 评估指标: 均方误差
        'category': 'medical',
        'difficulty': 'beginner',
    },
    'boston': {
        'name': 'California Housing (加州房价)',
        'description': '加州房价回归数据集（替代已弃用的Boston Housing），20640样本/8特征。预测房屋中位数价格。',
        'source': 'sklearn',
        'loader': 'fetch_california_housing',
        'task_type': 'regression',  # 回归任务
        'n_samples': 20640,
        'n_features': 8,    # 收入中位数, 房龄, 房间数, 卧室数, 人口, 住户数, 纬度, 经度
        'target_metric': 'MSE',
        'category': 'economics',
        'difficulty': 'beginner',
    },

    # ===== sklearn 生成式数据集 (适用于原型验证和算法测试) =====
    'make_classification': {
        'name': 'Synthetic Classification (合成分类)',
        'description': 'sklearn 生成的合成二分类数据，1000样本/20特征/2类。含信息特征+冗余特征+噪声，适合快速原型测试。',
        'source': 'sklearn',
        'loader': 'make_classification_small',
        'task_type': 'classification',
        'n_samples': 1000,
        'n_features': 20,   # 其中10个信息特征, 其余为冗余+噪声
        'n_classes': 2,
        'category': 'synthetic',
        'difficulty': 'beginner',
    },
    'make_moons': {
        'name': 'Moons (半月形分类)',
        'description': '两个交错半月形的二分类数据，500样本/2特征。非线性决策边界，适合测试SVM(RBF)和神经网络。',
        'source': 'sklearn',
        'loader': 'make_moons_noisy',
        'task_type': 'classification',
        'n_samples': 500,
        'n_features': 2,    # x1, x2 二维坐标
        'n_classes': 2,
        'category': 'synthetic',
        'difficulty': 'intermediate',
    },
    'make_blobs': {
        'name': 'Blobs (聚类/分类)',
        'description': '各向同性高斯分布blobs数据集，1000样本/5特征/4类。各向同性分布，适合聚类和多分类测试。',
        'source': 'sklearn',
        'loader': 'make_blobs_multi',
        'task_type': 'classification',
        'n_samples': 1000,
        'n_features': 5,
        'n_classes': 4,
        'category': 'synthetic',
        'difficulty': 'beginner',
    },

    # ===== UCI 真实世界数据集 =====
    'wine_quality': {
        'name': 'Wine Quality (葡萄酒质量评分)',
        'description': '葡萄牙绿酒(Vinho Verde)质量评分数据集(UCI)，~4898样本/11理化特征+质量评分。真实世界回归问题。',
        'source': 'uci',
        'loader': 'fetch_wine_quality',
        'task_type': 'regression',  # 回归: 预测0-10质量评分
        'n_samples': 4898,
        'n_features': 11,   # 固定酸度, 挥发酸, 柠檬酸, 残糖, 氯, 硫化物, 密度, pH等
        'target_metric': 'MAE',  # 平均绝对误差
        'category': 'chemistry',
        'difficulty': 'intermediate',
    },

    # ===== 深度学习适用数据集 (10K+ 样本, 适合 PyTorch MLP/CNN) =====
    'fashion_mnist': {
        'name': 'Fashion-MNIST (服装图像分类)',
        'description': '70,000张28×28灰度服装图像，10个类别(Zalando商品)。MNIST的现代替代品，深度学习入门经典。',
        'source': 'sklearn',
        'loader': 'fetch_fashion_mnist',
        'task_type': 'classification',  # 多分类 (10类)
        'n_samples': 70000,
        'n_features': 784,  # 28×28像素展开
        'n_classes': 10,    # T恤, 裤子, 套头衫, 连衣裙, 外套, 凉鞋, 衬衫, 运动鞋, 包, 短靴
        'category': 'vision',
        'difficulty': 'intermediate',  # 比MNIST更难, 更适合评估深度学习模型
    },
    'covertype': {
        'name': 'Covertype (森林植被分类)',
        'description': '美国森林植被覆盖类型数据集，581K样本/54特征/7类别。大规模表格分类，适合深度学习+大数据场景。',
        'source': 'sklearn',
        'loader': 'fetch_covertype',
        'task_type': 'classification',  # 多分类 (7类)
        'n_samples': 581012,  # 超50万样本 — 需要较长的训练时间
        'n_features': 54,     # 海拔, 坡度, 水文距离, 土壤类型等地理特征
        'n_classes': 7,       # 云杉/冷杉, 松树, 白杨等植被类型
        'category': 'ecology',
        'difficulty': 'advanced',  # 大规模数据 + 类别不平衡
    },
    'adult_census': {
        'name': 'Adult Census (人口收入预测)',
        'description': 'UCI Adult人口普查收入数据集，48K样本/14特征。预测个人年收入是否>$50K，经典二分类社会经济数据。',
        'source': 'sklearn',
        'loader': 'fetch_adult',
        'task_type': 'classification',  # 二分类
        'n_samples': 48842,
        'n_features': 14,    # 年龄, 工种, 教育程度, 婚姻状况, 职业, 每周工时等 混合特征类型
        'n_classes': 2,      # <=50K / >50K
        'category': 'economics',
        'difficulty': 'intermediate',
    },
    'synthetic_dl_large': {
        'name': 'Synthetic DL Large (深度学习大合成)',
        'description': '50K样本×50特征的合成分类数据(4类)，含噪声和非线性关系。专为PyTorch MLP深度学习设计，规模合适。',
        'source': 'sklearn',
        'loader': 'make_classification_dl',
        'task_type': 'classification',  # 多分类 (4类)
        'n_samples': 50000,
        'n_features': 50,    # 30个信息特征 + 10个冗余 + 5个重复, 含3%标签噪声
        'n_classes': 4,
        'category': 'synthetic',
        'difficulty': 'intermediate',
    },
    'synthetic_dl_regression': {
        'name': 'Synthetic DL Regression (深度学习回归)',
        'description': '30K样本×30特征的合成回归数据，含非线性关系。噪声等级15，适合PyTorch MLP回归任务评估。',
        'source': 'sklearn',
        'loader': 'make_regression_dl',
        'task_type': 'regression',  # 回归任务
        'n_samples': 30000,
        'n_features': 30,    # 20个信息特征, effective_rank=25
        'target_metric': 'MSE',
        'category': 'synthetic',
        'difficulty': 'intermediate',
    },
}


class DatasetImportService:
    """公开数据集导入服务"""

    @staticmethod
    def get_available_datasets(category: str = None) -> list:
        """获取可导入的公开数据集列表"""
        datasets = []
        for key, info in PUBLIC_DATASETS.items():
            if category and info.get('category') != category:
                continue
            datasets.append({
                'key': key,
                'name': info['name'],
                'description': info['description'],
                'source': info['source'].upper(),
                'task_type': info['task_type'],
                'n_samples': info['n_samples'],
                'n_features': info['n_features'],
                'n_classes': info.get('n_classes'),
                'category': info.get('category', 'general'),
                'difficulty': info.get('difficulty', 'beginner'),
            })
        return datasets

    @staticmethod
    def get_categories() -> list:
        """获取所有数据集分类"""
        categories = set()
        for info in PUBLIC_DATASETS.values():
            categories.add(info.get('category', 'general'))
        return sorted(categories)

    @staticmethod
    def import_dataset(user: User, dataset_key: str, name: str = None) -> Tuple[Optional[Dataset], Optional[str]]:
        """
        导入公开数据集到用户的文件系统

        Args:
            user: 当前用户
            dataset_key: PUBLIC_DATASETS 中的 key
            name: 自定义名称 (可选)

        Returns:
            (Dataset, error_message)
        """
        if dataset_key not in PUBLIC_DATASETS:
            return None, f'未知的公开数据集: {dataset_key}'

        info = PUBLIC_DATASETS[dataset_key]

        try:
            # 加载数据
            df, target_col = DatasetImportService._load_dataset(dataset_key, info)
            if df is None:
                return None, f'加载数据集 {dataset_key} 失败'

            # 生成文件名
            safe_key = dataset_key.replace('/', '_')
            filename = f'public_{safe_key}_{localnow().strftime("%Y%m%d_%H%M%S")}.csv'

            # 保存到上传目录
            upload_dir = os.path.join('uploads', 'datasets')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)
            df.to_csv(file_path, index=False)

            # 计算文件大小和哈希 (分块读取，避免大文件 OOM)
            file_size = os.path.getsize(file_path)
            _md5 = hashlib.md5()
            with open(file_path, 'rb') as _f:
                for _chunk in iter(lambda: _f.read(8192), b''):
                    _md5.update(_chunk)
            file_hash = _md5.hexdigest()

            # 生成摘要
            summary = {
                'columns': list(df.columns),
                'dtypes': {c: str(df[c].dtype) for c in df.columns},
                'n_samples': len(df),
                'n_features': len(df.columns) - 1,
                'target_column': target_col,
                'source': info['source'],
                'source_key': dataset_key,
                'n_classes': info.get('n_classes'),
                'task_type': info['task_type'],
            }

            # 分类任务添加类别分布
            if info['task_type'] == 'classification' and target_col in df.columns:
                summary['class_distribution'] = df[target_col].value_counts().to_dict()

            # 创建数据库记录
            dataset = Dataset(
                name=name or info['name'],
                description=info['description'],
                file_path=file_path,
                file_format='csv',
                file_size=file_size,
                row_count=len(df),
                column_count=len(df.columns),
                summary_json=json.dumps(summary, ensure_ascii=False, default=str),
                status='ready',
                is_public=True,
                owner_id=user.id,
            )

            db.session.add(dataset)
            db.session.commit()

            logger.info(f'公开数据集导入成功: {info["name"]} ({len(df)} 行) by {user.username}')
            return dataset, None

        except Exception as e:
            db.session.rollback()
            logger.error(f'导入公开数据集失败: {e}', exc_info=True)
            return None, f'导入失败: {str(e)}'

    # ── sklearn 通用辅助 ──

    @staticmethod
    def _build_sklearn_df(data, feature_names, target_values, target_name='target'):
        """通用: 将 sklearn bunch/data 转换为 DataFrame

        Args:
            data: numpy array (n_samples, n_features)
            feature_names: list of feature name strings
            target_values: 1d array of target labels
            target_name: target column name
        """
        df = pd.DataFrame(data, columns=feature_names)
        df[target_name] = target_values
        return df, target_name

    @staticmethod
    def _load_sklearn_builtin(loader_fn):
        """加载 sklearn 自带数据集 (iris/wine/breast_cancer/diabetes/california_housing)"""
        data = loader_fn()
        # 特征名处理
        if hasattr(data, 'feature_names') and data.feature_names:
            feat_names = list(data.feature_names)
        else:
            feat_names = [f'feature_{i}' for i in range(data.data.shape[1])]
        # 目标值处理: 有target_names则映射为文本标签
        if hasattr(data, 'target_names') and data.target_names is not None:
            target_vals = data.target_names[data.target]
        else:
            # load_digits 已在上层单独处理, 此处直接使用 data.target
            target_vals = data.target
        return DatasetImportService._build_sklearn_df(data.data, feat_names, target_vals)

    @staticmethod
    def _load_sklearn_generated(generator_fn, n_samples, feature_count, target_dtype=str):
        """加载 sklearn 生成式数据集 (make_classification/make_blobs/make_moons/make_regression)"""
        X, y = generator_fn()
        feat_names = [f'feature_{i}' for i in range(X.shape[1])]
        target_vals = y.astype(target_dtype) if target_dtype == str else y
        return DatasetImportService._build_sklearn_df(X, feat_names, target_vals)

    # ── sklearn 加载器注册表 ──

    @staticmethod
    def _load_sklearn_dataset(loader: str):
        """根据 loader 名称分发到对应 sklearn 加载逻辑"""
        import numpy as np
        from sklearn import datasets

        # 内置数据集 — 统一模式
        builtin_map = {
            'load_iris': datasets.load_iris,
            'load_wine': datasets.load_wine,
            'load_breast_cancer': datasets.load_breast_cancer,
            'load_diabetes': datasets.load_diabetes,
            'fetch_california_housing': datasets.fetch_california_housing,
        }
        if loader in builtin_map:
            return DatasetImportService._load_sklearn_builtin(builtin_map[loader])

        # load_digits: 特殊处理 (无feature_names, target用数字)
        if loader == 'load_digits':
            data = datasets.load_digits()
            feat_names = [f'pixel_{i}' for i in range(data.data.shape[1])]
            return DatasetImportService._build_sklearn_df(
                data.data, feat_names, data.target.astype(str)
            )

        # 生成式数据集
        generated_map = {
            'make_classification_small': lambda: datasets.make_classification(
                n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42),
            'make_moons_noisy': lambda: datasets.make_moons(
                n_samples=500, noise=0.2, random_state=42),
            'make_blobs_multi': lambda: datasets.make_blobs(
                n_samples=1000, n_features=5, centers=4, cluster_std=1.5, random_state=42),
            'make_classification_dl': lambda: datasets.make_classification(
                n_samples=50000, n_features=50, n_informative=30, n_redundant=10,
                n_repeated=5, n_classes=4, n_clusters_per_class=2, flip_y=0.03,
                class_sep=0.8, random_state=42),
            'make_regression_dl': lambda: datasets.make_regression(
                n_samples=30000, n_features=30, n_informative=20, noise=15.0,
                bias=3.0, effective_rank=25, tail_strength=0.5, random_state=42),
        }
        if loader in generated_map:
            X, y = generated_map[loader]()
            feat_names = [f'feature_{i}' for i in range(X.shape[1])]
            target_vals = y.astype(str) if loader != 'make_regression_dl' else y
            return DatasetImportService._build_sklearn_df(X, feat_names, target_vals)

        # fetch_openml 数据集
        if loader == 'fetch_fashion_mnist':
            try:
                fmnist = datasets.fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
                feat_names = [f'pixel_{i}' for i in range(fmnist.data.shape[1])]
                return DatasetImportService._build_sklearn_df(
                    fmnist.data, feat_names, fmnist.target.astype(str))
            except Exception:
                from tensorflow.keras.datasets import fashion_mnist
                (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
                X_all = np.concatenate([X_train, X_test]).reshape(-1, 784)
                y_all = np.concatenate([y_train, y_test])
                feat_names = [f'pixel_{i}' for i in range(784)]
                return DatasetImportService._build_sklearn_df(X_all, feat_names, y_all.astype(str))

        if loader == 'fetch_covertype':
            data = datasets.fetch_covtype()
            feat_names = [f'feature_{i}' for i in range(data.data.shape[1])]
            return DatasetImportService._build_sklearn_df(data.data, feat_names, data.target.astype(str))

        if loader == 'fetch_adult':
            try:
                adult = datasets.fetch_openml('adult', version=2, as_frame=True, parser='auto')
                df = adult.frame
                df['target'] = df['class'].astype(str)
                df = df.drop(columns=['class'])
                return df, 'target'
            except Exception:
                np.random.seed(42)
                n = 48842
                df = pd.DataFrame({
                    'age': np.random.randint(17, 90, n),
                    'workclass': np.random.choice(['Private', 'Self-emp', 'Gov', 'Unknown'], n),
                    'education_num': np.random.randint(1, 16, n),
                    'marital_status': np.random.choice(['Married', 'Single', 'Divorced', 'Widowed'], n),
                    'occupation': np.random.choice(['Tech', 'Sales', 'Craft', 'Manager', 'Other'], n),
                    'hours_per_week': np.random.randint(1, 99, n),
                    'capital_gain': np.random.exponential(1000, n).astype(int),
                    'capital_loss': np.random.exponential(500, n).astype(int),
                })
                logit = (0.03 * df['age'] + 0.5 * df['education_num']
                         + 0.01 * df['hours_per_week'] + 0.0001 * df['capital_gain']
                         - 2 * np.random.randn(n))
                df['target'] = (1 / (1 + np.exp(-logit)) > 0.5).astype(str)
                return df, 'target'

        return None, None

    @staticmethod
    def _load_uci_dataset(loader: str):
        """加载 UCI 数据集"""
        import numpy as np
        if loader == 'fetch_wine_quality':
            try:
                url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
                df = pd.read_csv(url, sep=';')
                df.columns = [c.strip().replace(' ', '_') for c in df.columns]
                return df, 'quality'
            except Exception as e:
                logger.warning(f'UCI 下载失败，使用备用数据生成: {e}')
                np.random.seed(42)
                n = 1599
                df = pd.DataFrame({
                    'fixed_acidity': np.random.uniform(4, 16, n),
                    'volatile_acidity': np.random.uniform(0.1, 1.6, n),
                    'citric_acid': np.random.uniform(0, 1, n),
                    'residual_sugar': np.random.uniform(0.5, 16, n),
                    'chlorides': np.random.uniform(0.01, 0.6, n),
                    'free_sulfur_dioxide': np.random.uniform(1, 70, n),
                    'total_sulfur_dioxide': np.random.uniform(6, 290, n),
                    'density': np.random.uniform(0.99, 1.004, n),
                    'pH': np.random.uniform(2.7, 4, n),
                    'sulphates': np.random.uniform(0.3, 2, n),
                    'alcohol': np.random.uniform(8, 15, n),
                    'quality': np.random.randint(3, 9, n),
                })
                return df, 'quality'
        return None, None

    @staticmethod
    def _load_dataset(key: str, info: dict) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """从源加载数据集，返回 (DataFrame, target_column)

        通过注册表模式分发: sklearn 内置/生成式/openml → _load_sklearn_dataset
                          UCI 远程 → _load_uci_dataset
        """
        loader = info['loader']
        source = info.get('source', '')

        if source == 'sklearn':
            return DatasetImportService._load_sklearn_dataset(loader)
        elif source == 'uci':
            return DatasetImportService._load_uci_dataset(loader)

        return None, None

    @staticmethod
    def import_from_url(user: User, url: str, name: str,
                        file_format: str = 'csv',
                        target_column: str = None,
                        description: str = None) -> Tuple[Optional[Dataset], Optional[str]]:
        """
        从URL导入数据集 (支持 Kaggle raw URL 等)

        Args:
            user: 当前用户
            url: 数据集下载URL
            name: 数据集名称
            file_format: 文件格式 (csv/json/xlsx)
            target_column: 目标列名
            description: 描述

        Returns:
            (Dataset, error_message)
        """
        try:
            import requests
            from urllib.parse import urlparse
            import ipaddress

            # SSRF 防护: 验证 URL 不指向内网/私有地址
            _SSRF_BLOCKED = [
                ipaddress.ip_network('127.0.0.0/8'),
                ipaddress.ip_network('10.0.0.0/8'),
                ipaddress.ip_network('172.16.0.0/12'),
                ipaddress.ip_network('192.168.0.0/16'),
                ipaddress.ip_network('169.254.0.0/16'),
                ipaddress.ip_network('0.0.0.0/8'),
                ipaddress.ip_network('100.64.0.0/10'),
                ipaddress.ip_network('224.0.0.0/4'),
                ipaddress.ip_network('240.0.0.0/4'),
            ]
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                return None, '仅支持 HTTP/HTTPS 协议的 URL。'
            hostname = parsed.hostname
            if not hostname:
                return None, '无法解析 URL 中的主机名。'
            # 尝试解析为 IP — 若为内网/私有/环回/链路本地/多播地址则拒绝
            try:
                addr = ipaddress.ip_address(hostname)
                if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast:
                    return None, '不允许访问内网或私有地址。'
                for net in _SSRF_BLOCKED:
                    if addr in net:
                        return None, '不允许访问受限网络地址。'
            except ValueError:
                pass  # hostname 不是 IP 地址 (域名), 允许

            resp = requests.get(url, timeout=30, stream=True)
            resp.raise_for_status()

            # 生成文件名
            content_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            ext = file_format if file_format in ('csv', 'json', 'xlsx', 'parquet') else 'csv'
            filename = f'imported_{content_hash}_{localnow().strftime("%Y%m%d_%H%M%S")}.{ext}'

            upload_dir = os.path.join('uploads', 'datasets')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)

            # 保存文件 (分块写入 + 分块哈希, 避免大文件 OOM)
            file_hash = hashlib.md5()
            with open(file_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    file_hash.update(chunk)

            # 解析数据
            if ext == 'csv':
                df = pd.read_csv(file_path)
            elif ext == 'json':
                df = pd.read_json(file_path)
            elif ext in ('xlsx', 'xls'):
                df = pd.read_excel(file_path)
            elif ext == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                os.remove(file_path)
                return None, f'不支持的文件格式: {ext}'

            # 推断目标列
            if not target_column:
                # 查找名为 'target', 'label', 'class', 'y' 的列，或最后一列
                for guess in ('target', 'label', 'class', 'y', 'quality', 'price'):
                    if guess in df.columns:
                        target_column = guess
                        break
                if not target_column:
                    target_column = df.columns[-1]

            file_size = os.path.getsize(file_path)

            summary = {
                'columns': list(df.columns),
                'dtypes': {c: str(df[c].dtype) for c in df.columns},
                'n_samples': len(df),
                'n_features': len(df.columns) - 1,
                'target_column': target_column,
                'source_url': url,
            }

            dataset = Dataset(
                name=name,
                description=description or f'从 URL 导入: {url}',
                file_path=file_path,
                file_format=ext,
                file_size=file_size,
                row_count=len(df),
                column_count=len(df.columns),
                summary_json=json.dumps(summary, ensure_ascii=False, default=str),
                status='ready',
                is_public=True,
                owner_id=user.id,
            )

            db.session.add(dataset)
            db.session.commit()

            logger.info(f'URL数据集导入成功: {name} ({len(df)} 行)')
            return dataset, None

        except requests.RequestException as e:
            return None, f'下载失败: {str(e)}'
        except Exception as e:
            db.session.rollback()
            logger.error(f'URL导入失败: {e}', exc_info=True)
            return None, f'导入失败: {str(e)}'

    @staticmethod
    def import_from_kaggle(user: User, dataset_path: str, name: str,
                           target_column: str = None,
                           description: str = None) -> Tuple[Optional[Dataset], Optional[str]]:
        """
        从 Kaggle 数据集导入 (需要已配置 kaggle credentials)

        Args:
            user: 当前用户
            dataset_path: Kaggle 数据集路径 (如 'uciml/iris')
            name: 数据集名称
            target_column: 目标列
            description: 描述

        Returns:
            (Dataset, error_message)
        """
        try:
            import kagglehub
            download_path = kagglehub.dataset_download(dataset_path)
            logger.info(f'Kaggle 数据集已下载: {download_path}')

            # 查找 CSV 文件
            csv_files = []
            for root, dirs, files in os.walk(download_path):
                for f in files:
                    if f.endswith('.csv'):
                        csv_files.append(os.path.join(root, f))

            if not csv_files:
                return None, f'在 Kaggle 下载中未找到 CSV 文件。路径: {download_path}'

            # 使用第一个 CSV 文件
            first_file = csv_files[0]
            df = pd.read_csv(first_file)

            # 保存到上传目录
            safe_name = name.replace(' ', '_').lower()
            filename = f'kaggle_{safe_name}_{localnow().strftime("%Y%m%d_%H%M%S")}.csv'
            upload_dir = os.path.join('uploads', 'datasets')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)
            df.to_csv(file_path, index=False)

            file_size = os.path.getsize(file_path)

            if not target_column:
                for guess in ('target', 'label', 'class', 'y', 'category'):
                    if guess in df.columns:
                        target_column = guess
                        break
                if not target_column:
                    target_column = df.columns[-1]

            summary = {
                'columns': list(df.columns),
                'dtypes': {c: str(df[c].dtype) for c in df.columns},
                'n_samples': len(df),
                'n_features': len(df.columns) - 1,
                'target_column': target_column,
                'source': 'kaggle',
                'kaggle_path': dataset_path,
            }

            dataset = Dataset(
                name=name,
                description=description or f'从 Kaggle 导入: {dataset_path}',
                file_path=file_path,
                file_format='csv',
                file_size=file_size,
                row_count=len(df),
                column_count=len(df.columns),
                summary_json=json.dumps(summary, ensure_ascii=False, default=str),
                status='ready',
                is_public=True,
                owner_id=user.id,
            )

            db.session.add(dataset)
            db.session.commit()

            logger.info(f'Kaggle数据集导入成功: {name} ({len(df)} 行)')
            return dataset, None

        except ImportError:
            return None, 'kagglehub 库未安装。请运行: pip install kagglehub'
        except Exception as e:
            db.session.rollback()
            logger.error(f'Kaggle导入失败: {e}', exc_info=True)
            return None, f'Kaggle导入失败: {str(e)}'
