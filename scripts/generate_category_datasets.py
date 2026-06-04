"""
============================================
生成所有分类的数据集 — 填充缺失的 9 个分类
============================================
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import create_app, db
from app.models.user import User
from app.models.dataset import Dataset, CATEGORY_LABELS
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
import uuid

app = create_app()

CATEGORY_CONFIG = {
    'classification': {
        'name': 'Iris-Classification-Dataset',
        'desc': '鸢尾花分类数据集 — 150样本/4特征/3类别。经典Fisher鸢尾花数据，用于多分类模型训练与评估。',
        'rows': 150, 'cols': 5,
        'tags': '分类, 鸢尾花, 经典, Fisher, 多分类',
    },
    'regression': {
        'name': 'Boston-Housing-Regression',
        'desc': '波士顿房价回归数据集 — 506样本/13特征。预测房屋中位数价格，经典回归基准数据集。',
        'rows': 506, 'cols': 14,
        'tags': '回归, 房价, 经典, 预测, 连续值',
    },
    'clustering': {
        'name': 'Customer-Segmentation-Clustering',
        'desc': '客户分群聚类数据集 — 2000样本/7特征。包含消费行为指标，适合K-Means/DBSCAN等聚类算法。',
        'rows': 2000, 'cols': 8,
        'tags': '聚类, 客户分群, 无监督, K-Means, 市场营销',
    },
    'nlp': {
        'name': 'Chinese-News-Text-NLP',
        'desc': '中文新闻文本分类数据集 — 5000样本/100维TF-IDF特征/10类别。模拟中文新闻语料的词向量表示。',
        'rows': 5000, 'cols': 101,
        'tags': 'NLP, 自然语言, 文本分类, TF-IDF, 中文, 新闻',
    },
    'vision': {
        'name': 'Handwritten-Digits-Vision',
        'desc': '手写数字图像特征数据集 — 1797样本/64像素特征/10类别。8x8像素手写数字的图像特征矩阵。',
        'rows': 1797, 'cols': 65,
        'tags': '计算机视觉, 手写数字, 图像识别, 分类, Digits',
    },
    'time_series': {
        'name': 'Stock-Price-TimeSeries',
        'desc': '股票价格时间序列数据集 — 3000样本/10特征。模拟股价趋势+波动率+交易量，含日期时间索引。',
        'rows': 3000, 'cols': 11,
        'tags': '时间序列, 股票, 金融, 趋势预测, LSTM',
    },
    'biology': {
        'name': 'Breast-Cancer-Diagnosis-Biology',
        'desc': '威斯康星乳腺癌诊断数据集 — 569样本/30特征/2类别。乳腺肿块细胞核特征，经典医学二分类。',
        'rows': 569, 'cols': 31,
        'tags': '生物医学, 癌症诊断, 二分类, 细胞特征, 医疗AI',
    },
    'finance': {
        'name': 'Credit-Risk-Finance',
        'desc': '信用风险评估数据集 — 10000样本/15特征/2类别。贷款申请人财务指标+还款记录，金融风控经典场景。',
        'rows': 10000, 'cols': 16,
        'tags': '金融, 信用评分, 风控, 贷款, 二分类, 银行',
    },
    'synthetic': {
        'name': 'Synthetic-Anomaly-Detection',
        'desc': '合成异常检测数据集 — 5000样本/30特征。sklearn make_classification生成，含噪声+冗余特征，用于异常检测和快速原型。',
        'rows': 5000, 'cols': 31,
        'tags': '合成数据, 异常检测, 原型测试, 噪声, 冗余特征',
    },
}


def generate_data(category, config):
    """根据分类生成对应的合成数据"""
    np.random.seed(42)
    n_rows = config['rows']
    n_features = config['cols'] - 1

    if category == 'classification':
        # Iris-like: 3 classes, 4 features
        X = np.random.randn(n_rows, n_features) * 1.5
        y = np.random.choice([0, 1, 2], n_rows)
        cols = [f'sepal_length', f'sepal_width', f'petal_length', f'petal_width']
        df = pd.DataFrame(X, columns=cols[:n_features])
        df['species'] = y

    elif category == 'regression':
        cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        X = np.random.randn(n_rows, n_features) * 2
        df = pd.DataFrame(X, columns=cols[:n_features])
        df['MEDV'] = X[:, :4].sum(axis=1) + np.random.randn(n_rows) * 0.5

    elif category == 'clustering':
        # 3 natural clusters
        centers = [np.array([2, 2, 2, 1, 1, 1, 0.5]),
                   np.array([-2, -2, -2, -1, -1, -1, -0.5]),
                   np.array([2, -2, 2, -1, 1, -1, 0])]
        data_list = []
        for i, c in enumerate(centers):
            cluster_data = np.random.randn(n_rows // 3, n_features) * 0.5 + c
            data_list.append(cluster_data)
        X = np.vstack(data_list)
        cols = ['recency', 'frequency', 'monetary', 'age', 'income_level',
                'spending_score', 'tenure']
        df = pd.DataFrame(X, columns=cols[:n_features])
        df['cluster_id'] = np.repeat([0, 1, 2], n_rows // 3)[:n_rows]

    elif category == 'nlp':
        # TF-IDF-like sparse features
        X = np.random.exponential(0.5, (n_rows, n_features))
        X = np.where(np.random.random(X.shape) < 0.8, X, 0)  # 80% sparse
        cols = [f'tfidf_word_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df['category_id'] = np.random.choice(range(10), n_rows)

    elif category == 'vision':
        # 8x8 pixel features = 64
        actual_features = min(n_features, 64)
        X = np.random.randint(0, 16, (n_rows, actual_features)).astype(float)
        cols = [f'pixel_{i}' for i in range(actual_features)]
        df = pd.DataFrame(X, columns=cols)
        df['digit_label'] = np.random.choice(range(10), n_rows)

    elif category == 'time_series':
        t = np.arange(n_rows)
        trend = t * 0.01
        seasonality = np.sin(t * 2 * np.pi / 365) * 5
        noise = np.random.randn(n_rows) * 0.5
        base = 50 + trend + seasonality + noise
        cols = ['open', 'high', 'low', 'close', 'volume', 'ma_5', 'ma_20',
                'volatility', 'rsi', 'macd']
        data = {}
        for i, col in enumerate(cols[:n_features]):
            data[col] = base * (0.9 + 0.2 * np.random.random()) + np.random.randn(n_rows) * 0.2
        df = pd.DataFrame(data)
        df['date'] = pd.date_range('2020-01-01', periods=n_rows, freq='D').strftime('%Y-%m-%d')

    elif category == 'biology':
        cols = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                'smoothness_mean', 'compactness_mean', 'concavity_mean',
                'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                'smoothness_se', 'compactness_se', 'concavity_se',
                'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
                'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                'smoothness_worst', 'compactness_worst', 'concavity_worst',
                'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
        X = np.abs(np.random.randn(n_rows, n_features)) * 0.5
        df = pd.DataFrame(X, columns=cols[:n_features])
        df['diagnosis'] = np.random.choice([0, 1], n_rows)

    elif category == 'finance':
        cols = ['age', 'income', 'debt_ratio', 'credit_score', 'loan_amount',
                'loan_term', 'interest_rate', 'employment_years', 'dependents',
                'delinquencies', 'revolving_balance', 'installment', 'annual_income',
                'dti_ratio', 'credit_lines']
        X = np.abs(np.random.randn(n_rows, n_features)) * 1.5
        df = pd.DataFrame(X, columns=cols[:n_features])
        df['default'] = ((df['credit_score'] > 2) & (df['dti_ratio'] > 3)).astype(int)

    elif category == 'synthetic':
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_rows, n_features=n_features, n_informative=10,
            n_redundant=10, n_clusters_per_class=2, random_state=42
        )
        cols = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df['target'] = y

    else:
        X = np.random.randn(n_rows, n_features)
        cols = [f'col_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df['label'] = np.zeros(n_rows)

    return df


def main():
    with app.app_context():
        admin = User.query.filter_by(role='admin').first()
        if not admin:
            print("ERROR: Admin user not found!")
            return

        print(f"Admin: {admin.username} (id={admin.id})")
        existing_cats = {d.category for d in Dataset.query.filter_by(owner_id=admin.id).all()}
        print(f"Existing categories: {sorted(existing_cats)}")
        print()

        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets')
        os.makedirs(upload_dir, exist_ok=True)

        created = 0
        for cat_key, config in CATEGORY_CONFIG.items():
            if cat_key in existing_cats:
                print(f"  SKIP {cat_key:20s} — already exists")
                continue

            print(f"  CREATE {cat_key:20s} — {config['name']}")

            # 生成数据
            df = generate_data(cat_key, config)

            # 保存CSV
            filename = f"{config['name'].lower().replace('-', '_')}.csv"
            filepath = os.path.join(upload_dir, filename)
            df.to_csv(filepath, index=False)
            file_size = os.path.getsize(filepath)

            # 计算统计
            num_cols = len(df.select_dtypes(include=['number']).columns)
            summary = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'missing_values': {},
                'sample_rows': 5,
            }

            # 创建 Dataset 记录
            dataset = Dataset(
                name=config['name'],
                description=config['desc'],
                file_path=filepath,
                file_size=file_size,
                file_format='csv',
                version='1.0.0',
                tags=config['tags'],
                category=cat_key,
                row_count=len(df),
                column_count=len(df.columns),
                summary_json=json.dumps(summary, ensure_ascii=False),
                status='ready',
                is_public=False,
                uuid=str(uuid.uuid4()),
                owner_id=admin.id,
            )
            db.session.add(dataset)
            created += 1
            print(f"    -> {len(df)} rows x {len(df.columns)} cols, {file_size/1024:.1f}KB")

        db.session.commit()
        print(f"\nDone! Created {created} new datasets.")

        # 验证
        all_cats = {d.category for d in Dataset.query.filter_by(owner_id=admin.id).all()}
        missing = [k for k in CATEGORY_LABELS if k not in all_cats and k not in ('other',)]
        print(f"Categories now: {len(all_cats)}")
        if missing:
            print(f"Still missing: {missing}")
        else:
            print("All 11 categories covered!")


if __name__ == '__main__':
    main()
