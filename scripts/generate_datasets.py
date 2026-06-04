"""
============================================
数据集批量生成脚本
从 sklearn 加载经典 ML 数据集 → 保存为 CSV
运行: python scripts/generate_datasets.py
============================================
"""
import os
import sys
import csv
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np


def generate_all_datasets(output_dir: str) -> list[dict]:
    """生成所有数据集并返回信息列表"""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    # ==========================================
    # 1. Iris (鸢尾花分类) — 经典三分类
    # ==========================================
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Series(iris.target).map({i: n for i, n in enumerate(iris.target_names)})
    path = os.path.join(output_dir, 'iris_classification.csv')
    df.to_csv(path, index=False)
    records.append({
        'name': 'Iris 鸢尾花分类',
        'file': 'iris_classification.csv',
        'rows': len(df),
        'columns': len(df.columns),
        'task': 'classification',
        'classes': 3,
        'target': 'species',
        'description': '经典鸢尾花三分类数据集：setosa/versicolor/virginica',
        'algorithms': 'random_forest, logistic_regression, svm, knn',
        'difficulty': 'easy',
    })

    # ==========================================
    # 2. Wine (葡萄酒分类) — 三分类
    # ==========================================
    from sklearn.datasets import load_wine
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['wine_class'] = wine.target
    path = os.path.join(output_dir, 'wine_classification.csv')
    df.to_csv(path, index=False)
    records.append({
        'name': 'Wine 葡萄酒分类',
        'file': 'wine_classification.csv',
        'rows': len(df),
        'columns': len(df.columns),
        'task': 'classification',
        'classes': 3,
        'target': 'wine_class',
        'description': '葡萄酒化学成分三分类数据集',
        'algorithms': 'random_forest, svm, logistic_regression, knn',
        'difficulty': 'easy',
    })

    # ==========================================
    # 3. Breast Cancer (乳腺癌诊断) — 二分类
    # ==========================================
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['diagnosis'] = cancer.target  # 0=malignant, 1=benign
    path = os.path.join(output_dir, 'breast_cancer_classification.csv')
    df.to_csv(path, index=False)
    records.append({
        'name': 'Breast Cancer 乳腺癌诊断',
        'file': 'breast_cancer_classification.csv',
        'rows': len(df),
        'columns': len(df.columns),
        'task': 'classification',
        'classes': 2,
        'target': 'diagnosis',
        'description': '乳腺癌良恶性二分类 (30个特征)',
        'algorithms': 'random_forest, logistic_regression, svm, knn',
        'difficulty': 'medium',
    })

    # ==========================================
    # 4. Diabetes (糖尿病回归) — 回归
    # ==========================================
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['disease_progression'] = diabetes.target
    path = os.path.join(output_dir, 'diabetes_regression.csv')
    df.to_csv(path, index=False)
    records.append({
        'name': 'Diabetes 糖尿病进展',
        'file': 'diabetes_regression.csv',
        'rows': len(df),
        'columns': len(df.columns),
        'task': 'regression',
        'classes': None,
        'target': 'disease_progression',
        'description': '糖尿病疾病进展定量预测 (回归任务)',
        'algorithms': 'linear_regression, random_forest_regressor, svr',
        'difficulty': 'medium',
    })

    # ==========================================
    # 5. Housing 房价回归 (合成 + 真实特征)
    # ==========================================
    # California housing 需要从外网下载，这里用合成数据替代
    from sklearn.datasets import make_regression as make_reg2
    X, y, coef = make_reg2(
        n_samples=5000, n_features=8, n_informative=6,
        noise=0.1, coef=True, random_state=42
    )
    feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                     'Population', 'AveOccup', 'Latitude', 'Longitude']
    df = pd.DataFrame(X, columns=feature_names)
    df['median_house_value'] = y * 100000 + 200000  # 缩放到合理房价范围
    df['median_house_value'] = df['median_house_value'].clip(50000, 500000)
    path = os.path.join(output_dir, 'housing_regression.csv')
    df.to_csv(path, index=False)
    records.append({
        'name': 'Housing 房价预测',
        'file': 'housing_regression.csv',
        'rows': len(df),
        'columns': len(df.columns),
        'task': 'regression',
        'classes': None,
        'target': 'median_house_value',
        'description': '房价中位数预测 (5000条, 8个特征, 合成数据)',
        'algorithms': 'random_forest_regressor, linear_regression, gradient_boosting_regressor, svr',
        'difficulty': 'medium',
    })

    # ==========================================
    # 6. Digits (手写数字识别) — 十分类
    # ==========================================
    from sklearn.datasets import load_digits
    digits = load_digits()
    df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(digits.data.shape[1])])
    df['digit'] = digits.target
    path = os.path.join(output_dir, 'digits_classification.csv')
    df.to_csv(path, index=False)
    records.append({
        'name': 'Digits 手写数字识别',
        'file': 'digits_classification.csv',
        'rows': len(df),
        'columns': len(df.columns),
        'task': 'classification',
        'classes': 10,
        'target': 'digit',
        'description': '8x8手写数字图像 → 64个像素特征 + 标签0-9',
        'algorithms': 'random_forest, svm, logistic_regression, knn',
        'difficulty': 'hard',
    })

    # ==========================================
    # 7. 合成二分类数据集 (make_classification)
    # ==========================================
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=10,
        n_redundant=5, n_classes=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
    df['label'] = y
    path = os.path.join(output_dir, 'synthetic_binary_classification.csv')
    df.to_csv(path, index=False)
    records.append({
        'name': 'Synthetic 合成二分类',
        'file': 'synthetic_binary_classification.csv',
        'rows': len(df),
        'columns': len(df.columns),
        'task': 'classification',
        'classes': 2,
        'target': 'label',
        'description': 'sklearn生成的合成二分类数据 (2000条, 20特征)',
        'algorithms': 'random_forest, logistic_regression, svm, knn',
        'difficulty': 'medium',
    })

    # ==========================================
    # 8. 合成回归数据集 (make_regression)
    # ==========================================
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=3000, n_features=15, n_informative=8,
        noise=15, random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
    df['target_value'] = y
    path = os.path.join(output_dir, 'synthetic_regression.csv')
    df.to_csv(path, index=False)
    records.append({
        'name': 'Synthetic 合成回归',
        'file': 'synthetic_regression.csv',
        'rows': len(df),
        'columns': len(df.columns),
        'task': 'regression',
        'classes': None,
        'target': 'target_value',
        'description': 'sklearn生成的合成回归数据 (3000条, 15特征)',
        'algorithms': 'linear_regression, random_forest_regressor, svr, gradient_boosting_regressor',
        'difficulty': 'easy',
    })

    return records


def save_summary_table(records: list[dict], output_dir: str):
    """保存数据集汇总表为 CSV"""
    df = pd.DataFrame(records)
    table_path = os.path.join(output_dir, 'datasets_summary.csv')
    df.to_csv(table_path, index=False, encoding='utf-8-sig')
    print(f'[OK] Summary table saved: {table_path}')

    # 同时保存为 JSON
    json_path = os.path.join(output_dir, 'datasets_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f'[OK] Summary JSON saved: {json_path}')

    return df


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'uploads', 'datasets')

    print('=' * 60)
    print('  Generating ML Datasets for AI Platform')
    print('=' * 60)

    records = generate_all_datasets(output_dir)

    print(f'\nGenerated {len(records)} datasets in: {output_dir}\n')

    # 汇总表
    df = save_summary_table(records, output_dir)
    print(f'\n{"="*80}')
    print(f'  Datasets Summary')
    print(f'{"="*80}')
    print(df[['name', 'rows', 'columns', 'task', 'target', 'difficulty']].to_string(index=False))
    print(f'\n[OK] All datasets generated successfully!')
    print(f'[INFO] Upload directory: {output_dir}')
