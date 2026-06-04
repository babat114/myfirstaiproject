#!/usr/bin/env python
"""
============================================
批量生成80个多类型训练数据集
8种模型类型 × 10个数据集 = 80个
大小: 2-20MB/个，总计 ~600MB
============================================
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.dataset import Dataset
from app.models.user import User


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ============================================================
# 数据集生成器定义
# ============================================================

def gen_classification_01():
    """二分类 - 金融欺诈检测"""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=15000, n_features=25, n_informative=18,
                                n_redundant=3, n_classes=2, weights=[0.85, 0.15],
                                flip_y=0.02, random_state=1)
    cols = [f'feature_{i}' for i in range(25)] + ['is_fraud']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df['is_fraud'] = df['is_fraud'].astype(int)
    return df, 'is_fraud', 'classification'

def gen_classification_02():
    """多分类 - 客户分级"""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=20000, n_features=20, n_informative=12,
                                n_classes=4, random_state=2)
    cols = [f'f{i}' for i in range(20)] + ['customer_tier']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df['customer_tier'] = df['customer_tier'].astype(int)
    return df, 'customer_tier', 'classification'

def gen_classification_03():
    """二分类 - 医疗诊断"""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=12000, n_features=30, n_informative=20,
                                n_classes=2, weights=[0.7, 0.3], random_state=3)
    cols = [f'biomarker_{i}' for i in range(30)] + ['diagnosis']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df['diagnosis'] = df['diagnosis'].astype(int)
    return df, 'diagnosis', 'classification'

def gen_classification_04():
    """多分类 - 产品类型预测"""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=18000, n_features=18, n_informative=14,
                                n_classes=5, random_state=4)
    cols = [f'attr_{i}' for i in range(18)] + ['product_type']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df['product_type'] = df['product_type'].astype(int)
    return df, 'product_type', 'classification'

def gen_classification_05():
    """二分类 - 信用评分"""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=25000, n_features=22, n_informative=16,
                                n_classes=2, weights=[0.8, 0.2], random_state=5)
    cols = [f'credit_feature_{i}' for i in range(22)] + ['default']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df['default'] = df['default'].astype(int)
    return df, 'default', 'classification'

def gen_classification_06():
    """多分类 - 图像场景识别"""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=16000, n_features=50, n_informative=35,
                                n_classes=6, random_state=6)
    cols = [f'pixel_feature_{i}' for i in range(50)] + ['scene_type']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df['scene_type'] = df['scene_type'].astype(int)
    return df, 'scene_type', 'classification'

def gen_classification_07():
    """二分类 - 邮件分类"""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=22000, n_features=40, n_informative=28,
                                n_classes=2, random_state=7)
    cols = [f'tfidf_{i}' for i in range(40)] + ['is_spam']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df['is_spam'] = df['is_spam'].astype(int)
    return df, 'is_spam', 'classification'

def gen_classification_08():
    """多分类 - 植物物种分类"""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=14000, n_features=12, n_informative=10,
                                n_classes=7, random_state=8)
    cols = [f'botanical_f{i}' for i in range(12)] + ['species']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df['species'] = df['species'].astype(int)
    return df, 'species', 'classification'

def gen_classification_09():
    """二分类 - 用户流失预测"""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=20000, n_features=16, n_informative=12,
                                n_classes=2, weights=[0.78, 0.22], random_state=9)
    cols = [f'user_feat_{i}' for i in range(16)] + ['churn']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df['churn'] = df['churn'].astype(int)
    return df, 'churn', 'classification'

def gen_classification_10():
    """多分类 - 手写数字"""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=30000, n_features=32, n_informative=24,
                                n_classes=8, random_state=10)
    cols = [f'digit_f{i}' for i in range(32)] + ['digit']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df['digit'] = df['digit'].astype(int)
    return df, 'digit', 'classification'


# === 回归数据集 ===
def gen_regression_01():
    """房价预测"""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=18000, n_features=15, n_informative=10,
                            noise=0.1, random_state=11)
    y = y * 250 + 500000 + np.random.normal(0, 30000, len(y))
    y = np.abs(y)
    cols = [f'house_feat_{i}' for i in range(15)] + ['price']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'price', 'regression'

def gen_regression_02():
    """股票收益预测"""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=20000, n_features=22, n_informative=16,
                            noise=0.15, random_state=12)
    y = y * 0.02 + 0.001
    cols = [f'factor_{i}' for i in range(22)] + ['daily_return']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'daily_return', 'regression'

def gen_regression_03():
    """能源消耗预测"""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=15000, n_features=18, n_informative=12,
                            noise=0.12, random_state=13)
    y = y * 100 + 500 + np.abs(np.random.normal(0, 50, len(y)))
    cols = [f'energy_f{i}' for i in range(18)] + ['consumption_kwh']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'consumption_kwh', 'regression'

def gen_regression_04():
    """保险理赔金额预测"""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=12000, n_features=20, n_informative=14,
                            noise=0.08, random_state=14)
    y = np.abs(y * 500 + 2000)
    cols = [f'insurance_f{i}' for i in range(20)] + ['claim_amount']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'claim_amount', 'regression'

def gen_regression_05():
    """销售额预测"""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=25000, n_features=14, n_informative=10,
                            noise=0.2, random_state=15)
    y = np.abs(y * 300 + 5000 + np.random.normal(0, 500, len(y)))
    cols = [f'sales_f{i}' for i in range(14)] + ['revenue']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'revenue', 'regression'

def gen_regression_06():
    """温度预测"""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=10000, n_features=12, n_informative=8,
                            noise=0.05, random_state=16)
    y = y * 10 + 20 + np.random.normal(0, 2, len(y))
    cols = [f'weather_f{i}' for i in range(12)] + ['temperature']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'temperature', 'regression'

def gen_regression_07():
    """汽车价格预测"""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=16000, n_features=16, n_informative=11,
                            noise=0.1, random_state=17)
    y = np.abs(y * 2000 + 25000 + np.random.normal(0, 3000, len(y)))
    cols = [f'car_f{i}' for i in range(16)] + ['price']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'price', 'regression'

def gen_regression_08():
    """作物产量预测"""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=14000, n_features=19, n_informative=13,
                            noise=0.14, random_state=18)
    y = np.abs(y * 1.5 + 5 + np.random.normal(0, 0.5, len(y)))
    cols = [f'crop_f{i}' for i in range(19)] + ['yield_tons']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'yield_tons', 'regression'

def gen_regression_09():
    """GDP增长率预测"""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=9000, n_features=24, n_informative=17,
                            noise=0.09, random_state=19)
    y = y * 0.03 + 0.025
    cols = [f'econ_f{i}' for i in range(24)] + ['gdp_growth']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'gdp_growth', 'regression'

def gen_regression_10():
    """广告ROI预测"""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=11000, n_features=10, n_informative=7,
                            noise=0.11, random_state=20)
    y = np.abs(y * 0.8 + 1.5)
    cols = [f'ad_channel_{i}' for i in range(10)] + ['roi']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'roi', 'regression'


# === 聚类数据集 ===
def gen_cluster_01():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=15000, n_features=10, centers=5, random_state=21)
    cols = [f'c_f{i}' for i in range(10)] + ['cluster']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'cluster', 'clustering'

def gen_cluster_02():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=20000, n_features=8, centers=3,
                       cluster_std=[0.8, 1.5, 2.5], random_state=22)
    cols = [f'coord_{i}' for i in range(8)] + ['group']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'group', 'clustering'

def gen_cluster_03():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=18000, n_features=12, centers=7, random_state=23)
    cols = [f'dim_{i}' for i in range(12)] + ['segment']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'segment', 'clustering'

def gen_cluster_04():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=12000, n_features=15, centers=4,
                       cluster_std=[0.5, 0.8, 1.5, 2.0], random_state=24)
    cols = [f'feat_{i}' for i in range(15)] + ['category']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'category', 'clustering'

def gen_cluster_05():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=25000, n_features=6, centers=8, random_state=25)
    cols = [f'point_{i}' for i in range(6)] + ['label']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'label', 'clustering'

def gen_cluster_06():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=10000, n_features=20, centers=6, random_state=26)
    cols = [f'metric_{i}' for i in range(20)] + ['class']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'class', 'clustering'

def gen_cluster_07():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=22000, n_features=9, centers=10, random_state=27)
    cols = [f'axis_{i}' for i in range(9)] + ['group_id']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'group_id', 'clustering'

def gen_cluster_08():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=17000, n_features=11, centers=4, random_state=28)
    cols = [f'var_{i}' for i in range(11)] + ['profile']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'profile', 'clustering'

def gen_cluster_09():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=13000, n_features=14, centers=5,
                       cluster_std=[0.6, 1.0, 1.4, 1.8, 2.2], random_state=29)
    cols = [f'attr_{i}' for i in range(14)] + ['type']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'type', 'clustering'

def gen_cluster_10():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=19000, n_features=8, centers=9, random_state=30)
    cols = [f'a_{i}' for i in range(8)] + ['target']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'target', 'clustering'


# === NLP 数据集 ===
def gen_nlp_text_data(n_samples, n_features, task='binary', seed=0):
    """生成模拟NLP文本特征数据"""
    np.random.seed(seed)
    vocab = [f'word_{i}' for i in range(n_features)]
    data = {}
    # TF-IDF like features (sparse, non-negative)
    density = 0.15  # 15% 非零
    for col in vocab:
        values = np.random.exponential(0.5, n_samples) * np.random.binomial(1, density, n_samples)
        data[col] = values
    # 添加文本元特征列
    data['text_length'] = np.random.randint(50, 5000, n_samples)
    data['avg_word_len'] = np.random.uniform(3, 8, n_samples)
    data['unique_words_ratio'] = np.random.uniform(0.1, 0.9, n_samples)

    if task == 'binary':
        y = (np.random.random(n_samples) > 0.5).astype(int)
        data['sentiment'] = y
        target = 'sentiment'
    elif task == 'multiclass':
        y = np.random.randint(0, 5, n_samples)
        data['topic'] = y
        target = 'topic'
    else:
        y = np.random.randint(0, 3, n_samples)
        data['category'] = y
        target = 'category'
    return pd.DataFrame(data), target, 'nlp'

def gen_nlp_01(): return gen_nlp_text_data(18000, 80, 'binary', 41)
def gen_nlp_02(): return gen_nlp_text_data(15000, 100, 'multiclass', 42)
def gen_nlp_03(): return gen_nlp_text_data(22000, 60, 'binary', 43)
def gen_nlp_04(): return gen_nlp_text_data(12000, 120, 'multiclass', 44)
def gen_nlp_05(): return gen_nlp_text_data(20000, 70, 'binary', 45)
def gen_nlp_06(): return gen_nlp_text_data(16000, 90, 'multiclass', 46)
def gen_nlp_07(): return gen_nlp_text_data(25000, 50, 'binary', 47)
def gen_nlp_08(): return gen_nlp_text_data(14000, 110, 'multiclass', 48)
def gen_nlp_09(): return gen_nlp_text_data(10000, 130, 'binary', 49)
def gen_nlp_10(): return gen_nlp_text_data(19000, 75, 'multiclass', 50)


# === CV 数据集 ===
def gen_cv_data(n_samples, img_dim=32, task='binary', seed=0):
    """生成模拟CV图像特征 (展平像素值)"""
    np.random.seed(seed)
    n_features = img_dim * img_dim
    # 模拟像素值
    X = np.random.randint(0, 256, (n_samples, n_features)).astype(float)
    # 添加空间结构 (相邻像素有相关性)
    for i in range(min(n_features - 1, 100)):
        X[:, i + 1] = X[:, i] * 0.3 + X[:, i + 1] * 0.7

    cols = [f'pixel_{i}' for i in range(n_features)]
    if task == 'binary':
        y = (X[:, :10].mean(axis=1) > 128).astype(int)
        cols.append('object_present')
        target = 'object_present'
    else:
        y = (X[:, :10].mean(axis=1) // 40).astype(int)
        cols.append('image_class')
        target = 'image_class'
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, target, 'computer_vision'

def gen_cv_01(): return gen_cv_data(10000, 28, 'binary', 51)
def gen_cv_02(): return gen_cv_data(8000, 32, 'multiclass', 52)
def gen_cv_03(): return gen_cv_data(12000, 24, 'binary', 53)
def gen_cv_04(): return gen_cv_data(9000, 36, 'multiclass', 54)
def gen_cv_05(): return gen_cv_data(11000, 20, 'binary', 55)
def gen_cv_06(): return gen_cv_data(7000, 40, 'multiclass', 56)
def gen_cv_07(): return gen_cv_data(13000, 16, 'binary', 57)
def gen_cv_08(): return gen_cv_data(6000, 48, 'multiclass', 58)
def gen_cv_09(): return gen_cv_data(15000, 14, 'binary', 59)
def gen_cv_10(): return gen_cv_data(5000, 64, 'multiclass', 60)


# === RL 数据集 ===
def gen_rl_data(n_samples, seed=0):
    """生成模拟RL状态-动作-奖励数据"""
    np.random.seed(seed)
    n_states = np.random.randint(15, 30)
    data = {}
    for i in range(n_states):
        data[f'state_{i}'] = np.random.uniform(-1, 1, n_samples)
    data['action'] = np.random.randint(0, 4, n_samples)  # 4种动作
    data['reward'] = np.random.uniform(-10, 100, n_samples)
    data['next_state_value'] = np.random.uniform(0, 50, n_samples)
    data['done'] = (np.random.random(n_samples) > 0.9).astype(int)
    df = pd.DataFrame(data)
    return df, 'action', 'reinforcement'

def gen_rl_01(): return gen_rl_data(20000, 61)
def gen_rl_02(): return gen_rl_data(18000, 62)
def gen_rl_03(): return gen_rl_data(22000, 63)
def gen_rl_04(): return gen_rl_data(15000, 64)
def gen_rl_05(): return gen_rl_data(25000, 65)
def gen_rl_06(): return gen_rl_data(12000, 66)
def gen_rl_07(): return gen_rl_data(19000, 67)
def gen_rl_08(): return gen_rl_data(16000, 68)
def gen_rl_09(): return gen_rl_data(21000, 69)
def gen_rl_10(): return gen_rl_data(14000, 70)


# === Generative 数据集 ===
def gen_generative_data(n_samples, seed=0):
    """生成模拟生成模型训练数据"""
    np.random.seed(seed)
    n_features = np.random.randint(30, 50)
    data = {}
    for i in range(n_features):
        # 混合高斯分布模拟潜在空间
        mode = np.random.randint(0, 3)
        data[f'latent_{i}'] = np.random.normal(mode * 2, 0.8, n_samples)
    data['is_real'] = 1  # 所有都是真实样本
    # 添加噪声特征模拟生成质量评估
    data['quality_score'] = np.clip(np.random.normal(0.8, 0.15, n_samples), 0, 1)
    df = pd.DataFrame(data)
    return df, 'quality_score', 'generative'

def gen_gen_01(): return gen_generative_data(15000, 71)
def gen_gen_02(): return gen_generative_data(18000, 72)
def gen_gen_03(): return gen_generative_data(12000, 73)
def gen_gen_04(): return gen_generative_data(20000, 74)
def gen_gen_05(): return gen_generative_data(10000, 75)
def gen_gen_06(): return gen_generative_data(22000, 76)
def gen_gen_07(): return gen_generative_data(14000, 77)
def gen_gen_08(): return gen_generative_data(17000, 78)
def gen_gen_09(): return gen_generative_data(13000, 79)
def gen_gen_10(): return gen_generative_data(16000, 80)


# === Other 数据集 ===
def gen_other_01():
    """时序数据"""
    n = 12000
    t = np.arange(n)
    data = {'timestamp': t}
    for i in range(15):
        data[f'series_{i}'] = np.sin(t * (i + 1) * 0.01) + np.random.normal(0, 0.3, n)
    data['anomaly'] = (np.abs(np.random.normal(0, 1, n)) > 2.5).astype(int)
    df = pd.DataFrame(data)
    return df, 'anomaly', 'other'

def gen_other_02():
    """推荐系统数据"""
    n = 15000
    df = pd.DataFrame({
        'user_id': np.random.randint(1, 1001, n),
        'item_id': np.random.randint(1, 2001, n),
        'rating': np.random.randint(1, 6, n),
        'timestamp': np.random.randint(1600000000, 1700000000, n),
    })
    for i in range(8):
        df[f'user_feat_{i}'] = np.random.uniform(0, 1, n)
        df[f'item_feat_{i}'] = np.random.uniform(0, 1, n)
    return df, 'rating', 'other'

def gen_other_03():
    """异常检测数据"""
    n = 10000
    X = np.random.normal(0, 1, (n, 20))
    y = np.zeros(n, dtype=int)
    # 5% 异常
    anomaly_idx = np.random.choice(n, int(n * 0.05), replace=False)
    X[anomaly_idx] += np.random.normal(0, 5, (len(anomaly_idx), 20))
    y[anomaly_idx] = 1
    cols = [f'feature_{i}' for i in range(20)] + ['is_anomaly']
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    return df, 'is_anomaly', 'other'

def gen_other_04():
    """生存分析数据"""
    n = 8000
    df = pd.DataFrame({
        'age': np.random.randint(18, 90, n),
        'treatment': np.random.binomial(1, 0.5, n),
        'biomarker_a': np.random.normal(100, 20, n),
        'biomarker_b': np.random.exponential(50, n),
    })
    for i in range(8):
        df[f'gene_{i}'] = np.random.binomial(1, 0.3, n)
    df['survival_days'] = np.random.exponential(365 * 3, n).astype(int)
    df['event'] = np.random.binomial(1, 0.4, n)
    return df, 'survival_days', 'other'

def gen_other_05():
    """多标签分类数据"""
    n = 12000
    X = np.random.normal(0, 1, (n, 15))
    cols = [f'feature_{i}' for i in range(15)]
    df = pd.DataFrame(X, columns=cols)
    for j in range(5):
        df[f'label_{j}'] = np.random.binomial(1, 0.3, n)
    return df, 'label_0', 'other'

def gen_other_06():
    """时间序列预测"""
    n = 20000
    df = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=n, freq='h')})
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    for i in range(10):
        df[f'metric_{i}'] = np.sin(np.arange(n) * (i + 1) * 0.001) + np.random.normal(0, 0.2, n)
    df['target_value'] = df[[f'metric_{i}' for i in range(5)]].mean(axis=1) + np.random.normal(0, 0.1, n)
    return df, 'target_value', 'other'

def gen_other_07():
    """图数据特征"""
    n = 10000
    df = pd.DataFrame({'node_id': range(n)})
    df['degree'] = np.random.zipf(2.0, n)
    df['clustering_coef'] = np.random.beta(2, 5, n)
    df['betweenness'] = np.random.exponential(0.1, n)
    for i in range(8):
        df[f'embedding_{i}'] = np.random.normal(0, 1, n)
    df['community'] = np.random.randint(0, 6, n)
    return df, 'community', 'other'

def gen_other_08():
    """A/B测试数据"""
    n = 15000
    df = pd.DataFrame({
        'group': np.random.binomial(1, 0.5, n),
        'visitor_type': np.random.choice(['new', 'returning', 'loyal'], n),
        'device': np.random.choice(['mobile', 'desktop', 'tablet'], n),
        'session_duration': np.random.exponential(120, n),
        'pages_viewed': np.random.poisson(5, n),
    })
    for i in range(6):
        df[f'behavior_{i}'] = np.random.uniform(0, 1, n)
    df['conversion'] = np.random.binomial(1, 0.08, n)
    return df, 'conversion', 'other'

def gen_other_09():
    """音频特征数据"""
    n = 8000
    df = pd.DataFrame()
    for i in range(20):
        df[f'mfcc_{i}'] = np.random.normal(0, 1, n)
    df['spectral_centroid'] = np.random.uniform(500, 4000, n)
    df['zero_crossing_rate'] = np.random.uniform(0.01, 0.3, n)
    df['rms_energy'] = np.random.uniform(0.001, 0.5, n)
    df['tempo'] = np.random.uniform(60, 180, n)
    df['genre'] = np.random.randint(0, 8, n)
    return df, 'genre', 'other'

def gen_other_10():
    """混合特征数据集 (数值+分类+文本)"""
    n = 10000
    df = pd.DataFrame()
    # 数值特征
    for i in range(15):
        df[f'num_{i}'] = np.random.normal(0, 1 + i * 0.2, n)
    # 分类特征
    df['category_a'] = np.random.choice(['A', 'B', 'C', 'D'], n)
    df['category_b'] = np.random.choice(['X', 'Y', 'Z'], n)
    df['category_c'] = np.random.choice(['high', 'medium', 'low'], n, p=[0.2, 0.5, 0.3])
    # 文本特征
    df['title_length'] = np.random.randint(5, 200, n)
    df['description_words'] = np.random.poisson(50, n)
    # 目标
    df['target_score'] = np.random.uniform(0, 100, n)
    return df, 'target_score', 'other'


# ============================================================
# 注册所有生成器
# ============================================================
TYPE_GENERATORS = {
    'classification': [gen_classification_01, gen_classification_02, gen_classification_03,
                        gen_classification_04, gen_classification_05, gen_classification_06,
                        gen_classification_07, gen_classification_08, gen_classification_09,
                        gen_classification_10],
    'regression': [gen_regression_01, gen_regression_02, gen_regression_03,
                    gen_regression_04, gen_regression_05, gen_regression_06,
                    gen_regression_07, gen_regression_08, gen_regression_09,
                    gen_regression_10],
    'clustering': [gen_cluster_01, gen_cluster_02, gen_cluster_03,
                    gen_cluster_04, gen_cluster_05, gen_cluster_06,
                    gen_cluster_07, gen_cluster_08, gen_cluster_09,
                    gen_cluster_10],
    'nlp': [gen_nlp_01, gen_nlp_02, gen_nlp_03, gen_nlp_04, gen_nlp_05,
            gen_nlp_06, gen_nlp_07, gen_nlp_08, gen_nlp_09, gen_nlp_10],
    'computer_vision': [gen_cv_01, gen_cv_02, gen_cv_03, gen_cv_04, gen_cv_05,
                         gen_cv_06, gen_cv_07, gen_cv_08, gen_cv_09, gen_cv_10],
    'reinforcement': [gen_rl_01, gen_rl_02, gen_rl_03, gen_rl_04, gen_rl_05,
                       gen_rl_06, gen_rl_07, gen_rl_08, gen_rl_09, gen_rl_10],
    'generative': [gen_gen_01, gen_gen_02, gen_gen_03, gen_gen_04, gen_gen_05,
                    gen_gen_06, gen_gen_07, gen_gen_08, gen_gen_09, gen_gen_10],
    'other': [gen_other_01, gen_other_02, gen_other_03, gen_other_04, gen_other_05,
              gen_other_06, gen_other_07, gen_other_08, gen_other_09, gen_other_10],
}

TYPE_NAMES_CN = {
    'classification': '分类', 'regression': '回归', 'clustering': '聚类',
    'nlp': '自然语言处理', 'computer_vision': '计算机视觉',
    'reinforcement': '强化学习', 'generative': '生成模型', 'other': '其他',
}

DATASET_NAMES = {
    'classification': ['金融欺诈检测', '客户分级', '医疗诊断', '产品类型预测',
                        '信用评分', '图像场景识别', '邮件分类', '植物物种分类',
                        '用户流失预测', '手写数字识别'],
    'regression': ['房价预测', '股票收益预测', '能源消耗预测', '保险理赔金额',
                    '销售额预测', '温度预测', '汽车价格预测', '作物产量预测',
                    'GDP增长率预测', '广告ROI预测'],
    'clustering': ['客户分群', '市场细分', '用户画像', '行为聚类',
                    '文本主题聚类', '基因表达聚类', '社交网络分组', '产品分类',
                    '图像色彩聚类', '异常模式聚类'],
    'nlp': ['情感分析', '新闻主题分类', '垃圾评论检测', '意图识别',
             '文本相似度', '命名实体识别', '文档分类', '语义角色标注',
             '情感强度预测', '话题聚类'],
    'computer_vision': ['手写数字识别', '物体检测', '场景分类', '人脸识别',
                         '图像分割', '姿态估计', '纹理分类', '颜色识别',
                         '边缘检测', '深度估计'],
    'reinforcement': ['游戏策略学习', '机器人导航', '推荐策略优化', '资源调度',
                       '交通信号控制', '投资组合优化', '库存管理', '路径规划',
                       '对话策略学习', '能效优化'],
    'generative': ['图像生成', '文本生成', '音乐生成', '数据增强',
                    '风格迁移', '超分辨率', '图像修复', '语音合成',
                    '分子生成', '视频预测'],
    'other': ['时序异常检测', '推荐系统', '异常值检测', '生存分析',
               '多标签分类', '时间序列预测', '图数据分析', 'A/B测试',
               '音频分类', '混合特征分析'],
}


def compute_summary(df, target_col):
    """计算数据集摘要信息"""
    try:
        summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'target_column': target_col,
            'numeric_cols': int(df.select_dtypes(include=[np.number]).shape[1]),
            'categorical_cols': int(df.select_dtypes(include=['object', 'category']).shape[1]),
            'missing_total': int(df.isnull().sum().sum()),
            'column_names': list(df.columns),
        }
        return json.dumps(summary, ensure_ascii=False)
    except Exception:
        return json.dumps({'rows': len(df), 'columns': len(df.columns), 'target_column': target_col})


def main():
    app = create_app()

    with app.app_context():
        # 获取第一个admin用户
        admin = User.query.filter_by(role='admin').first()
        if not admin:
            print("[FAIL] 没有找到admin用户，请先初始化数据库并创建用户")
            return

        output_dir = os.path.join('uploads', 'datasets')
        ensure_dir(output_dir)

        total_generated = 0
        total_skipped = 0

        for model_type, generators in TYPE_GENERATORS.items():
            type_name = TYPE_NAMES_CN.get(model_type, model_type)
            names = DATASET_NAMES.get(model_type, [f'{type_name}_{i}' for i in range(10)])

            for i, gen_func in enumerate(generators):
                ds_name = f"[{type_name}] {names[i]}"

                # 检查是否已存在
                existing = Dataset.query.filter_by(name=ds_name).first()
                if existing:
                    print(f"[SKIP] 跳过已存在: {ds_name}")
                    total_skipped += 1
                    continue

                try:
                    print(f"[GEN] 生成: {ds_name} ...", end=' ')
                    df, target_col, task_type = gen_func()

                    # 保存CSV
                    filename = f"{model_type}_{i + 1:02d}_{len(df)}rows.csv"
                    file_path = os.path.join(output_dir, filename)
                    df.to_csv(file_path, index=False)

                    file_size = os.path.getsize(file_path)
                    file_size_mb = round(file_size / (1024 * 1024), 2)

                    # 创建Dataset记录
                    dataset = Dataset(
                        name=ds_name,
                        description=f'自动生成的{type_name}数据集 #{i + 1}',
                        file_path=file_path,
                        file_format='csv',
                        file_size=file_size,
                        category=model_type,
                        status='ready',
                        owner_id=admin.id,
                        is_public=True,
                        row_count=len(df),
                        column_count=len(df.columns),
                        summary_json=compute_summary(df, target_col),
                        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                    )
                    db.session.add(dataset)
                    db.session.commit()

                    print(f"[OK] {len(df):,}行 × {len(df.columns)}列 | {file_size_mb:.1f}MB")
                    total_generated += 1

                except Exception as e:
                    db.session.rollback()
                    print(f"[FAIL] 失败: {e}")
                    continue

        print(f"\n{'='*60}")
        print(f"完成! 新增: {total_generated} 个, 跳过: {total_skipped} 个")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
