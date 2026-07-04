"""
采集真实世界ML数据集 — 多来源自动下载

为每个算法类型匹配2-3个真实应用场景的数据集:
  - sklearn 内置数据集 (fetch_openml, load_iris, load_digits 等)
  - UCI ML Repository 直接下载
  - OpenML 通过 sklearn.datasets.fetch_openml

输出: 标准CSV文件到 uploads/datasets/

Usage:
    python scripts/fetch_real_datasets.py --dry-run --verbose
    python scripts/fetch_real_datasets.py --verbose
    python scripts/fetch_real_datasets.py --algo random_forest
    python scripts/fetch_real_datasets.py --source sklearn  # 仅sklearn内置
"""

import os
import sys
import json
import hashlib
import time
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import (
    PROJECT_ROOT, create_base_parser, app_context, setup_verbose
)

# ═══════════════════════════════════════════════════════════════
# 数据集清单 — 每个算法 2-3 个真实应用场景
# ═══════════════════════════════════════════════════════════════

DATASET_MANIFEST = [
    # ── 分类 (Classification) ──
    # === 随机森林 (random_forest) ===
    {
        'key': 'german_credit',
        'display_name': '德国信用卡评分',
        'category': 'classification',
        'target': 'class',
        'algorithms': ['random_forest', 'gradient_boosting', 'logistic_regression', 'mlp'],
        'source': 'openml',
        'openml_id': 31,
        'domain': '金融风控',
        'description': '预测客户信用好坏，1000样本20特征，含类别型和数值型混合',
    },
    {
        'key': 'telco_churn',
        'display_name': '电信客户流失',
        'category': 'classification',
        'target': 'Churn',
        'algorithms': ['random_forest', 'gradient_boosting', 'decision_tree', 'mlp'],
        'source': 'openml',
        'openml_id': 42178,  # Telco Customer Churn on OpenML
        'domain': '客户留存',
        'description': '预测电信客户是否流失，7043样本21特征，类别不平衡',
    },
    {
        'key': 'mushroom',
        'display_name': '蘑菇毒性分类',
        'category': 'classification',
        'target': 'class',
        'algorithms': ['random_forest', 'decision_tree', 'gradient_boosting'],
        'source': 'openml',
        'openml_id': 24,
        'domain': '食品安全',
        'description': '根据形态特征判断蘑菇是否有毒，8124样本22分类特征',
    },

    # === 逻辑回归 (logistic_regression) ===
    {
        'key': 'spambase',
        'display_name': '垃圾邮件识别',
        'category': 'classification',
        'target': 'class',
        'algorithms': ['logistic_regression', 'random_forest', 'gradient_boosting', 'mlp'],
        'source': 'openml',
        'openml_id': 44,
        'domain': '邮件过滤',
        'description': '基于词频和字符频率识别垃圾邮件，4601样本57特征',
    },
    {
        'key': 'pima_diabetes',
        'display_name': '皮马印第安人糖尿病',
        'category': 'classification',
        'target': 'class',
        'algorithms': ['logistic_regression', 'knn', 'decision_tree'],
        'source': 'openml',
        'openml_id': 37,
        'domain': '医疗诊断',
        'description': 'Pima印第安女性糖尿病筛查，768样本8特征，医学经典数据集',
    },
    {
        'key': 'credit_approval',
        'display_name': '信用卡审批',
        'category': 'classification',
        'target': 'class',
        'algorithms': ['logistic_regression', 'random_forest', 'svm'],
        'source': 'openml',
        'openml_id': 29,
        'domain': '信用评估',
        'description': '信用卡申请审批决策，690样本15混合特征',
    },

    # === SVM (svm) ===
    {
        'key': 'wine_quality_binary',
        'display_name': '葡萄酒品质分类',
        'category': 'classification',
        'target': 'quality',
        'algorithms': ['svm', 'random_forest', 'knn', 'mlp'],
        'source': 'openml',
        'openml_id': 287,  # Wine quality (combined red+white, 6497 rows, target=quality)
        'domain': '食品检测',
        'description': '理化指标预测葡萄酒品质等级，4898样本11特征',
    },

    # === KNN (knn) ===
    {
        'key': 'glass',
        'display_name': '玻璃类型识别',
        'category': 'classification',
        'target': 'Type',
        'algorithms': ['knn', 'svm', 'random_forest', 'mlp'],
        'source': 'openml',
        'openml_id': 41,
        'domain': '法医学',
        'description': '根据折射率和元素含量识别玻璃类型，214样本9特征',
    },
    {
        'key': 'vehicle',
        'display_name': '车辆轮廓分类',
        'category': 'classification',
        'target': 'Class',
        'algorithms': ['knn', 'svm', 'random_forest', 'mlp'],
        'source': 'openml',
        'openml_id': 54,
        'domain': '计算机视觉',
        'description': '根据轮廓特征识别车辆类型，846样本18特征',
    },

    # === 梯度提升 (gradient_boosting) ===
    {
        'key': 'adult',
        'display_name': '成人收入预测',
        'category': 'classification',
        'target': 'class',
        'algorithms': ['gradient_boosting', 'random_forest', 'mlp'],
        'source': 'openml',
        'openml_id': 1590,
        'domain': '社会经济',
        'description': '根据人口统计特征预测年收入>50K，48842样本14特征',
    },
    {
        'key': 'bank_marketing',
        'display_name': '银行营销电话',
        'category': 'classification',
        'target': 'Class',
        'algorithms': ['gradient_boosting', 'random_forest', 'logistic_regression', 'mlp'],
        'source': 'openml',
        'openml_id': 1461,
        'domain': '市场营销',
        'description': '葡萄牙银行电话营销成功预测，45211样本16特征',
    },

    # === 决策树 (decision_tree) ===
    {
        'key': 'car_eval',
        'display_name': '汽车评估',
        'category': 'classification',
        'target': 'class',
        'algorithms': ['decision_tree', 'random_forest', 'gradient_boosting'],
        'source': 'openml',
        'openml_id': 21,
        'domain': '消费决策',
        'description': '基于价格/安全/容量评估汽车可接受度，1728样本6分类特征',
    },
    {
        'key': 'titanic',
        'display_name': '泰坦尼克号生存预测',
        'category': 'classification',
        'target': 'survived',
        'algorithms': ['decision_tree', 'random_forest', 'logistic_regression'],
        'source': 'openml',
        'openml_id': 40945,
        'domain': '历史分析',
        'description': '泰坦尼克乘客生存预测，1309样本10特征',
    },

    # === MLP (多层感知机) ===
    {
        'key': 'mnist_small',
        'display_name': '手写数字MNIST',
        'category': 'classification',
        'target': 'class',
        'algorithms': ['mlp', 'knn', 'svm'],
        'source': 'openml',
        'openml_id': 554,  # MNIST on OpenML
        'domain': '图像识别',
        'description': '手写数字0-9识别，70000样本784像素特征',
    },

    # ── 回归 (Regression) ──
    # === 线性回归 (linear_regression) ===
    {
        'key': 'auto_mpg',
        'display_name': '汽车油耗',
        'category': 'regression',
        'target': 'mpg',
        'algorithms': ['linear_regression', 'random_forest_regressor', 'ridge'],
        'source': 'openml',
        'openml_id': 43574,  # auto-mpg on OpenML
        'domain': '汽车工业',
        'description': '根据发动机参数预测油耗MPG，398样本7特征',
    },
    {
        'key': 'student_performance',
        'display_name': '学生成绩预测',
        'category': 'regression',
        'target': 'G3',
        'algorithms': ['linear_regression', 'random_forest_regressor', 'ridge'],
        'source': 'openml',
        'openml_id': 42352,  # Student Performance Math (395 rows, 33 features, target=G3)
        'domain': '教育评估',
        'description': '基于家庭/学校因素预测期末成绩，649样本32特征',
    },

    # === Ridge回归 ===
    {
        'key': 'bodyfat',
        'display_name': '体脂率预测',
        'category': 'regression',
        'target': 'class',  # OpenML 560 目标列名为 class (体脂率百分比)
        'algorithms': ['ridge', 'linear_regression', 'random_forest_regressor'],
        'source': 'openml',
        'openml_id': 560,  # Bodyfat prediction (252 rows, 14 features, target=BodyFat)
        'domain': '健康管理',
        'description': '根据身体测量数据预测体脂率，252样本14特征',
    },
    {
        'key': 'concrete',
        'display_name': '混凝土强度',
        'category': 'regression',
        'target': 'strength',
        'algorithms': ['ridge', 'random_forest_regressor', 'gradient_boosting_regressor'],
        'source': 'openml',
        'openml_id': 43570,  # Concrete Compressive Strength
        'domain': '土木工程',
        'description': '根据配比预测混凝土抗压强度，1030样本8特征',
    },

    # === SVR ===
    {
        'key': 'energy_efficiency',
        'display_name': '建筑能源效率',
        'category': 'regression',
        'target': 'Heating_Load',
        'algorithms': ['svr', 'random_forest_regressor', 'gradient_boosting_regressor'],
        'source': 'openml',
        'openml_id': 43383,  # Energy efficiency
        'domain': '绿色建筑',
        'description': '根据建筑参数预测供暖负荷，768样本8特征',
    },
    {
        'key': 'air_quality',
        'display_name': '空气质量预测',
        'category': 'regression',
        'target': 'CO_GT_',
        'algorithms': ['svr', 'random_forest_regressor', 'gradient_boosting_regressor'],
        'source': 'synth',  # OpenML ID 错误 (返回Titanic), 用合成数据替代
        'domain': '环境监测',
        'description': '预测城市空气CO浓度，8991样本13特征',
    },

    # === 随机森林回归 (random_forest_regressor) ===
    {
        'key': 'california_housing',
        'display_name': '加州房价',
        'category': 'regression',
        'target': 'MedHouseVal',
        'algorithms': ['random_forest_regressor', 'gradient_boosting_regressor', 'linear_regression'],
        'source': 'sklearn',
        'sklearn_name': 'california_housing',
        'domain': '房地产',
        'description': '加州街区房价中位数预测，20640样本8特征',
    },
    {
        'key': 'boston_housing',
        'display_name': '波士顿房价',
        'category': 'regression',
        'target': 'MEDV',
        'algorithms': ['random_forest_regressor', 'linear_regression', 'ridge'],
        'source': 'openml',
        'openml_id': 43389,
        'domain': '城市规划',
        'description': '波士顿郊区房价预测，506样本13特征',
    },

    # === 梯度提升回归 (gradient_boosting_regressor) ===
    {
        'key': 'cpu_performance',
        'display_name': 'CPU性能',
        'category': 'regression',
        'target': 'class',
        'algorithms': ['gradient_boosting_regressor', 'random_forest_regressor', 'svr'],
        'source': 'openml',
        'openml_id': 43391,  # cpu
        'domain': '计算机硬件',
        'description': '根据CPU参数预测相对性能，209样本7特征',
    },
    {
        'key': 'wine_quality_red',
        'display_name': '红葡萄酒品质评分',
        'category': 'regression',
        'target': 'quality',
        'algorithms': ['gradient_boosting_regressor', 'random_forest_regressor', 'svr'],
        'source': 'openml',
        'openml_id': 43387,  # wine-quality-red
        'domain': '食品检测',
        'description': '理化指标预测红葡萄酒品质评分(0-10)，1599样本11特征',
    },

    # ── 聚类 (Clustering) ──
    # === KMeans ===
    {
        'key': 'seeds',
        'display_name': '小麦种子聚类',
        'category': 'clustering',
        'target': 'Type',  # 用于评估，训练时移除
        'algorithms': ['kmeans', 'agglomerative', 'minibatch_kmeans'],
        'source': 'openml',
        'openml_id': 1495,  # seeds
        'domain': '农业',
        'description': '根据几何特征聚类小麦品种，210样本7特征，3个自然簇',
    },
    {
        'key': 'ecoli',
        'display_name': '大肠杆菌蛋白质定位',
        'category': 'clustering',
        'target': 'class',
        'algorithms': ['kmeans', 'agglomerative', 'dbscan'],
        'source': 'openml',
        'openml_id': 39,
        'domain': '生物信息学',
        'description': '蛋白质细胞定位聚类，336样本7特征',
    },

    # === DBSCAN ===
    {
        'key': 'blobs_varied',
        'display_name': '变密度团状聚类',
        'category': 'clustering',
        'target': 'label',
        'algorithms': ['dbscan', 'kmeans', 'agglomerative'],
        'source': 'sklearn',
        'sklearn_name': 'blobs_varied',
        'domain': '合成基准',
        'description': '变密度高斯团，3个不同密度簇，适合DBSCAN密度聚类',
    },
    {
        'key': 'moons',
        'display_name': '双月牙形聚类',
        'category': 'clustering',
        'target': 'label',
        'algorithms': ['dbscan', 'agglomerative', 'kmeans'],
        'source': 'sklearn',
        'sklearn_name': 'moons',
        'domain': '合成基准',
        'description': '双月牙形非凸聚类，500样本，DBSCAN优势场景',
    },

    # ── 补充数据集（合成生成 — 网络受限时替代方案）──
    # === 回归补充 ===
    {
        'key': 'diabetes_regression',
        'display_name': '糖尿病进展回归',
        'category': 'regression',
        'target': 'disease_progression',
        'algorithms': ['linear_regression', 'ridge', 'random_forest_regressor'],
        'source': 'synth',
        'domain': '医疗健康',
        'description': '糖尿病进展指标预测，442样本10特征',
    },
    {
        'key': 'concrete',
        'display_name': '混凝土强度预测',
        'category': 'regression',
        'target': 'strength',
        'algorithms': ['ridge', 'random_forest_regressor', 'gradient_boosting_regressor'],
        'source': 'synth',
        'domain': '土木工程',
        'description': '混凝土配比→抗压强度，1030样本8特征',
    },
    {
        'key': 'energy_efficiency',
        'display_name': '建筑能源效率',
        'category': 'regression',
        'target': 'Heating_Load',
        'algorithms': ['svr', 'random_forest_regressor', 'gradient_boosting_regressor'],
        'source': 'synth',
        'domain': '绿色建筑',
        'description': '建筑参数→供暖负荷预测，768样本8特征',
    },
    {
        'key': 'cpu_performance',
        'display_name': 'CPU性能预测',
        'category': 'regression',
        'target': 'class',
        'algorithms': ['gradient_boosting_regressor', 'random_forest_regressor', 'svr'],
        'source': 'synth',
        'domain': '计算机硬件',
        'description': 'CPU参数→相对性能，209样本7特征',
    },
    {
        'key': 'wine_quality_red',
        'display_name': '红酒品质评分',
        'category': 'regression',
        'target': 'quality',
        'algorithms': ['gradient_boosting_regressor', 'random_forest_regressor', 'svr'],
        'source': 'synth',
        'domain': '食品检测',
        'description': '理化指标→品质评分，1599样本11特征',
    },

    # === 聚类补充 ===
    {
        'key': 'mall_customers',
        'display_name': '商场客户分群',
        'category': 'clustering',
        'target': 'Segment',
        'algorithms': ['kmeans', 'agglomerative', 'minibatch_kmeans'],
        'source': 'synth',
        'domain': '市场营销',
        'description': '客户5群分群，消费/收入/频次特征，500样本6特征',
    },
    {
        'key': 'seeds',
        'display_name': '小麦种子聚类',
        'category': 'clustering',
        'target': 'Type',
        'algorithms': ['kmeans', 'agglomerative'],
        'source': 'synth',
        'domain': '农业科技',
        'description': '种子几何特征3簇聚类，210样本7特征',
    },
    {
        'key': 'moons_dense',
        'display_name': '月牙形密集聚类',
        'category': 'clustering',
        'target': 'label',
        'algorithms': ['dbscan', 'agglomerative'],
        'source': 'synth',
        'domain': '非凸聚类',
        'description': '双月牙低噪声聚类，300样本，DBSCAN强项',
    },
    {
        'key': 'circles',
        'display_name': '同心圆聚类',
        'category': 'clustering',
        'target': 'label',
        'algorithms': ['dbscan', 'agglomerative'],
        'source': 'synth',
        'domain': '非凸聚类',
        'description': '内外环聚类，300样本，密度聚类专属场景',
    },

    # ═══════════════════════════════════════════════════════════════
    # 新增 20 个真实数据集 (2026-07-01)
    # ═══════════════════════════════════════════════════════════════

    # ── 金融风控 ──
    {
        'key': 'electricity',
        'display_name': '电力市场价格',
        'category': 'classification',
        'target': 'class',
        'algorithms': ['random_forest', 'gradient_boosting', 'logistic_regression', 'mlp'],
        'source': 'openml',
        'openml_id': 151,
        'domain': '能源金融',
        'description': '澳大利亚电力市场价格波动预测，45312样本8特征',
    },

    # ── 医疗健康 ──
    {
        'key': 'breast_wisconsin',
        'display_name': '威斯康星乳腺癌诊断',
        'category': 'classification',
        'target': 'Class',
        'algorithms': ['logistic_regression', 'svm', 'random_forest', 'mlp'],
        'source': 'openml',
        'openml_id': 15,
        'domain': '医疗诊断',
        'description': '乳腺癌良恶性诊断，699样本9特征，最经典医学数据集',
    },
    {
        'key': 'dermatology',
        'display_name': '皮肤病分类',
        'category': 'classification',
        'target': 'class',
        'algorithms': ['random_forest', 'knn', 'svm'],
        'source': 'openml',
        'openml_id': 35,
        'domain': '医疗诊断',
        'description': '6类皮肤病鉴别诊断，366样本34特征',
    },
    {
        'key': 'heart_disease',
        'display_name': '心脏病预测',
        'category': 'classification',
        'target': 'class',
        'algorithms': ['logistic_regression', 'random_forest', 'gradient_boosting'],
        'source': 'openml',
        'openml_id': 53,  # Cleveland Heart Disease (processed)
        'domain': '医疗诊断',
        'description': '克利夫兰心脏病诊断，303样本13特征',
    },

    # ── 工业工程 ──
    {
        'key': 'steel_plates',
        'display_name': '钢板缺陷检测',
        'category': 'classification',
        'target': 'Class',
        'algorithms': ['random_forest', 'gradient_boosting', 'svm', 'mlp'],
        'source': 'openml',
        'openml_id': 1504,
        'domain': '工业制造',
        'description': '钢板7类表面缺陷识别，1941样本27特征',
    },
    {
        'key': 'qsar_biodeg',
        'display_name': '化学品生物降解',
        'category': 'classification',
        'target': 'Class',
        'algorithms': ['random_forest', 'decision_tree', 'gradient_boosting', 'mlp'],
        'source': 'openml',
        'openml_id': 1494,
        'domain': '化学工程',
        'description': '化学物质生物降解能力预测，1055样本41特征',
    },
    {
        'key': 'airfoil_noise',
        'display_name': '机翼噪声预测',
        'category': 'regression',
        'target': 'class',
        'algorithms': ['random_forest_regressor', 'gradient_boosting_regressor', 'linear_regression'],
        'source': 'openml',
        'openml_id': 43378,  # Airfoil Self-Noise
        'domain': '航空工程',
        'description': 'NASA机翼气动噪声预测，1503样本5特征',
    },

    # ── 环境科学 ──
    {
        'key': 'ozone_level',
        'display_name': '臭氧浓度检测',
        'category': 'classification',
        'target': 'Class',
        'algorithms': ['random_forest', 'gradient_boosting', 'logistic_regression', 'mlp'],
        'source': 'openml',
        'openml_id': 1487,
        'domain': '环境监测',
        'description': '8小时臭氧浓度超标预测，2536样本72特征',
    },
    {
        'key': 'forest_fires',
        'display_name': '森林火灾面积预测',
        'category': 'regression',
        'target': 'area',
        'algorithms': ['random_forest_regressor', 'svr', 'gradient_boosting_regressor'],
        'source': 'openml',
        'openml_id': 43803,  # Forest Fires (burned area)
        'domain': '环境科学',
        'description': '葡萄牙森林火灾烧毁面积预测，517样本12特征',
    },

    # ── 交通出行 ──
    {
        'key': 'bike_sharing',
        'display_name': '共享单车需求',
        'category': 'regression',
        'target': 'count',
        'algorithms': ['random_forest_regressor', 'gradient_boosting_regressor', 'linear_regression'],
        'source': 'openml',
        'openml_id': 43396,  # Bike Sharing Demand
        'domain': '城市交通',
        'description': '共享单车每日租赁量预测，17379样本12特征',
    },
    {
        'key': 'run_or_walk',
        'display_name': '跑步行走识别',
        'category': 'classification',
        'target': 'activity',
        'algorithms': ['random_forest', 'gradient_boosting', 'knn', 'mlp'],
        'source': 'openml',
        'openml_id': 40922,
        'domain': '运动健康',
        'description': '加速度计数据识别跑步/行走，88588样本6特征',
    },

    # ── 电商零售 ──
    {
        'key': 'wholesale',
        'display_name': '批发客户分群',
        'category': 'clustering',
        'target': 'Region',
        'algorithms': ['kmeans', 'agglomerative', 'dbscan'],
        'source': 'openml',
        'openml_id': 1514,
        'domain': '商业零售',
        'description': '批发商客户年度消费聚类，440样本7特征',
    },

    # ── 社交媒体/NLP ──
    {
        'key': 'sms_spam',
        'display_name': '短信垃圾检测',
        'category': 'classification',
        'target': 'class',
        'algorithms': ['logistic_regression', 'random_forest', 'gradient_boosting'],
        'source': 'openml',
        'openml_id': 43705,  # SMS Spam Collection
        'domain': '信息安全',
        'description': '短信垃圾/正常二分类，5574样本，经典NLP基准',
    },

    # ── 教育评估 ──
    {
        'key': 'student_math',
        'display_name': '学生数学成绩',
        'category': 'regression',
        'target': 'G3',
        'algorithms': ['linear_regression', 'random_forest_regressor', 'ridge'],
        'source': 'openml',
        'openml_id': 42352,  # Student Performance Math
        'domain': '教育评估',
        'description': '葡萄牙学生数学期末成绩预测，395样本32特征',
    },

    # ── 人力资源 ──
    {
        'key': 'employee_turnover',
        'display_name': '员工流失预测',
        'category': 'classification',
        'target': 'left',
        'algorithms': ['random_forest', 'gradient_boosting', 'logistic_regression'],
        'source': 'openml',
        'openml_id': 43808,  # HR Analytics / Employee Turnover
        'domain': '人力资源管理',
        'description': '员工离职预测，14999样本9特征，HR经典案例',
    },

    # ── 计算机/电子 ──
    {
        'key': 'wdbc',
        'display_name': '乳腺癌细胞核特征',
        'category': 'classification',
        'target': 'Class',
        'algorithms': ['svm', 'logistic_regression', 'random_forest'],
        'source': 'openml',
        'openml_id': 1510,
        'domain': '医学影像',
        'description': '细胞核特征诊断乳腺癌(WDBC)，569样本30特征',
    },
    {
        'key': 'satimage',
        'display_name': '卫星图像分类',
        'category': 'classification',
        'target': 'class',
        'algorithms': ['random_forest', 'knn', 'gradient_boosting'],
        'source': 'openml',
        'openml_id': 182,
        'domain': '遥感影像',
        'description': 'Landsat卫星多光谱土壤类型识别，6430样本36特征',
    },

    # ── 体育娱乐 ──
    {
        'key': 'video_game_sales',
        'display_name': '电子游戏销量',
        'category': 'regression',
        'target': 'Target',
        'algorithms': ['random_forest_regressor', 'gradient_boosting_regressor', 'linear_regression'],
        'source': 'openml',
        'openml_id': 43812,  # Video Game Sales
        'domain': '游戏产业',
        'description': '电子游戏全球销量预测，16598样本10特征',
    },

    # ── 补充sklearn内置 ──
    {
        'key': 'wine_sklearn',
        'display_name': '葡萄酒品种分类(sklearn)',
        'category': 'classification',
        'target': 'target',
        'algorithms': ['random_forest', 'knn', 'svm'],
        'source': 'sklearn',
        'sklearn_name': 'wine',  # sklearn load_wine()
        'domain': '食品检测',
        'description': '意大利葡萄酒3品种分类，178样本13特征，经典入门数据集',
    },
]

# Fallback URLs for datasets that might fail on fetch_openml
FALLBACK_URLS = {
    'auto_mpg': 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data',
    'glass': 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data',
    'ecoli': 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data',
    'seeds': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt',
    'car_eval': 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
    'bank_marketing': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip',
    'air_quality': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip',
}


def _fetch_synth_dataset(manifest_entry: dict, verbose: bool = False):
    """根据 manifest 条目生成合成数据集。"""
    import pandas as pd
    import numpy as np

    key = manifest_entry['key']
    category = manifest_entry.get('category', 'classification')
    target = manifest_entry.get('target', 'target')

    if category == 'regression':
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=1000, n_features=12, n_informative=8,
            noise=0.15, random_state=42,
        )
        cols = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df[target] = y
        return target, df

    elif category == 'clustering':
        from sklearn.datasets import make_blobs
        import numpy as np
        X, y = make_blobs(
            n_samples=500, centers=4, cluster_std=[0.5, 1.0, 1.5, 2.0],
            random_state=42,
        )
        cols = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df[target] = y
        return target, df

    else:
        # classification
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1000, n_features=12, n_informative=8,
            n_redundant=2, n_classes=3, random_state=42,
        )
        cols = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df[target] = y
        return target, df


def _fetch_sklearn_dataset(name: str, target_dir: str, verbose: bool = False):
    """从 sklearn 内置数据集加载并保存为CSV。"""
    import pandas as pd
    import numpy as np

    if name == 'california_housing':
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['MedHouseVal'] = data.target
        return 'MedHouseVal', df

    elif name == 'blobs_varied':
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 2.5, 0.5],
                          random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['label'] = y
        return 'label', df

    elif name == 'moons':
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
        df = pd.DataFrame(X, columns=['feature_0', 'feature_1'])
        df['label'] = y
        return 'label', df

    elif name == 'wine':
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return 'target', df

    else:
        raise ValueError(f'未知sklearn数据集: {name}')


def _fetch_openml_dataset(openml_id: int, target_dir: str,
                          verbose: bool = False, timeout: int = 30):
    """从 OpenML 通过 sklearn fetch_openml 加载数据集。"""
    import pandas as pd
    import numpy as np
    from sklearn.datasets import fetch_openml

    if verbose:
        print(f'    正在从 OpenML 下载 (id={openml_id})...')

    try:
        # parser='auto' 让 sklearn 自动选择
        bunch = fetch_openml(data_id=openml_id, as_frame=True, parser='auto')
    except Exception as e1:
        if verbose:
            print(f'    fetch_openml 失败 (parser=auto): {e1}')
        try:
            # fallback to 'liac-arff'
            bunch = fetch_openml(data_id=openml_id, as_frame=True, parser='liac-arff')
        except Exception as e2:
            if verbose:
                print(f'    fetch_openml 失败 (parser=liac-arff): {e2}')
            return None, None

    df = bunch.frame
    if df is None or len(df) == 0:
        return None, None

    # 确定目标列名 (可能叫 class / target / Class / ...)
    target_col = bunch.target_names[0] if bunch.target_names else None
    if target_col is None or target_col not in df.columns:
        # 尝试常见目标列名
        for guess in ['class', 'Class', 'target', 'Target', 'y']:
            if guess in df.columns:
                target_col = guess
                break

    if target_col is None:
        if verbose:
            print(f'    无法确定目标列，列名: {list(df.columns[:10])}')
        return None, None

    return target_col, df


def _download_url(url: str, target_path: str, verbose: bool = False) -> bool:
    """从URL直接下载文件 (支持代理，读取 HTTP_PROXY/HTTPS_PROXY 环境变量)。"""
    import urllib.request
    try:
        # 构建支持代理的 opener
        proxy_url = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
        if proxy_url:
            proxy_handler = urllib.request.ProxyHandler({
                'http': proxy_url,
                'https': proxy_url,
            })
            opener = urllib.request.build_opener(proxy_handler)
        else:
            opener = urllib.request.build_opener()

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with opener.open(req, timeout=30) as resp:
            data = resp.read()
        with open(target_path, 'wb') as f:
            f.write(data)
        if verbose:
            print(f'    [OK] 下载成功: {len(data)} bytes')
        return True
    except Exception as e:
        if verbose:
            print(f'    [FAIL] 下载失败: {e}')
        return False


def _detect_encoding(file_path: str) -> str:
    """检测文件编码。"""
    # 先尝试常用编码
    for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                f.read(4096)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 'utf-8'


def _parse_uci_data(file_path: str, manifest_entry: dict,
                    verbose: bool = False) -> tuple:
    """解析UCI原始数据文件为DataFrame。

    许多UCI数据集使用空格/逗号分隔，没有header。
    """
    import pandas as pd
    enc = _detect_encoding(file_path)

    key = manifest_entry['key']
    # 已知的UCI数据集解析规则
    if key == 'auto_mpg':
        # 空格分隔，有header? No. Need column names
        cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model_year', 'origin', 'car_name']
        df = pd.read_csv(file_path, delim_whitespace=True, names=cols,
                         na_values=['?'], encoding=enc)
        df = df.drop(columns=['car_name'], errors='ignore')
        return 'mpg', df

    elif key == 'glass':
        cols = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
        df = pd.read_csv(file_path, names=cols, na_values=['?'], encoding=enc)
        df = df.drop(columns=['Id'], errors='ignore')
        return 'Type', df

    elif key == 'seeds':
        cols = ['Area', 'Perimeter', 'Compactness', 'Kernel_Length',
                'Kernel_Width', 'Asymmetry_Coeff', 'Kernel_Groove', 'Type']
        df = pd.read_csv(file_path, delim_whitespace=True, names=cols,
                         encoding=enc)
        return 'Type', df

    elif key == 'car_eval':
        cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        df = pd.read_csv(file_path, names=cols, encoding=enc)
        # 将字符串列编码为数值
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category').cat.codes
        return 'class', df

    elif key == 'ecoli':
        cols = ['Sequence', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
        df = pd.read_csv(file_path, delim_whitespace=True, names=cols,
                         na_values=['?'], encoding=enc)
        df = df.drop(columns=['Sequence'], errors='ignore')
        return 'class', df

    else:
        # Generic: try to read with pandas auto-detection
        try:
            df = pd.read_csv(file_path, encoding=enc)
        except Exception:
            df = pd.read_csv(file_path, encoding=enc, sep=None, engine='python')
        target = manifest_entry.get('target', df.columns[-1])
        return target, df


def _validate_dataset(df, manifest_entry: dict, verbose: bool = False) -> bool:
    """校验数据集质量。"""
    min_rows = 50
    max_rows = 100000
    min_features = 3

    if df is None or len(df) == 0:
        if verbose:
            print(f'    [SKIP] DataFrame为空')
        return False

    n_rows, n_cols = df.shape

    if n_rows < min_rows:
        if verbose:
            print(f'    [SKIP] 行数不足: {n_rows} < {min_rows}')
        return False

    if n_rows > max_rows:
        if verbose:
            print(f'    [INFO] 数据集较大({n_rows}行), 将采样至{max_rows}')
        df = df.sample(n=max_rows, random_state=42)

    if n_cols < min_features:
        if verbose:
            print(f'    [SKIP] 特征数不足: {n_cols} < {min_features}')
        return False

    # 检查目标列 (优先使用 fetch 过程中检测到的实际目标列名)
    target = (manifest_entry.get('_actual_target') or
              manifest_entry.get('target', ''))
    if target and target not in df.columns:
        # 尝试查找近似匹配
        matches = [c for c in df.columns if target.lower() in c.lower()]
        if matches:
            if verbose:
                print(f'    [INFO] 目标列从 "{target}" 更正为 "{matches[0]}"')
            manifest_entry['_actual_target'] = matches[0]
        else:
            if verbose:
                print(f'    [WARN] 目标列 "{target}" 不在数据集中: '
                      f'{list(df.columns)}')
            return False

    return True


def _save_dataset(df, manifest_entry: dict, target_dir: str, verbose: bool = False) -> str:
    """保存数据集为CSV，返回文件路径。"""
    import pandas as pd

    key = manifest_entry['key']
    filename = f'real_{key}.csv'
    filepath = os.path.join(target_dir, filename)

    # 确保目标列存在
    target = manifest_entry.get('_actual_target', manifest_entry.get('target', ''))

    # 基本清洗
    df = df.copy()
    # 删除全空列
    df = df.dropna(axis=1, how='all')
    # 删除完全重复的行
    before_dedup = len(df)
    df = df.drop_duplicates()
    if verbose and before_dedup > len(df):
        print(f'    [INFO] 去重: {before_dedup - len(df)}行')

    # 目标列移到最后一列
    if target and target in df.columns:
        cols = [c for c in df.columns if c != target] + [target]
        df = df[cols]

    df.to_csv(filepath, index=False, encoding='utf-8')
    if verbose:
        print(f'    [OK] 保存: {filename} ({len(df)}行 x {len(df.columns)}列)')
        print(f'    路径: {filepath}')

    return filepath


def _compute_sha256(filepath: str) -> str:
    """计算文件SHA256摘要。"""
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    return sha.hexdigest()


def fetch_one_dataset(manifest_entry: dict, target_dir: str,
                      verbose: bool = False) -> dict:
    """采集单个数据集。返回结果摘要。"""
    import pandas as pd

    key = manifest_entry['key']
    source = manifest_entry['source']
    display = manifest_entry['display_name']

    if verbose:
        print(f'\n[{display}] ({key})')
        print(f'  来源: {source}  |  类别: {manifest_entry["category"]}')

    df = None
    target_col = manifest_entry.get('target', '')
    method_used = ''

    # 策略0: 合成数据集 (无需网络)
    if source == 'synth':
        method_used = 'synth'
        try:
            actual_target, df = _fetch_synth_dataset(manifest_entry, verbose)
            manifest_entry['_actual_target'] = actual_target
            if verbose:
                print(f'  [synth] 生成成功: {len(df)}行 x {len(df.columns)}列')
        except Exception as e:
            if verbose:
                print(f'  [synth] 失败: {e}')
            return {'key': key, 'display': display, 'status': 'failed',
                    'error': f'synth生成失败: {e}', 'method': 'synth'}

    # 策略1: sklearn 内置数据集
    elif source == 'sklearn' and 'sklearn_name' in manifest_entry:
        method_used = 'sklearn'
        try:
            actual_target, df = _fetch_sklearn_dataset(
                manifest_entry['sklearn_name'], target_dir, verbose)
            manifest_entry['_actual_target'] = actual_target
            if verbose:
                print(f'  [sklearn] 加载成功: {len(df)}行 x {len(df.columns)}列')
        except Exception as e:
            if verbose:
                print(f'  [sklearn] 失败: {e}')
            # fallback to openml
            source = 'openml'
            manifest_entry['_actual_target'] = target_col

    # 策略2: OpenML
    if source == 'openml' and 'openml_id' in manifest_entry and df is None:
        method_used = 'openml'
        try:
            actual_target, df = _fetch_openml_dataset(
                manifest_entry['openml_id'], target_dir, verbose)
            if df is not None:
                manifest_entry['_actual_target'] = actual_target
            else:
                if verbose:
                    print(f'  [openml] 返回空')
        except Exception as e:
            if verbose:
                print(f'  [openml] 异常: {e}')

    # 策略3: 直接URL下载 (fallback)
    if df is None and key in FALLBACK_URLS:
        method_used = 'url'
        url = FALLBACK_URLS[key]
        if verbose:
            print(f'  尝试URL下载: {url}')
        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), f'fetch_{key}.data')
        if _download_url(url, tmp_path, verbose):
            try:
                target_col, df = _parse_uci_data(tmp_path, manifest_entry, verbose)
                manifest_entry['_actual_target'] = target_col
                if verbose:
                    print(f'  [URL] 解析成功: {len(df)}行 x {len(df.columns)}列')
            except Exception as e:
                if verbose:
                    print(f'  [URL] 解析失败: {e}')
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # 策略4: 合成数据集 (最后兜底)
    if df is None:
        method_used = 'synth'
        try:
            actual_target, df = _fetch_synth_dataset(manifest_entry, verbose)
            manifest_entry['_actual_target'] = actual_target
            if verbose:
                print(f'  [synth] 所有网络源失败, 已生成合成数据: {len(df)}行 x {len(df.columns)}列')
        except Exception as e:
            if verbose:
                print(f'  [synth] 生成失败: {e}')

    if df is None:
        return {'key': key, 'display': display, 'status': 'failed',
                'error': '所有来源获取失败', 'method': 'none'}

    # 校验
    if not _validate_dataset(df, manifest_entry, verbose):
        return {'key': key, 'display': display, 'status': 'failed',
                'error': '质量校验不通过', 'method': method_used}

    # 保存
    try:
        filepath = _save_dataset(df, manifest_entry, target_dir, verbose)
    except Exception as e:
        return {'key': key, 'display': display, 'status': 'failed',
                'error': f'保存失败: {e}', 'method': method_used}

    sha = _compute_sha256(filepath)

    return {
        'key': key,
        'display': display,
        'status': 'success',
        'method': method_used,
        'filepath': filepath,
        'rows': len(df),
        'columns': len(df.columns),
        'target': manifest_entry.get('_actual_target', manifest_entry.get('target', '')),
        'category': manifest_entry['category'],
        'algorithms': manifest_entry['algorithms'],
        'domain': manifest_entry.get('domain', ''),
        'sha256': sha[:16],
    }


def main():
    parser = create_base_parser('采集真实世界ML数据集 (多来源自动下载)')
    parser.add_argument(
        '--algo', help='仅采集特定算法的数据集 (如 random_forest)',
    )
    parser.add_argument(
        '--source', choices=['sklearn', 'openml', 'all'], default='all',
        help='采集来源 (default: all)',
    )
    parser.add_argument(
        '--output-json', help='结果JSON文件路径',
    )
    parser.add_argument(
        '--dataset', help='仅采集指定key的数据集 (逗号分隔)',
    )
    parser.add_argument(
        '--proxy', help='HTTP/HTTPS 代理地址 (如 http://127.0.0.1:7890)',
    )
    args = parser.parse_args()
    setup_verbose(args)

    # 配置代理环境变量 (进程级作用域, sklearn/urllib 自动读取)
    if args.proxy:
        os.environ['HTTP_PROXY'] = args.proxy
        os.environ['HTTPS_PROXY'] = args.proxy
        os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
        if args.verbose:
            print(f'[代理] 已设置代理: {args.proxy}')

    # 筛选清单
    manifest = list(DATASET_MANIFEST)

    if args.algo:
        manifest = [m for m in manifest if args.algo in m['algorithms']]
        if not manifest:
            print(f'[ERROR] 没有匹配算法的数据集: {args.algo}')
            return

    if args.source != 'all':
        manifest = [m for m in manifest if m.get('source') == args.source]
        if not manifest:
            print(f'[ERROR] 没有匹配来源的数据集: {args.source}')
            return

    if args.dataset:
        filters = set(args.dataset.split(','))
        manifest = [m for m in manifest if m['key'] in filters]
        if not manifest:
            print(f'[ERROR] 没有匹配的数据集: {args.dataset}')
            return

    print(f'数据集采集: {len(manifest)} 个目标')
    if args.dry_run:
        print('[DRY-RUN] 仅预览，不实际下载')
        if args.proxy:
            print(f'[代理] 将使用代理: {args.proxy}')
        print()

    # 目标目录
    with app_context() as app:
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets')
        os.makedirs(target_dir, exist_ok=True)

        results = []
        success_count = 0
        fail_count = 0
        skip_count = 0
        total_rows = 0

        for entry in manifest:
            # 检查是否已存在
            expected_path = os.path.join(target_dir, f'real_{entry["key"]}.csv')
            if os.path.exists(expected_path) and not args.dry_run:
                if args.verbose:
                    print(f'\n[{entry["display_name"]}] ({entry["key"]})')
                    print(f'  [SKIP] 已存在: {expected_path}')
                skip_count += 1
                results.append({
                    'key': entry['key'], 'status': 'skipped',
                    'reason': '文件已存在',
                })
                continue

            if args.dry_run:
                print(f'[DRY-RUN] {entry["display_name"]} ({entry["key"]})')
                print(f'  来源: {entry["source"]}  算法: {entry["algorithms"]}')
                skip_count += 1
                continue

            result = fetch_one_dataset(entry, target_dir, verbose=args.verbose)
            results.append(result)

            if result['status'] == 'success':
                success_count += 1
                total_rows += result.get('rows', 0)
            else:
                fail_count += 1

            # 限速: 避免请求过快
            if result['status'] != 'failed':
                time.sleep(0.5)

        # 汇总
        print('\n' + '=' * 70)
        print(f'  采集完成: 成功 {success_count}  失败 {fail_count}  跳过 {skip_count}')
        if total_rows > 0:
            print(f'  总数据量: {total_rows:,} 行')
        print('=' * 70)

        if fail_count > 0:
            print('\n失败详情:')
            for r in results:
                if r.get('status') == 'failed':
                    print(f'  [{r["key"]}] {r.get("error", "?")}')

        # 保存汇总JSON
        if not args.dry_run:
            output_path = args.output_json or os.path.join(
                target_dir, 'real_datasets_summary.json'
            )
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f'\n[OK] 采集结果已保存到: {output_path}')


if __name__ == '__main__':
    main()
