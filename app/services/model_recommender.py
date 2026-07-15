"""
============================================
模型智能推荐服务
从模型文件中提取信息, 自动推荐名称/描述/版本
支持规则推荐 (离线) 和 LLM 增强推荐 (可选)
============================================
"""
import json
import os
import re

from app import logger


class ModelRecommender:
    """模型元数据智能推荐 — 规则为主, LLM 增强可选

    使用流程:
        1. 解析模型文件提取 info dict
        2. ModelRecommender.recommend(info) → {name, description, version}

    规则推荐覆盖场景:
        - 从特征名前缀推断领域 (tfidf_, petal_, age 等)
        - 从算法名生成可读缩写 (RandomForestClassifier → 随机森林)
        - 从任务类型 + 领域 + 算法拼接名称
    """

    # ── 领域检测关键词 (与 routes/models.py _build_model_hints 一致) ──
    DOMAIN_KEYWORDS = {
        'movie':       ['douban', 'movie', '电影', '影评', '豆瓣', 'film', 'cinema'],
        'shopping':    ['shopping', 'taobao', 'jd', '购物', '商品', '电商', '淘宝', '好评', '差评'],
        'restaurant':  ['restaurant', 'food', '餐厅', '美食', '外卖', '点评', '菜单'],
        'hotel':       ['hotel', '酒店', '住宿', '旅行', '宾馆'],
        'finance':     ['finance', 'stock', '金融', '股票', '财经', '基金', '贷款'],
        'social':      ['weibo', 'twitter', '社交', '舆情', '微博', '评论'],
        'news':        ['news', '新闻', '头条', '资讯', '报道'],
        'medical':     ['medical', 'health', '医疗', '健康', '诊断', '疾病', '病历', 'symptom'],
        'education':   ['education', '学校', '教育', '学生', '课程', '考试', '成绩'],
        'iris':        ['sepal', 'petal', 'iris'],
        'wine':        ['alcohol', 'malic', 'ash', 'proline', 'wine'],
        'breast_cancer': ['radius', 'texture', 'perimeter', 'concave'],
        'diabetes':    ['glucose', 'insulin', 'bmi', 'diabetes'],
        'housing':     ['medv', 'crim', 'tax', 'rm', 'lstat', 'housing', '房价'],
        'titanic':     ['pclass', 'sibsp', 'parch', 'fare', 'titanic', '幸存'],
        'sentiment':   ['sentiment', '情感', '正向', '负向', 'positive', 'negative'],
    }

    DOMAIN_LABELS = {
        'movie':        '影评',
        'shopping':     '电商评价',
        'restaurant':   '餐饮评价',
        'hotel':        '酒店评价',
        'finance':      '金融分析',
        'social':       '社交舆情',
        'news':         '新闻分类',
        'medical':      '医疗诊断',
        'education':    '教育分析',
        'iris':         '鸢尾花',
        'wine':         '红酒',
        'breast_cancer':'乳腺癌',
        'diabetes':     '糖尿病',
        'housing':      '房价',
        'titanic':      '泰坦尼克',
        'sentiment':    '情感',
    }

    # ── 算法名 → 可读缩写 ──
    ALGORITHM_SHORT = {
        # 分类
        'LogisticRegression': '逻辑回归',
        'RandomForestClassifier': '随机森林',
        'DecisionTreeClassifier': '决策树',
        'KNeighborsClassifier': 'K近邻',
        'SVC': '支持向量机',
        'GaussianNB': '朴素贝叶斯',
        'GradientBoostingClassifier': '梯度提升',
        'XGBClassifier': 'XGBoost',
        'LGBMClassifier': 'LightGBM',
        'CatBoostClassifier': 'CatBoost',
        'MLPClassifier': 'MLP神经网络',
        'AdaBoostClassifier': 'AdaBoost',
        'LinearSVC': '线性SVM',
        'RidgeClassifier': '岭回归分类',
        # 回归
        'LinearRegression': '线性回归',
        'Ridge': '岭回归',
        'Lasso': 'Lasso回归',
        'ElasticNet': '弹性网络',
        'RandomForestRegressor': '随机森林回归',
        'GradientBoostingRegressor': '梯度提升回归',
        'DecisionTreeRegressor': '决策树回归',
        'SVR': '支持向量回归',
        'XGBRegressor': 'XGBoost回归',
        'LGBMRegressor': 'LightGBM回归',
        'MLPRegressor': 'MLP回归',
        # 聚类
        'KMeans': 'KMeans',
        'DBSCAN': 'DBSCAN',
        'AgglomerativeClustering': '层次聚类',
        'GaussianMixture': '高斯混合',
        'Birch': 'BIRCH聚类',
        'MeanShift': '均值漂移',
        'SpectralClustering': '谱聚类',
        # NLP
        'transformers': 'Transformer',
        'BertForSequenceClassification': 'BERT',
        'XLNetForSequenceClassification': 'XLNet',
        'RobertaForSequenceClassification': 'RoBERTa',
        'TfidfVectorizer': 'TF-IDF',
    }

    TASK_LABELS = {
        'classification': '分类',
        'regression': '回归',
        'clustering': '聚类',
        'nlp': '文本分析',
        'computer_vision': '图像识别',
        'other': '通用',
    }

    @classmethod
    def recommend(cls, info: dict) -> dict:
        """主入口: 从模型信息生成推荐元数据

        Args:
            info: 从模型文件提取的信息字典, 包含:
                - algorithm: str — 算法类名
                - task_type: str — classification/regression/clustering/nlp
                - feature_names: list[str]
                - class_labels: list[str]
                - filename: str — 上传文件名
                - dataset_name: str (optional)
                - existing_metadata: dict (optional, 来自 metadata.json)

        Returns:
            {name: str, description: str, version: str}
        """
        name = cls._recommend_name(info)
        description = cls._recommend_description(info)
        version = cls._recommend_version(info)

        result = {
            'name': name,
            'description': description,
            'version': version,
        }
        logger.info(f"ModelRecommender 推荐: name={name}, version={version}")
        return result

    @classmethod
    def recommend_with_llm(cls, info: dict) -> dict | None:
        """LLM 增强推荐 — 适用于有 API Key 的场景

        生成更自然、更准确的中文名称和描述。
        调用失败时自动降级到规则推荐。
        """
        try:
            from app.services.llm_service import llm_complete
        except ImportError:
            logger.warning("llm_service 不可用, 降级到规则推荐")
            return cls.recommend(info)

        feature_sample = info.get('feature_names', [])[:10]
        class_sample = info.get('class_labels', [])[:6]

        prompt = f"""你是一个AI模型元数据专家。根据以下模型信息生成推荐的中文名称和描述。

模型信息:
- 算法: {info.get('algorithm', '未知')}
- 任务类型: {cls.TASK_LABELS.get(info.get('task_type', ''), info.get('task_type', '未知'))}
- 特征数量: {len(info.get('feature_names', []))}
- 特征示例: {feature_sample}
- 类别标签: {class_sample}
- 上传文件名: {info.get('filename', '未知')}

请严格按照JSON格式返回, 不要包含其他文字:
{{"name": "简短中文名(5-20字)", "description": "一句话描述模型用途(20-60字)", "version": "1.0.0"}}"""

        try:
            resp = llm_complete(prompt, max_tokens=200)
            result = json.loads(resp)
            # 校验必填字段
            if not result.get('name') or not result.get('description'):
                raise ValueError("LLM返回缺少必填字段")
            logger.info(f"ModelRecommender LLM推荐: {result}")
            return result
        except Exception as e:
            logger.warning(f"LLM 推荐失败 ({e}), 降级到规则推荐")
            return cls.recommend(info)

    # ══════════════════════════════════════════════════════
    # 规则推荐实现
    # ══════════════════════════════════════════════════════

    @classmethod
    def _recommend_name(cls, info: dict) -> str:
        """生成推荐名称: {领域}{任务类型}({算法缩写})"""
        # 优先从已有 metadata.json 取原名
        existing = info.get('existing_metadata', {})
        if existing.get('name'):
            return existing['name']

        # 拼接式生成 (优先于文件名, 因为领域+任务+算法更可读)
        # 先尝试检测领域来决定是否优先用复合名
        domain = cls._detect_domain(info)
        has_domain = domain is not None

        # 有领域时优先复合名; 无领域时用文件名为 fallback
        if not has_domain:
            filename = (info.get('filename') or '').strip()
            clean_name = cls._clean_filename(filename)
            if clean_name and len(clean_name) >= 3:
                return clean_name

        # 拼接式生成
        parts = []

        # 1) 领域 (已在上方检测, 复用)
        if domain:
            parts.append(cls.DOMAIN_LABELS.get(domain, domain))

        # 2) 任务类型
        task_type = info.get('task_type', '')
        task_label = cls.TASK_LABELS.get(task_type, '')
        if task_label:
            parts.append(task_label)

        # 3) 算法缩写
        algo = info.get('algorithm', '')
        if algo:
            short = cls.ALGORITHM_SHORT.get(algo)
            if not short:
                short = cls._shorten_generic(algo)
            if short:
                parts.append(f'({short})')

        if parts:
            return '_'.join(parts)
        return '导入模型'

    @classmethod
    def _recommend_description(cls, info: dict) -> str:
        """生成推荐描述 — 应用场景 + 使用方式 + 算法原理"""
        existing = info.get('existing_metadata', {})
        if existing.get('description'):
            return existing['description']

        # 从 info 提取算法 key (兼容 algorithm key 和 class name)
        algo = info.get('algorithm', '')
        algo_key = cls._algo_name_to_key(algo)

        return generate_enhanced_description(
            dataset_name=info.get('dataset_name', ''),
            task_type=info.get('task_type', ''),
            algorithm=algo_key,
            class_labels=info.get('class_labels', []),
            feature_names=info.get('feature_names', []),
            target_column='',
            model_name=info.get('filename', ''),
        )

    @classmethod
    def _algo_name_to_key(cls, name: str) -> str:
        """将算法类名/算法名 映射为 ALGORITHM_INFO 的 key"""
        return _normalize_algo_key(name)

    @classmethod
    def _recommend_version(cls, info: dict) -> str:
        """生成推荐版本号"""
        existing = info.get('existing_metadata', {})
        if existing.get('version'):
            return existing['version']
        return '1.0.0'

    # ══════════════════════════════════════════════════════
    # 工具方法
    # ══════════════════════════════════════════════════════

    @classmethod
    def _detect_domain(cls, info: dict) -> str | None:
        """从特征名、算法名、类别标签推断领域"""
        search_text = ''

        # 特征名
        feature_names = info.get('feature_names', [])
        search_text += ' '.join(str(f) for f in feature_names)

        # 算法名
        algo = info.get('algorithm', '')
        search_text += ' ' + algo

        # 类别标签
        class_labels = info.get('class_labels', [])
        search_text += ' ' + ' '.join(str(c) for c in class_labels)

        # 文件名
        search_text += ' ' + (info.get('filename') or '')

        search_text = search_text.lower()

        best_domain = None
        best_score = 0

        for domain, keywords in cls.DOMAIN_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw.lower() in search_text:
                    score += 1
            if score > best_score:
                best_score = score
                best_domain = domain

        if best_score >= 1:
            return best_domain
        return None

    @classmethod
    def _clean_filename(cls, filename: str) -> str | None:
        """从文件名提取可读名称: 去掉 uuid 前缀和扩展名"""
        if not filename:
            return None

        # 去掉扩展名
        name = re.sub(r'\.(pkl|pt|pth|h5|keras|onnx|joblib|zip)$', '', filename, flags=re.IGNORECASE)

        # 去掉 uuid 前缀 (32位hex + 下划线)
        name = re.sub(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_', '', name, flags=re.IGNORECASE)
        name = re.sub(r'^[0-9a-f]{32}_', '', name, flags=re.IGNORECASE)

        # 下划线 → 空格
        name = name.replace('_', ' ').replace('-', ' ').strip()

        # 如果太短 (只剩扩展名之类) 返回 None
        if len(name) < 2:
            return None

        return name

    @classmethod
    def _shorten_generic(cls, algo: str) -> str | None:
        """通用算法名缩写: CamelCase → 空格分隔"""
        if not algo:
            return None
        # CamelCase → 空格分隔
        s = re.sub(r'([a-z])([A-Z])', r'\1 \2', algo)
        s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
        # 取前2-3个单词
        words = s.split()
        if len(words) <= 3:
            return s
        return ' '.join(words[:3]) + '...'

    @classmethod
    def extract_from_model_file(cls, model, inference_metadata: dict = None) -> dict:
        """从 ModelRecord + InferenceService 元数据提取推荐所需的 info dict

        Args:
            model: ModelRecord 实例
            inference_metadata: InferenceService.load_model() 返回的 metadata

        Returns:
            可供 recommend() 使用的 info dict
        """
        metadata = inference_metadata or {}
        hyperparams = model.hyperparameters_dict

        info = {
            'algorithm': metadata.get('algorithm', hyperparams.get('algorithm', '')),
            'task_type': metadata.get('task_type', model.model_type),
            'feature_names': metadata.get('feature_names', []),
            'class_labels': metadata.get('class_labels', []),
            'dataset_name': model.training_dataset.name if model.training_dataset else None,
            'filename': os.path.basename(model.model_file_path) if model.model_file_path else None,
        }

        # 如果已有完整 DB 记录, 直接使用
        if model.name and model.name != '导入模型':
            info['existing_metadata'] = {
                'name': model.name,
                'description': model.description,
                'version': model.version,
            }

        return info


# 模块级便捷函数
def recommend_model_metadata(info: dict) -> dict:
    """便捷函数: 单次调用推荐"""
    return ModelRecommender.recommend(info)


# ══════════════════════════════════════════════════════
# 增强描述生成 — 应用场景 + 使用方式 + 算法原理
# ══════════════════════════════════════════════════════

TASK_LABELS_DESC = {
    'classification': '分类',
    'regression':      '回归',
    'clustering':      '聚类',
    'nlp':             '文本分析',
    'computer_vision': '图像识别',
    'other':           '预测',
}


def generate_enhanced_description(
    dataset_name: str = '',
    task_type: str = '',
    algorithm: str = '',
    class_labels: list = None,
    feature_names: list = None,
    target_column: str = '',
    model_name: str = '',
) -> str:
    """生成增强模型描述

    格式:
        应用场景：基于"{数据集}"的{任务}模型，{用途说明}。
        使用方式：输入{特征说明}，模型输出{输出说明}。

        {算法原理描述}

    Args:
        dataset_name: 数据集名称
        task_type: 任务类型 (classification/regression/clustering/nlp)
        algorithm: 算法 key或类名 (如 gradient_boosting / RandomForestClassifier)
        class_labels: 类别标签列表
        feature_names: 特征名列表
        target_column: 目标列名
        model_name: 模型名称 (备选语境来源)
    """
    from app.utils.algorithm_info import ALGORITHM_INFO

    class_labels = class_labels or []
    feature_names = feature_names or []
    ctx = dataset_name or model_name or ''
    has_tfidf = any(str(f).startswith('tfidf_') for f in feature_names)

    # ── 算法名归一化 (兼容 key 和 class name) ──
    algo_key = _normalize_algo_key(algorithm)
    algo_info = ALGORITHM_INFO.get(algo_key, {})
    algo_cn_name = algo_info.get('name', algorithm)
    algo_desc = algo_info.get('description', '')

    task_cn = TASK_LABELS_DESC.get(task_type, task_type)

    # ── 应用场景 ──
    scene_parts = [f'应用场景：基于"{ctx}"数据的{task_cn}模型']

    if task_type == 'classification' or task_type == 'nlp':
        if len(class_labels) == 2:
            a, b = class_labels[0], class_labels[1]
            scene_parts.append(f'可对样本进行二分类预测（{a}/{b}）')
        elif len(class_labels) > 2:
            labels_show = '/'.join(str(c) for c in class_labels[:5])
            if len(class_labels) > 5:
                labels_show += f'等{len(class_labels)}个类别'
            scene_parts.append(f'可识别{labels_show}共{len(class_labels)}个类别')
        elif target_column:
            scene_parts.append(f'根据特征预测目标列"{target_column}"的类别')
        else:
            scene_parts.append('根据输入特征预测所属类别')
    elif task_type == 'regression':
        if target_column:
            scene_parts.append(f'根据特征预测目标列"{target_column}"的连续数值')
        else:
            scene_parts.append('根据输入特征预测连续数值')
    elif task_type == 'clustering':
        scene_parts.append('无监督聚类，根据特征相似性自动将样本分组')

    if ctx:
        scene_parts[-1] = scene_parts[-1].rstrip('。')

    scene = '，'.join(scene_parts) + '。'

    # ── 使用方式 ──
    use_items = []

    if has_tfidf:
        use_items.append('输入原始文本，系统自动提取TF-IDF特征后进行预测')
    elif feature_names:
        n = len(feature_names)
        if n <= 10:
            fshow = '、'.join(str(f) for f in feature_names)
            use_items.append(f'输入{n}个特征（{fshow}）的数值')
        else:
            fshow = '、'.join(str(f) for f in feature_names[:5])
            use_items.append(f'输入{n}个特征（如{fshow}等）的数值')
    elif target_column:
        use_items.append('输入与训练时一致的特征数据')
    else:
        use_items.append('输入与训练数据格式一致的特征向量')

    if task_type == 'classification':
        if class_labels:
            use_items.append(f'模型输出所属类别（{"/".join(str(c) for c in class_labels[:5])}）及对应的置信度概率')
        else:
            use_items.append('模型输出预测类别及对应的置信度概率')
    elif task_type == 'nlp':
        use_items.append('模型输出文本分类结果及置信度概率')
    elif task_type == 'regression':
        use_items.append('模型输出预测数值')
    elif task_type == 'clustering':
        use_items.append('模型输出每个样本所属的簇编号')

    usage = '使用方式：' + '，'.join(use_items) + '。' if use_items else '使用方式：输入特征数据，获取预测结果。'

    # ── 算法原理 ──
    algo_section = f'使用{algo_cn_name}算法训练 - {algo_desc}' if algo_desc else f'使用{algo_cn_name}算法训练'

    return f'{scene}\n{usage}\n\n{algo_section}'


def _normalize_algo_key(algorithm: str) -> str:
    """将任意算法名/类名归一化为 ALGORITHM_INFO 的 key"""
    if not algorithm:
        return ''
    from app.utils.algorithm_info import ALGORITHM_INFO
    key = algorithm.lower().replace('-', '_')
    if key in ALGORITHM_INFO:
        return key
    mapping = {
        'logisticregression': 'logistic_regression',
        'randomforestclassifier': 'random_forest',
        'randomforestregressor': 'random_forest_regressor',
        'gradientboostingclassifier': 'gradient_boosting',
        'gradientboostingregressor': 'gradient_boosting_regressor',
        'decisiontreeclassifier': 'decision_tree',
        'decisiontreeregressor': 'decision_tree',
        'kneighborsclassifier': 'knn',
        'kneighborsregressor': 'knn_regressor',
        'svc': 'svm',
        'svr': 'svr',
        'linearsvc': 'svm',
        'linearregression': 'linear_regression',
        'ridge': 'ridge',
        'lasso': 'linear_regression',
        'elasticnet': 'linear_regression',
        'mlpclassifier': 'mlp',
        'mlpregressor': 'mlp',
        'mlp_classifier': 'mlp',
        'kmeans': 'kmeans',
        'dbscan': 'dbscan',
        'gaussiannb': 'logistic_regression',
        'adaboostclassifier': 'gradient_boosting',
        'xgbclassifier': 'gradient_boosting',
        'lgbmclassifier': 'gradient_boosting',
    }
    return mapping.get(key, key)

