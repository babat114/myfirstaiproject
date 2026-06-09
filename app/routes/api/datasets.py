"""
============================================
数据集 API
RESTful JSON 接口
============================================
"""
import os
import json
from flask import Blueprint, request, jsonify, current_app
from app.services.dataset_service import DatasetService
from app.services.auth_service import AuthService
from app.utils.decorators import api_login_required, api_admin_required
from app.utils.auth_helpers import get_current_user

datasets_api_bp = Blueprint('datasets_api', __name__)


@datasets_api_bp.route('/', methods=['GET'])
@api_login_required
def list_datasets():
    """GET /api/datasets - 获取数据集列表"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 15, type=int)
    category = request.args.get('category')
    search = request.args.get('search')

    user = get_current_user()
    result = DatasetService.list_datasets(
        page=page, per_page=per_page,
        category=category, search=search,
        owner_id=user.id,
    )

    return jsonify({
        'success': True,
        'data': result,
    })


@datasets_api_bp.route('/<string:dataset_uuid>', methods=['GET'])
@api_login_required
def get_dataset(dataset_uuid):
    """GET /api/datasets/<uuid> - 获取数据集详情"""
    dataset = DatasetService.get_dataset_by_uuid(dataset_uuid)
    if not dataset:
        return jsonify({'success': False, 'message': '数据集不存在。'}), 404

    return jsonify({
        'success': True,
        'data': dataset.to_dict(include_file_path=True),
    })


@datasets_api_bp.route('/', methods=['POST'])
@api_login_required
def create_dataset():
    """POST /api/datasets - 创建新数据集"""
    user = get_current_user()
    if not user.can_upload:
        return jsonify({'success': False, 'message': '没有上传权限。'}), 403

    name = request.form.get('name')
    file = request.files.get('file')

    if not name or not file:
        return jsonify({'success': False, 'message': '缺少必要字段。'}), 400

    dataset, error = DatasetService.create_dataset(
        user=user,
        name=name,
        file=file,
        description=request.form.get('description'),
        category=request.form.get('category', 'other'),
        is_public=request.form.get('is_public', 'false').lower() == 'true',
        upload_folder=current_app.config['UPLOAD_FOLDER'],
    )

    if error:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({
        'success': True,
        'message': '数据集创建成功。',
        'data': dataset.to_dict(),
    }), 201


@datasets_api_bp.route('/<string:dataset_uuid>', methods=['PUT'])
@api_login_required
def update_dataset(dataset_uuid):
    """PUT /api/datasets/<uuid> - 更新数据集"""
    dataset = DatasetService.get_dataset_by_uuid(dataset_uuid)
    if not dataset:
        return jsonify({'success': False, 'message': '数据集不存在。'}), 404

    user = get_current_user()
    if dataset.owner_id != user.id and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    data = request.get_json(silent=True) or {}
    success, error = DatasetService.update_dataset(dataset, data)

    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({
        'success': True,
        'message': '数据集已更新。',
        'data': dataset.to_dict(),
    })


@datasets_api_bp.route('/<string:dataset_uuid>', methods=['DELETE'])
@api_login_required
def delete_dataset(dataset_uuid):
    """DELETE /api/datasets/<uuid> - 删除数据集"""
    dataset = DatasetService.get_dataset_by_uuid(dataset_uuid)
    if not dataset:
        return jsonify({'success': False, 'message': '数据集不存在。'}), 404

    user = get_current_user()
    if dataset.owner_id != user.id and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    success, error = DatasetService.delete_dataset(dataset)
    if not success:
        return jsonify({'success': False, 'message': error}), 500

    return jsonify({'success': True, 'message': '数据集已删除。'})


# ============ 数据集推荐 API ============

@datasets_api_bp.route('/<string:dataset_uuid>/analyze', methods=['GET'])
@api_login_required
def analyze_dataset(dataset_uuid):
    """GET /api/datasets/<uuid>/analyze - 分析数据集并返回推荐"""
    dataset = DatasetService.get_dataset_by_uuid(dataset_uuid)
    if not dataset:
        return jsonify({'success': False, 'message': '数据集不存在。'}), 404

    file_path = dataset.file_path
    if not file_path or not os.path.exists(file_path):
        return jsonify({'success': False, 'message': '数据集文件不存在。'}), 404

    from app.services.dataset_recommendation_service import DatasetRecommendationService

    target_col = None
    if dataset.summary_json:
        try:
            summary = json.loads(dataset.summary_json)
            target_col = summary.get('target_column')
        except Exception:
            pass

    result = DatasetRecommendationService.recommend(
        file_path, target_col, dataset.file_format
    )

    if 'error' in result:
        return jsonify({'success': False, 'message': result['error']}), 400

    return jsonify({'success': True, 'data': result})


@datasets_api_bp.route('/<int:dataset_id>/analyze', methods=['GET'])
@api_login_required
def analyze_dataset_by_id(dataset_id):
    """GET /api/datasets/<id>/analyze - 按ID分析数据集"""
    dataset = DatasetService.get_dataset_by_id(dataset_id)
    if not dataset:
        return jsonify({'success': False, 'message': '数据集不存在。'}), 404

    from app.services.dataset_recommendation_service import DatasetRecommendationService

    target_col = None
    if dataset.summary_json:
        try:
            summary = json.loads(dataset.summary_json)
            target_col = summary.get('target_column')
        except Exception:
            pass

    result = DatasetRecommendationService.recommend(
        dataset.file_path, target_col, dataset.file_format,
        known_n_samples=dataset.row_count or None
    )

    if 'error' in result:
        return jsonify({'success': False, 'message': result['error']}), 400

    return jsonify({'success': True, 'data': result})


# 训练表单可用的算法白名单 (与 create.html 中的 ALGORITHMS 保持一致)
_FORM_CLASSIFICATION_ALGOS = {'random_forest', 'logistic_regression', 'svm', 'knn', 'gradient_boosting'}
_FORM_REGRESSION_ALGOS = {'random_forest_regressor', 'linear_regression', 'ridge', 'svr', 'gradient_boosting_regressor'}
_FORM_CLUSTERING_ALGOS = {'kmeans', 'dbscan', 'agglomerative', 'minibatch_kmeans'}

# 推荐算法 → 表单算法映射 (推荐引擎可能返回表单不支持的算法)
_ALGO_MAP = {
    'ridge': 'ridge',                             # Ridge → 岭回归 (L2正则, 保留)
    'knn_regressor': 'random_forest_regressor',   # KNN回归 → 随机森林回归
    'tfidf_logistic': 'logistic_regression',      # TF-IDF逻辑回归 → 逻辑回归 (SklearnTrainer已加TF-IDF管道)
    'tfidf_svm': 'svm',                           # TF-IDF SVM → SVM
    'transformer_bert': 'transformer_bert',       # BERT → TransformersNLPTrainer
    'mlp': 'mlp',                                 # MLP → PyTorch MLP (宽网络)
    'decision_tree': 'random_forest',
    'decision_tree_regressor': 'random_forest_regressor',
}


@datasets_api_bp.route('/<int:dataset_id>/auto-config', methods=['GET'])
@api_login_required
def auto_config(dataset_id):
    """GET /api/datasets/<id>/auto-config — 分析数据集并返回表单就绪的训练配置

    返回字段:
        ml_task_type:  推荐的任务类型 (classification/regression)
        algorithm:     推荐的算法 (表单可用值)
        target_column: 自动检测的目标列名
        framework:     推荐的框架 (sklearn/pytorch)
        test_size:     推荐的测试集比例
        total_epochs:  推荐的训练轮数
        reason:        推荐理由 (简短说明)
        alternative_algorithms: 备选算法列表
    """
    dataset = DatasetService.get_dataset_by_id(dataset_id)
    if not dataset:
        return jsonify({'success': False, 'message': '数据集不存在。'}), 404

    if not dataset.file_path or not os.path.exists(dataset.file_path):
        return jsonify({'success': False, 'message': '数据集文件不存在。'}), 404

    from app.services.dataset_recommendation_service import DatasetRecommendationService

    # 尝试从 summary_json 获取已保存的目标列
    target_col = None
    if dataset.summary_json:
        try:
            summary = json.loads(dataset.summary_json)
            target_col = summary.get('target_column')
        except Exception:
            pass

    result = DatasetRecommendationService.recommend(
        dataset.file_path, target_col, dataset.file_format,
        known_n_samples=dataset.row_count or None
    )

    if 'error' in result:
        return jsonify({'success': False, 'message': result['error']}), 400

    analysis = result['analysis']
    target_type = analysis.get('target_type', 'categorical')

    # --- 数据集类别感知: 类别与任务类型映射 ---
    # 数据集声明类别 → 允许的任务类型 & 警告
    ds_category = getattr(dataset, 'category', 'other') or 'other'

    CATEGORY_TASK_MAP = {
        'classification': ('classification', None),     # 强制分类
        'regression':     ('regression', None),          # 强制回归
        'clustering':     ('clustering', None),
        'nlp':            ('classification', '⚠ 此数据集为NLP文本类型，标准分类/回归可能不适用。建议使用TF-IDF+分类器或深度学习。'),
        'vision':         ('classification', '⚠ 此数据集为视觉类型，表格型算法不适用于图像特征。建议使用PyTorch/TensorFlow CNN。'),
        'synthetic':      (None, '⚠ 此数据集为合成/生成式数据，target列可能为随机合成值而非真实标签，模型可能无法学到有效模式。'),
        'time_series':    (None, '⚠ 此数据集为时间序列类型，标准train_test_split会破坏时序依赖。建议使用时序交叉验证。'),
        'biology':        (None, None),   # 生物医学: 跟随分析器结果
        'finance':        (None, None),   # 金融: 跟随分析器结果
        'tabular':        (None, None),   # 通用表格: 跟随分析器结果
        'other':          (None, None),   # 其他: 跟随分析器结果
    }

    forced_task, category_warning = CATEGORY_TASK_MAP.get(
        ds_category, (None, None)
    )

    # 启发式检测
    column_names = analysis.get('column_names', [])
    latent_cols = [c for c in column_names if c.startswith('latent_')]
    is_latent_data = (len(latent_cols) > 0 and
                      len(latent_cols) >= len(column_names) * 0.7)
    if is_latent_data and not category_warning:
        category_warning = ('⚠ 检测到' + str(len(latent_cols)) + '/' + str(len(column_names)) +
                           '特征为latent_*隐空间维度，此数据可能来自生成式模型（GAN/VAE/AE）。'
                           '隐空间特征与target之间的相关关系可能是随机噪声，模型可能无法学到有效模式。')

    # NLP 专属: 强制 transformer 框架
    is_nlp = ds_category == 'nlp'
    is_vision = ds_category == 'vision'

    if forced_task:
        ml_task_type = forced_task
    elif target_type == 'continuous':
        ml_task_type = 'regression'
    else:
        ml_task_type = 'classification'

    if ml_task_type == 'regression':
        form_algos = _FORM_REGRESSION_ALGOS
    elif ml_task_type == 'clustering':
        form_algos = _FORM_CLUSTERING_ALGOS
    else:
        form_algos = _FORM_CLASSIFICATION_ALGOS

    # 警告信息拼入推荐理由
    reason_text = str(result.get('summary', ''))
    if category_warning:
        reason_text = category_warning + ' ' + reason_text

    # --- 算法: 从推荐列表中选择第一个表单支持的算法 ---
    algorithm = None
    alternative_algorithms = []
    for rec in result.get('recommended_algorithms', []):
        algo_name = rec['algorithm']
        # 映射到表单值
        mapped = _ALGO_MAP.get(algo_name, algo_name)
        if mapped and mapped in form_algos:
            if algorithm is None:
                algorithm = mapped
            else:
                alternative_algorithms.append({
                    'value': mapped,
                    'display': rec.get('display', mapped),
                    'confidence': rec['confidence'],
                })

    # 如果推荐算法全部不匹配，使用默认值
    if algorithm is None:
        if ml_task_type == 'clustering':
            algorithm = 'kmeans'
        elif ml_task_type == 'classification':
            algorithm = 'random_forest'
        else:
            algorithm = 'random_forest_regressor'

    if not alternative_algorithms:
        # 至少提供一个备选
        alt_set = form_algos - {algorithm}
        for a in list(alt_set)[:2]:
            alternative_algorithms.append({'value': a, 'display': a, 'confidence': 0.5})

    # --- 目标列 ---
    # 使用分析器自动检测的目标列 (最后一列)
    target_column = target_col
    if not target_column and analysis.get('column_names'):
        target_column = analysis['column_names'][-1]

    # --- NLP 专属: 强制 Transformer 迁移学习 ---
    if is_nlp:
        algorithm = 'transformer_bert'
        framework = 'transformers'
        params = {
            'epochs': 3,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'max_length': 256,
            'test_size': 0.2,
        }
        total_epochs = 3
        # 替换推荐理由
        reason_text = ('[NLP Transformer] 使用预训练BERT模型进行迁移学习微调。'
                       '自动检测文本列+语言，3轮训练即可达到高准确率。' + reason_text)
    # --- 视觉专属: 强制 PyTorch 深度学习 ---
    elif is_vision:
        framework = 'pytorch'
        algorithm = 'mlp'
        params = result.get('param_presets', {})
        # 视觉 embedding 通常高维，需要更宽的网络
        n_feat = analysis.get('n_features', 100)
        if n_feat > 500:
            params['hidden_layers'] = [1024, 512, 256, 128]
        elif n_feat > 200:
            params['hidden_layers'] = [512, 256, 128, 64]
        else:
            params['hidden_layers'] = [256, 128, 64, 32]
        params['epochs'] = 20
        params['dropout'] = 0.4
        params['batch_size'] = 32
        total_epochs = 20
        reason_text = ('[Vision DL] 视觉特征使用PyTorch MLP深度网络训练。'
                       '自动加宽隐藏层+增加dropout防过拟合。' + reason_text)
    else:
        # --- 框架 ---
        frameworks = result.get('recommended_frameworks', [])
        framework = frameworks[0]['framework'] if frameworks else 'sklearn'

        # --- 超参数预设 ---
        params = result.get('param_presets', {})

        # --- Epochs: sklearn 算法固定为1 ---
        if algorithm in ('random_forest', 'gradient_boosting', 'random_forest_regressor',
                         'gradient_boosting_regressor', 'svm', 'svr',
                         'logistic_regression', 'linear_regression', 'knn', 'ridge',
                         'kmeans', 'dbscan', 'agglomerative', 'minibatch_kmeans'):
            total_epochs = 1
        else:
            total_epochs = params.get('epochs', 10)

    # 转换 numpy 数值为 Python 原生类型 (避免 JSON 序列化失败)
    def _py(val):
        """将 numpy/pandas 类型转换为 Python 原生类型"""
        if val is None:
            return None
        try:
            if hasattr(val, 'item'):  # numpy scalar
                val = val.item()
        except Exception:
            pass
        if isinstance(val, bool):
            return bool(val)
        if isinstance(val, (int, float)):
            return val
        return val

    # --- 推荐任务名称和描述 ---
    # 算法显示名映射
    _ALGO_DISPLAY = {
        'random_forest': '随机森林', 'gradient_boosting': '梯度提升',
        'logistic_regression': '逻辑回归', 'svm': 'SVM', 'knn': 'KNN',
        'random_forest_regressor': '随机森林回归', 'linear_regression': '线性回归',
        'svr': 'SVR', 'gradient_boosting_regressor': '梯度提升回归',
        'kmeans': 'K-Means', 'dbscan': 'DBSCAN',
        'agglomerative': '层次聚类', 'minibatch_kmeans': 'MiniBatchKMeans',
    }
    algo_display = _ALGO_DISPLAY.get(algorithm, algorithm)
    if ml_task_type == 'clustering':
        task_cn = '聚类'
    elif ml_task_type == 'regression':
        task_cn = '回归'
    else:
        task_cn = '分类'
    # 避免名称重复: "线性回归" + "回归" → "线性回归回归" → 改为 "线性回归"
    if task_cn in algo_display:
        task_cn = ''  # 算法名已包含任务类型

    # 从数据集中提取简短的名称片段
    ds_name = dataset.name or 'dataset'
    import re
    # 去除数据集名中的尺寸/样本数标记: _20K, (30K samples), _15K 等
    ds_short = re.sub(r'\s*\(?\d{1,4}[kK]\w*\s*\w*\)?\s*', ' ', ds_name)
    ds_short = re.sub(r'[_-]\d{1,4}[kK]\b', '', ds_short)
    ds_short = re.sub(r'\s+', ' ', ds_short).strip('_- ()')

    suggested_name = f'{ds_short}-{algo_display}{task_cn}'
    # 限制长度
    if len(suggested_name) > 80:
        suggested_name = f'{ds_short[:40]}-{algo_display}{task_cn}'

    # 描述
    n_samples = _py(analysis.get('n_samples', 0))
    n_features = _py(analysis.get('n_features', 0))
    n_classes = _py(analysis.get('n_classes', 0))
    missing_rate = _py(analysis.get('missing_rate', 0))
    imbalanced = bool(analysis.get('imbalanced', False))

    desc_parts = [f'使用 {algo_display} 算法在 {ds_name} 数据集上执行{task_cn}任务']
    if n_classes > 0:
        desc_parts.append(f'{n_classes}分类' if n_classes > 2 else '二分类')
    desc_parts.append(f'{n_samples:,}样本, {n_features}特征')
    if imbalanced:
        desc_parts.append('类别不平衡')
    if missing_rate > 0.01:
        desc_parts.append(f'缺失率{missing_rate:.1%}')

    suggested_description = '。'.join(desc_parts) + '。'

    return jsonify({
        'success': True,
        'data': {
            'ml_task_type': ml_task_type,
            'algorithm': algorithm,
            'target_column': target_column or '',
            'framework': framework,
            'test_size': _py(params.get('test_size', 0.2)),
            'total_epochs': _py(total_epochs),
            'suggested_name': suggested_name,
            'suggested_description': suggested_description,
            'reason': reason_text,
            'category_warning': bool(category_warning),
            'alternative_algorithms': [
                {'value': a['value'], 'display': a.get('display', a['value']),
                 'confidence': _py(a.get('confidence', 0.5))}
                for a in alternative_algorithms[:3]
            ],
            # 额外分析信息供前端展示
            'dataset_info': {
                'n_samples': _py(analysis.get('n_samples', 0)),
                'n_features': _py(analysis.get('n_features', 0)),
                'n_classes': _py(analysis.get('n_classes', 0)),
                'missing_rate': _py(analysis.get('missing_rate', 0)),
                'imbalanced': bool(analysis.get('imbalanced', False)),
            },
        },
    })


@datasets_api_bp.route('/public', methods=['GET'])
@api_login_required
def list_public_datasets():
    """GET /api/datasets/public - 获取可导入的公开数据集列表"""
    from app.services.dataset_import_service import DatasetImportService
    category = request.args.get('category')
    datasets = DatasetImportService.get_available_datasets(category=category)
    categories = DatasetImportService.get_categories()
    return jsonify({
        'success': True,
        'data': datasets,
        'categories': categories,
    })


@datasets_api_bp.route('/import/<dataset_key>', methods=['POST'])
@api_login_required
def import_public_dataset(dataset_key):
    """POST /api/datasets/import/<key> - 导入公开数据集"""
    from app.services.dataset_import_service import DatasetImportService
    user = get_current_user()
    data = request.get_json(silent=True) or {}
    name = data.get('name')
    dataset, error = DatasetImportService.import_dataset(user, dataset_key, name=name)
    if error:
        return jsonify({'success': False, 'message': error}), 400
    return jsonify({
        'success': True,
        'message': f'数据集 "{dataset.name}" 导入成功',
        'data': dataset.to_dict(),
    }), 201


@datasets_api_bp.route('/import/url', methods=['POST'])
@api_login_required
def import_from_url():
    """POST /api/datasets/import/url - 从URL导入数据集"""
    from app.services.dataset_import_service import DatasetImportService
    user = get_current_user()
    data = request.get_json(silent=True) or {}
    url = data.get('url')
    name = data.get('name')
    if not url or not name:
        return jsonify({'success': False, 'message': '缺少url或name参数'}), 400
    dataset, error = DatasetImportService.import_from_url(
        user, url, name,
        target_column=data.get('target_column'),
        description=data.get('description'),
    )
    if error:
        return jsonify({'success': False, 'message': error}), 400
    return jsonify({
        'success': True,
        'message': f'数据集 "{dataset.name}" 导入成功',
        'data': dataset.to_dict(),
    }), 201


@datasets_api_bp.route('/<int:dataset_id>/smart-params', methods=['GET'])
@api_login_required
def smart_params(dataset_id):
    """GET /api/datasets/<id>/smart-params — AI智能推荐高级超参数

    Query params:
        algorithm:     算法名称 (必填)
        ml_task_type:  任务类型 (默认 classification)
        framework:     框架 (默认 sklearn)

    返回:
        params:     推荐参数键值对 (可直接填入表单)
        reason:     推荐理由
        confidence: 置信度
        tips:       使用提示
        gridsearch_suggestion: 是否建议使用GridSearchCV
        gridsearch_reason:     推荐GridSearchCV的理由
    """
    from app.services.dataset_recommendation_service import DatasetAnalyzer
    from app.services.parameter_guidance_service import ParameterGuidanceService

    dataset = DatasetService.get_dataset_by_id(dataset_id)
    if not dataset:
        return jsonify({'success': False, 'message': '数据集不存在。'}), 404

    if not dataset.file_path or not os.path.exists(dataset.file_path):
        return jsonify({'success': False, 'message': '数据集文件不存在。'}), 404

    algorithm = request.args.get('algorithm', 'random_forest')
    ml_task_type = request.args.get('ml_task_type', 'classification')
    framework = request.args.get('framework', 'sklearn')

    # 分析数据集
    target_col = None
    if dataset.summary_json:
        try:
            summary = json.loads(dataset.summary_json)
            target_col = summary.get('target_column')
        except Exception:
            pass

    analysis = DatasetAnalyzer.analyze(
        dataset.file_path, target_col, dataset.file_format,
    )

    if 'error' in analysis:
        return jsonify({'success': False, 'message': analysis['error']}), 400

    # 使用已知的实际样本数
    if dataset.row_count and dataset.row_count > 0:
        analysis['n_samples'] = dataset.row_count

    # 生成推荐
    result = ParameterGuidanceService.recommend_initial_params(
        analysis=analysis,
        algorithm=algorithm,
        ml_task_type=ml_task_type,
        framework=framework,
    )

    # 转换 numpy 类型
    def _py(val):
        if val is None:
            return None
        try:
            if hasattr(val, 'item'):
                return val.item()
        except Exception:
            pass
        if isinstance(val, bool):
            return bool(val)
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, list):
            return [_py(v) for v in val]
        if isinstance(val, dict):
            return {str(k): _py(v) for k, v in val.items()}
        return val

    return jsonify({
        'success': True,
        'data': {
            'params': _py(result['params']),
            'reason': result['reason'],
            'scale': result['scale'],
            'confidence': _py(result['confidence']),
            'tips': result['tips'],
            'gridsearch_suggestion': result['gridsearch_suggestion'],
            'gridsearch_reason': result['gridsearch_reason'],
        },
    })


@datasets_api_bp.route('/stats', methods=['GET'])
@api_login_required
def dataset_stats():
    """GET /api/datasets/stats - 获取数据集统计"""
    user = get_current_user()
    stats = DatasetService.get_dataset_statistics(user_id=user.id)
    return jsonify({'success': True, 'data': stats})
