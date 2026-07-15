"""
============================================
AI模型 Web 路由
模型注册表管理的页面路由
============================================
"""

import json

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from app.services.model_service import ModelService

models_bp = Blueprint('models', __name__)


@models_bp.route('/')
@login_required
def list_models():
    """模型列表页面 — 支持排序、筛选、公开/私有切换"""
    page = request.args.get('page', 1, type=int)
    model_type = request.args.get('model_type')
    framework = request.args.get('framework')
    search = request.args.get('search', '').strip() or None
    sort_by = request.args.get('sort_by', 'created_at')
    sort_order = request.args.get('sort_order', 'desc')
    visibility = request.args.get('visibility', 'owned')  # owned | public | all

    # 根据可见性构建查询参数
    if visibility == 'public':
        # 仅公开模型 (不限owner)
        result = ModelService.list_models(
            page=page,
            model_type=model_type,
            framework=framework,
            search=search,
            is_public=True,
            sort_by=sort_by,
            sort_order=sort_order,
        )
    elif visibility == 'all':
        # 我的 + 其他公开模型
        result = ModelService.list_models(
            page=page,
            model_type=model_type,
            framework=framework,
            search=search,
            owner_id=current_user.id,
            include_public=True,
            sort_by=sort_by,
            sort_order=sort_order,
        )
    else:
        # 仅我的模型
        result = ModelService.list_models(
            page=page,
            model_type=model_type,
            framework=framework,
            search=search,
            owner_id=current_user.id,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    return render_template(
        'models/list.html',
        models=result['items'],
        pagination=result,
        current_type=model_type,
        current_framework=framework,
        search_query=search,
        current_sort=sort_by,
        current_order=sort_order,
        current_visibility=visibility,
    )


@models_bp.route('/public')
@login_required
def public_models():
    """公开模型浏览 — 支持排序"""
    page = request.args.get('page', 1, type=int)
    model_type = request.args.get('model_type')
    framework = request.args.get('framework')
    search = request.args.get('search', '').strip() or None
    sort_by = request.args.get('sort_by', 'accuracy')
    sort_order = request.args.get('sort_order', 'desc')

    result = ModelService.list_models(
        page=page,
        model_type=model_type,
        framework=framework,
        search=search,
        is_public=True,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    return render_template(
        'models/public.html',
        models=result['items'],
        pagination=result,
        current_type=model_type,
        current_framework=framework,
        search_query=search,
        current_sort=sort_by,
        current_order=sort_order,
    )


@models_bp.route('/create', methods=['GET', 'POST'])
@login_required
def create_model():
    """注册新模型"""
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip() or None
        model_type = request.form.get('model_type', 'other')
        framework = request.form.get('framework', '').strip() or None
        version = request.form.get('version', '1.0.0').strip()
        is_public = request.form.get('is_public') == 'on'

        if not name:
            flash('请输入模型名称。', 'danger')
            return render_template('models/create.html')

        # 解析超参数
        hyperparams = {}
        hp_keys = request.form.getlist('hp_key[]')
        hp_values = request.form.getlist('hp_value[]')
        for k, v in zip(hp_keys, hp_values, strict=False):
            if k.strip() and v.strip():
                try:
                    hyperparams[k.strip()] = float(v.strip()) if '.' in v else int(v.strip())
                except ValueError:
                    hyperparams[k.strip()] = v.strip()

        model, error = ModelService.create_model(
            user=current_user,
            name=name,
            model_type=model_type,
            framework=framework,
            description=description,
            version=version,
            hyperparameters=hyperparams if hyperparams else None,
            is_public=is_public,
        )

        if error:
            flash(error, 'danger')
            return render_template('models/create.html')

        flash(f'模型 "{model.name}" 注册成功！', 'success')
        return redirect(url_for('models.model_detail', model_id=model.id))

    return render_template('models/create.html')


@models_bp.route('/import')
@login_required
def import_model():
    """导入模型页面 — 上传 .pkl / .zip 文件创建新模型记录"""
    return render_template('models/import.html')


@models_bp.route('/<int:model_id>')
@login_required
def model_detail(model_id):
    """模型详情页面"""
    model = ModelService.get_model_by_id(model_id)
    if not model:
        flash('模型不存在。', 'danger')
        return redirect(url_for('models.list_models'))

    if not model.is_viewable_by(current_user):
        flash('您没有权限查看此模型。', 'danger')
        return redirect(url_for('models.list_models'))

    # 加载评论区数据 (仅公开模型)
    comments_data = None
    if model.is_public:
        from app.services.comment_service import CommentService

        page = request.args.get('comment_page', 1, type=int)
        comments_data = CommentService.get_comments_for_model(
            model_id=model.id,
            user=current_user,
            page=page,
            per_page=10,
            include_hidden=current_user.is_admin,
        )

    return render_template('models/detail.html', model=model, comments_data=comments_data)


@models_bp.route('/<int:model_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_model(model_id):
    """编辑模型"""
    model = ModelService.get_model_by_id(model_id)
    if not model:
        flash('模型不存在。', 'danger')
        return redirect(url_for('models.list_models'))

    if not model.is_editable_by(current_user):
        flash('您没有权限编辑此模型。', 'danger')
        return redirect(url_for('models.list_models'))

    if request.method == 'POST':
        data = {
            'name': request.form.get('name', '').strip(),
            'description': request.form.get('description', '').strip() or None,
            'model_type': request.form.get('model_type'),
            'framework': request.form.get('framework', '').strip() or None,
            'version': request.form.get('version', '').strip(),
            'is_public': request.form.get('is_public') == 'on',
        }

        hp_keys = request.form.getlist('hp_key[]')
        hp_values = request.form.getlist('hp_value[]')
        hyperparams = {}
        for k, v in zip(hp_keys, hp_values, strict=False):
            if k.strip() and v.strip():
                try:
                    hyperparams[k.strip()] = float(v.strip()) if '.' in v else int(v.strip())
                except ValueError:
                    hyperparams[k.strip()] = v.strip()
        if hyperparams:
            data['hyperparameters'] = hyperparams

        success, error = ModelService.update_model(model, data)
        if success:
            flash('模型信息已更新。', 'success')
            return redirect(url_for('models.model_detail', model_id=model.id))
        else:
            flash(error, 'danger')

    return render_template('models/edit.html', model=model)


@models_bp.route('/<int:model_id>/upload', methods=['POST'])
@login_required
def upload_model_file(model_id):
    """上传模型文件"""
    model = ModelService.get_model_by_id(model_id)
    if not model or not model.is_editable_by(current_user):
        flash('模型不存在或权限不足。', 'danger')
        return redirect(url_for('models.list_models'))

    file = request.files.get('model_file')
    if not file:
        flash('请选择文件。', 'danger')
        return redirect(url_for('models.model_detail', model_id=model.id))

    success, error = ModelService.upload_model_file(model, file, upload_folder=current_app.config['UPLOAD_FOLDER'])
    if success:
        flash('模型文件上传成功！', 'success')
    else:
        flash(error, 'danger')

    return redirect(url_for('models.model_detail', model_id=model.id))


@models_bp.route('/<int:model_id>/delete', methods=['POST'])
@login_required
def delete_model(model_id):
    """删除模型"""
    model = ModelService.get_model_by_id(model_id)
    if not model:
        flash('模型不存在。', 'danger')
        return redirect(url_for('models.list_models'))

    if model.owner_id != current_user.id and not current_user.is_admin:
        flash('您没有权限删除此模型。', 'danger')
        return redirect(url_for('models.list_models'))

    success, error = ModelService.delete_model(model)
    if success:
        flash('模型已删除。', 'success')
    else:
        flash(error, 'danger')

    return redirect(url_for('models.list_models'))


@models_bp.route('/leaderboard')
@login_required
def leaderboard():
    """模型排行榜"""
    top_by_accuracy = ModelService.get_top_models(limit=10, metric='accuracy')
    top_by_f1 = ModelService.get_top_models(limit=10, metric='f1_score')
    return render_template(
        'models/leaderboard.html',
        top_accuracy=top_by_accuracy,
        top_f1=top_by_f1,
    )


@models_bp.route('/compare', methods=['GET', 'POST'])
@login_required
def compare():
    """模型对比页面 — 并排比较 2-4 个模型的指标 (PRD 第五节)"""
    if request.method == 'POST':
        model_ids = request.form.getlist('model_ids')
        model_ids = [int(mid) for mid in model_ids if mid.isdigit()]
    else:
        model_ids = request.args.getlist('model_ids', type=int)

    # 去重并限制 2-4 个
    model_ids = list(dict.fromkeys(model_ids))[:4]

    models_to_compare = []
    error = None

    if len(model_ids) < 2:
        error = '请至少选择 2 个模型进行对比。'
        models_to_compare = []
    else:
        for mid in model_ids:
            m = ModelService.get_model_by_id(mid)
            if m and (m.is_public or m.owner_id == current_user.id or current_user.is_admin):
                models_to_compare.append(m)

        if len(models_to_compare) < 2:
            error = '没有足够的有权限模型可供对比。'

    # 获取用户的所有模型列表 (供选择)
    user_models_result = ModelService.list_models(owner_id=current_user.id, per_page=100)
    user_models = user_models_result['items']

    # 对比指标列表
    compare_metrics = [
        {'key': 'accuracy', 'label': '准确率 (Accuracy)', 'higher_better': True},
        {'key': 'precision', 'label': '查准率 (Precision)', 'higher_better': True},
        {'key': 'recall', 'label': '召回率 (Recall)', 'higher_better': True},
        {'key': 'f1_score', 'label': 'F1 分数', 'higher_better': True},
        {'key': 'loss', 'label': '损失 (Loss)', 'higher_better': False},
    ]

    # 找出每列的最佳值
    best_values = {}
    if models_to_compare:
        for metric in compare_metrics:
            values = [
                (getattr(m, metric['key']), m) for m in models_to_compare if getattr(m, metric['key']) is not None
            ]
            if values:
                best_values[metric['key']] = (
                    max(values, key=lambda x: x[0])[1].id
                    if metric['higher_better']
                    else min(values, key=lambda x: x[0])[1].id
                )

    return render_template(
        'models/compare.html',
        models=models_to_compare,
        user_models=user_models,
        error=error,
        compare_metrics=compare_metrics,
        best_values=best_values,
        selected_ids=model_ids,
    )


def _build_model_hints(model, metadata, hyperparams, feature_names):
    """根据模型元数据构建输入指导提示

    分析模型名称/数据集/类别标签/特征名等, 生成面向用户的输入指南。
    """
    hints = {
        'model_type': model.model_type or 'general',
        'domain': 'general',
        'domain_label': '通用',
        'classes': [],
        'input_guide': {},
        'capabilities': [],
        'limitations': [],
    }

    task_type = (metadata or {}).get('task_type', model.model_type)
    class_labels = (metadata or {}).get('class_labels', [])
    algorithm = (metadata or {}).get('algorithm', hyperparams.get('algorithm', ''))
    framework = (metadata or {}).get('framework', model.framework or '')
    dataset_name = (model.training_dataset.name if model.training_dataset else '') or ''
    dataset_desc = (model.training_dataset.description if model.training_dataset else '') or ''
    model_name = model.name or ''

    # ── 域名检测 ──
    domain_keywords = {
        'movie': ['douban', 'movie', '电影', '影评', '豆瓣'],
        'shopping': ['shopping', 'taobao', 'jd', '购物', '商品', '电商', '淘宝'],
        'restaurant': ['restaurant', 'food', '餐厅', '美食', '外卖', '点评'],
        'hotel': ['hotel', '酒店', '住宿', '旅行'],
        'finance': ['finance', 'stock', '金融', '股票', '财经'],
        'social': ['weibo', 'twitter', '微博', '社交', '舆情'],
        'news': ['news', '新闻', '头条', '资讯'],
    }
    search_text = (model_name + ' ' + dataset_name + ' ' + dataset_desc).lower()
    detected_domain = 'general'
    for domain, keywords in domain_keywords.items():
        for kw in keywords:
            if kw in search_text:
                detected_domain = domain
                break
        if detected_domain != 'general':
            break

    domain_config = {
        'movie': {
            'label': '电影评论',
            'placeholder': '输入电影评论，例如: 这部电影非常精彩，演员表现出色，剧情扣人心弦。',
            'examples': [
                '这部电影非常精彩，演员表现出色，剧情扣人心弦。',
                '剧情太拖沓了，浪费了两个小时。',
                '特效很棒但故事一般，总体还行。',
            ],
        },
        'shopping': {
            'label': '购物评价',
            'placeholder': '输入商品评价，例如: 质量很好，做工精细，物流也很快，非常满意！',
            'examples': [
                '质量很好，做工精细，物流也很快，非常满意！',
                '跟描述不符，质量很差，不建议购买。',
                '性价比不错，对得起这个价格。',
            ],
        },
        'restaurant': {
            'label': '餐厅点评',
            'placeholder': '输入餐厅评价，例如: 菜品口味地道，环境优雅，服务周到。',
            'examples': ['菜品口味地道，环境优雅，服务周到。', '上菜太慢了，味道也一般。'],
        },
        'hotel': {
            'label': '酒店评价',
            'placeholder': '输入酒店评价，例如: 房间干净整洁，前台服务热情，地理位置方便。',
            'examples': ['房间干净整洁，前台服务热情，地理位置方便。', '隔音太差，一晚上没睡好。'],
        },
        'finance': {
            'label': '财经文本',
            'placeholder': '输入财经相关文本，例如: 今日大盘震荡上行，科技板块领涨。',
            'examples': ['今日大盘震荡上行，科技板块领涨。'],
        },
        'social': {
            'label': '社交媒体',
            'placeholder': '输入社交媒体文本，例如: 今天天气真好，出去玩了一天！',
            'examples': ['今天天气真好，出去玩了一天！'],
        },
        'news': {
            'label': '新闻文本',
            'placeholder': '输入新闻文本，例如: 据新华社报道，今日国家统计局发布了最新经济数据。',
            'examples': ['据新华社报道，今日国家统计局发布了最新经济数据。'],
        },
        'general': {'label': '文本内容', 'placeholder': '输入文本内容进行预测...', 'examples': []},
    }
    dc = domain_config.get(detected_domain, domain_config['general'])
    hints['domain'] = detected_domain
    hints['domain_label'] = dc['label']

    # ── 类别标签 ──
    if class_labels:
        hints['classes'] = [str(c) for c in class_labels]
    elif task_type == 'classification':
        # 尝试从特征名推断 (二分类常见标签)
        hints['classes'] = []

    # ── 按模型类型构建输入指导 ──
    if model.model_type == 'nlp':
        # 输入长度建议
        (metadata or {}).get('text_column', '')
        max_len = (metadata or {}).get('max_length', 0)

        hints['input_guide'] = {
            'type': 'text',
            'min_length': 5,
            'max_length': max_len if max_len else 200,
            'placeholder': dc['placeholder'],
            'examples': dc['examples'],
            'note': (
                f'此模型训练于{dc["label"]}数据'
                + (f'（{dataset_name}）' if dataset_name else '')
                + '。建议输入10字以上的完整句子以获得准确预测。'
            ),
        }
        hints['capabilities'] = ['文本情感分析', '短文本分类']
        hints['limitations'] = [
            '短文本（<5字）预测准确度较低，建议输入完整句子',
            '仅支持中文文本',
            '请勿输入与训练领域无关的内容（如用电影评论模型预测财经文本）',
        ]
        if class_labels:
            hints['capabilities'].append(f'可识别类别: {", ".join(str(c) for c in class_labels)}')

    elif model.model_type in ('classification', 'regression'):
        hints['input_guide'] = {
            'type': 'tabular',
            'feature_count': len(feature_names) if feature_names else 0,
            'note': (
                f'此模型使用 {len(feature_names)} 个特征进行'
                + ('分类' if task_type == 'classification' else '回归')
                + '预测。请手动输入特征值或上传CSV文件批量预测。'
            ),
        }
        hints['capabilities'] = [
            f'{"多分类" if task_type == "classification" else "回归"}预测',
            f'输入特征数: {len(feature_names)}',
        ]
        if class_labels:
            hints['capabilities'].append(f'可预测类别: {", ".join(str(c) for c in class_labels)}')
        hints['limitations'] = [
            '请确保输入特征与训练数据分布一致',
            '异常值可能导致预测偏差',
        ]

    elif model.model_type == 'computer_vision':
        hints['input_guide'] = {
            'type': 'image',
            'formats': ['JPG', 'PNG', 'WebP', 'BMP'],
            'note': (
                '此模型使用 CNN 提取图像特征后进行'
                + ('分类' if task_type == 'classification' else '预测')
                + '。支持常见图像格式，建议图像分辨率不低于224×224。'
            ),
        }
        hints['capabilities'] = ['图像特征提取', '基于CNN的预测']
        hints['limitations'] = [
            '图像质量过低会影响预测准确度',
            '建议使用清晰、光线充足的照片',
        ]

    else:
        hints['input_guide'] = {
            'type': 'general',
            'note': '请根据模型类型选择合适的输入方式。',
        }

    # ── 框架/算法信息 ──
    if algorithm:
        hints['algorithm'] = algorithm
    if framework:
        hints['framework'] = framework

    return hints


@models_bp.route('/<int:model_id>/test', methods=['GET', 'POST'])
@login_required
def test_model(model_id):
    """模型测试页面 — 上传数据并执行预测"""
    model = ModelService.get_model_by_id(model_id)
    if not model:
        flash('模型不存在。', 'danger')
        return redirect(url_for('models.list_models'))

    if not model.is_viewable_by(current_user):
        flash('您没有权限测试此模型。', 'danger')
        return redirect(url_for('models.list_models'))

    from app.services.inference_service import ModelInferenceService

    result = None
    feature_importance = None
    test_report = None
    error = None
    preview_data = None
    preview_columns = None
    image_preview = None  # base64 缩略图 (CV 模型上传预览)

    if request.method == 'POST':
        action = request.form.get('action', 'predict')

        if action == 'predict_file':
            # 上传文件进行预测
            file = request.files.get('test_file')
            if file and file.filename:
                try:
                    import pandas as pd

                    fmt = file.filename.rsplit('.', 1)[-1].lower()
                    if fmt == 'csv':
                        df = pd.read_csv(file)
                    elif fmt in ('xlsx', 'xls'):
                        df = pd.read_excel(file)
                    elif fmt == 'json':
                        df = pd.read_json(file)
                    else:
                        error = f'不支持的文件格式: {fmt}'
                        df = None

                    if df is not None:
                        result = ModelInferenceService.predict(model, df)
                        if result.get('success') and len(df) <= 100:
                            preview_data = df.head(20).values.tolist()
                            preview_columns = list(df.columns)
                except Exception as e:
                    error = f'文件解析失败: {str(e)}'

        elif action == 'predict_manual':
            # 手动输入特征值 — 使用模型真实特征名
            try:
                import pandas as pd

                fnames_json = request.form.get('feature_names_json', '[]')
                try:
                    fnames = json.loads(fnames_json)
                except Exception:
                    fnames = []
                feature_count = int(request.form.get('feature_count', 0))
                manual_data = {}
                for i in range(feature_count):
                    val = request.form.get(f'feature_{i}', '')
                    if val:
                        real_name = fnames[i] if i < len(fnames) else f'feature_{i}'
                        manual_data[real_name] = [float(val)]
                if manual_data:
                    df = pd.DataFrame(manual_data)
                    result = ModelInferenceService.predict(model, df)
                else:
                    error = '请至少输入一个特征值后再点击预测。'
            except Exception as e:
                error = f'输入解析失败: {str(e)}'

        elif action == 'predict_text':
            # NLP 模型: 原始文本 → TF-IDF 特征 → 预测
            text_input = request.form.get('text_input', '').strip()
            if not text_input:
                error = '请输入文本内容后再点击预测。'
            else:
                try:
                    import pandas as pd

                    # 优先使用训练时保存的 TF-IDF vectorizer
                    _, _md, _, _le = ModelInferenceService.load_model(model)
                    _vec = (_md or {}).get('vectorizer')
                    _fn = (_md or {}).get('feature_names', [])
                    if _vec is not None:
                        X_vec = _vec.transform([text_input])
                        X_dense = X_vec.toarray()
                        _tfn = [c for c in _fn if str(c).startswith('tfidf_')]
                        if _tfn and len(_tfn) == X_dense.shape[1]:
                            df = pd.DataFrame(X_dense, columns=_tfn)
                        else:
                            df = pd.DataFrame(X_dense)
                    else:
                        from app.services.feature_extractor import FeatureExtractor

                        n_features = len(_fn) if _fn else 100
                        features, feat_error = FeatureExtractor.extract_text_features(text_input, max(n_features, 10))
                        if feat_error:
                            error = feat_error
                            df = None
                        else:
                            df = pd.DataFrame(
                                features,
                                columns=[_fn[i] if i < len(_fn) else f'feature_{i}' for i in range(features.shape[1])],
                            )
                    if df is not None:
                        result = ModelInferenceService.predict(model, df)
                except Exception as e:
                    error = f'文本预测失败: {str(e)}'

        elif action == 'predict_image':
            # CV 模型: 上传图像 → CNN特征提取 → 预测
            img_file = request.files.get('image_file')
            if not img_file or not img_file.filename:
                error = '请选择图像文件后再点击预测。'
            else:
                try:
                    from app.services.feature_extractor import FeatureExtractor

                    _, _md, _, _le = ModelInferenceService.load_model(model)
                    _fn = (_md or {}).get('feature_names', [])
                    image_data = img_file.read()
                    n_features = len(_fn) if _fn else 100
                    features, feat_error = FeatureExtractor.extract_image_features(image_data, max(n_features, 10))
                    if feat_error:
                        error = feat_error
                    else:
                        import pandas as pd

                        df = pd.DataFrame(
                            features,
                            columns=[_fn[i] if i < len(_fn) else f'feature_{i}' for i in range(features.shape[1])],
                        )
                        result = ModelInferenceService.predict(model, df)
                        # 生成缩略图用于前端回显
                        thumb = FeatureExtractor.load_image_thumbnail(image_data)
                        if thumb:
                            import base64

                            image_preview = base64.b64encode(thumb).decode('utf-8')
                except Exception as e:
                    error = f'图像预测失败: {str(e)}'

        elif action == 'run_test':
            # 使用原始数据集进行完整测试评估
            test_report = ModelInferenceService.test_model_with_split(model)
            if test_report.get('success'):
                flash('模型评估完成！', 'success')
            else:
                error = test_report.get('error', '评估失败')

        elif action == 'feature_importance':
            feature_importance = ModelInferenceService.get_feature_importance(model)
            if not feature_importance.get('success'):
                error = feature_importance.get('error')

    # 获取特征名 + 模型文件状态 (优先从模型元数据，其次从数据集)
    feature_names = []
    model_file_error = None
    hyperparams = model.hyperparameters_dict
    # 尝试从模型文件元数据中获取 (PyTorch/sklearn/Transformers saved config)
    try:
        from app.services.inference_service import ModelInferenceService

        _, metadata, _, load_error = ModelInferenceService.load_model(model)
        if load_error:
            model_file_error = load_error
        if metadata and metadata.get('feature_names'):
            feature_names = list(metadata['feature_names'])
    except Exception:
        metadata = None
    # 回退: 从数据集 summary 中获取
    if not feature_names and model.training_dataset and model.training_dataset.summary_json:
        try:
            summary = (
                json.loads(model.training_dataset.summary_json)
                if isinstance(model.training_dataset.summary_json, str)
                else model.training_dataset.summary_json
            )
            cols = summary.get('columns', [])
            target_col = hyperparams.get('target_column', cols[-1] if cols else None)
            feature_names = [c for c in cols if c != target_col]
        except Exception:
            pass

    return render_template(
        'models/test.html',
        model=model,
        result=result,
        feature_importance=feature_importance,
        test_report=test_report,
        error=error,
        feature_names=feature_names,
        preview_data=preview_data,
        preview_columns=preview_columns,
        model_file_error=model_file_error,
        image_preview=image_preview,
        hints=_build_model_hints(model, metadata, hyperparams, feature_names),
    )
