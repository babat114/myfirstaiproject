"""
============================================
AI模型 Web 路由
模型注册表管理的页面路由
============================================
"""
import json
from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, current_app
)
from flask_login import login_required, current_user
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
            page=page, model_type=model_type, framework=framework,
            search=search, is_public=True,
            sort_by=sort_by, sort_order=sort_order,
        )
    elif visibility == 'all':
        # 我的 + 其他公开模型
        result = ModelService.list_models(
            page=page, model_type=model_type, framework=framework,
            search=search, owner_id=current_user.id, include_public=True,
            sort_by=sort_by, sort_order=sort_order,
        )
    else:
        # 仅我的模型
        result = ModelService.list_models(
            page=page, model_type=model_type, framework=framework,
            search=search, owner_id=current_user.id,
            sort_by=sort_by, sort_order=sort_order,
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
        for k, v in zip(hp_keys, hp_values):
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

    return render_template('models/detail.html', model=model)


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
        for k, v in zip(hp_keys, hp_values):
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

    success, error = ModelService.upload_model_file(
        model, file, upload_folder=current_app.config['UPLOAD_FOLDER']
    )
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
            if m:
                # 权限检查
                if m.is_public or m.owner_id == current_user.id or current_user.is_admin:
                    models_to_compare.append(m)

        if len(models_to_compare) < 2:
            error = '没有足够的有权限模型可供对比。'

    # 获取用户的所有模型列表 (供选择)
    user_models_result = ModelService.list_models(
        owner_id=current_user.id, per_page=100
    )
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
            values = [(getattr(m, metric['key']), m) for m in models_to_compare
                      if getattr(m, metric['key']) is not None]
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
                # 获取特征名
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
                        # 用真实特征名作为 DataFrame 列名
                        real_name = fnames[i] if i < len(fnames) else f'feature_{i}'
                        manual_data[real_name] = [float(val)]
                if manual_data:
                    df = pd.DataFrame(manual_data)
                    result = ModelInferenceService.predict(model, df)
            except Exception as e:
                error = f'输入解析失败: {str(e)}'

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

    # 获取特征名 (优先从模型元数据，其次从数据集)
    feature_names = []
    hyperparams = model.hyperparameters_dict
    # 尝试从模型文件元数据中获取 (PyTorch/sklearn saved config)
    try:
        from app.services.inference_service import ModelInferenceService
        _, metadata, _ = ModelInferenceService.load_model(model)
        if metadata and metadata.get('feature_names'):
            feature_names = list(metadata['feature_names'])
    except Exception:
        metadata = None
    # 回退: 从数据集 summary 中获取
    if not feature_names and model.training_dataset and model.training_dataset.summary_json:
        try:
            summary = json.loads(model.training_dataset.summary_json) if isinstance(model.training_dataset.summary_json, str) else model.training_dataset.summary_json
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
    )
