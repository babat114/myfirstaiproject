"""
============================================
AI模型 Web 路由
模型注册表管理的页面路由
============================================
"""
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
    """模型列表页面"""
    page = request.args.get('page', 1, type=int)
    model_type = request.args.get('model_type')
    framework = request.args.get('framework')
    search = request.args.get('search', '').strip() or None

    result = ModelService.list_models(
        page=page,
        owner_id=current_user.id,
        model_type=model_type,
        framework=framework,
        search=search,
    )

    return render_template(
        'models/list.html',
        models=result['items'],
        pagination=result,
        current_type=model_type,
        current_framework=framework,
        search_query=search,
    )


@models_bp.route('/public')
@login_required
def public_models():
    """公开模型浏览"""
    page = request.args.get('page', 1, type=int)
    model_type = request.args.get('model_type')
    framework = request.args.get('framework')
    search = request.args.get('search', '').strip() or None

    result = ModelService.list_models(
        page=page,
        model_type=model_type,
        framework=framework,
        search=search,
    )

    return render_template(
        'models/public.html',
        models=result['items'],
        pagination=result,
        current_type=model_type,
        current_framework=framework,
        search_query=search,
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

    if not model.is_public and model.owner_id != current_user.id and not current_user.is_admin:
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

    if model.owner_id != current_user.id and not current_user.is_admin:
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
    if not model or model.owner_id != current_user.id:
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
