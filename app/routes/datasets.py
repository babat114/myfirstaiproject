"""
============================================
数据集 Web 路由
数据集管理的页面路由
============================================
"""
from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, current_app
)
from flask_login import login_required, current_user
from app.services.dataset_service import DatasetService

datasets_bp = Blueprint('datasets', __name__)


@datasets_bp.route('/')
@login_required
def list_datasets():
    """数据集列表页面"""
    page = request.args.get('page', 1, type=int)
    category = request.args.get('category')
    search = request.args.get('search', '').strip() or None

    result = DatasetService.list_datasets(
        page=page,
        owner_id=current_user.id,
        category=category,
        search=search,
    )

    return render_template(
        'datasets/list.html',
        datasets=result['items'],
        pagination=result,
        current_category=category,
        search_query=search,
    )


@datasets_bp.route('/public')
@login_required
def public_datasets():
    """公开数据集浏览"""
    page = request.args.get('page', 1, type=int)
    category = request.args.get('category')
    search = request.args.get('search', '').strip() or None

    result = DatasetService.list_datasets(
        page=page,
        category=category,
        search=search,
        public_only=True,
    )

    return render_template(
        'datasets/public.html',
        datasets=result['items'],
        pagination=result,
        current_category=category,
        search_query=search,
    )


@datasets_bp.route('/create', methods=['GET', 'POST'])
@login_required
def create_dataset():
    """创建新数据集"""
    if not current_user.can_upload:
        flash('您没有上传权限。', 'danger')
        return redirect(url_for('datasets.list_datasets'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip() or None
        category = request.form.get('category', 'other')
        tags_str = request.form.get('tags', '').strip()
        is_public = request.form.get('is_public') == 'on'
        file = request.files.get('file')

        if not name:
            flash('请输入数据集名称。', 'danger')
            return render_template('datasets/create.html')

        tags = [t.strip() for t in tags_str.split(',') if t.strip()] if tags_str else None

        dataset, error = DatasetService.create_dataset(
            user=current_user,
            name=name,
            file=file,
            description=description,
            category=category,
            tags=tags,
            is_public=is_public,
            upload_folder=current_app.config['UPLOAD_FOLDER'],
        )

        if error:
            flash(error, 'danger')
            return render_template('datasets/create.html')

        flash(f'数据集 "{dataset.name}" 创建成功！', 'success')
        return redirect(url_for('datasets.dataset_detail', dataset_id=dataset.id))

    return render_template('datasets/create.html')


@datasets_bp.route('/<int:dataset_id>')
@login_required
def dataset_detail(dataset_id):
    """数据集详情页面"""
    dataset = DatasetService.get_dataset_by_id(dataset_id)
    if not dataset:
        flash('数据集不存在。', 'danger')
        return redirect(url_for('datasets.list_datasets'))

    # 权限检查
    if not dataset.is_public and dataset.owner_id != current_user.id and not current_user.is_admin:
        flash('您没有权限查看此数据集。', 'danger')
        return redirect(url_for('datasets.list_datasets'))

    return render_template('datasets/detail.html', dataset=dataset)


@datasets_bp.route('/<int:dataset_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_dataset(dataset_id):
    """编辑数据集"""
    dataset = DatasetService.get_dataset_by_id(dataset_id)
    if not dataset:
        flash('数据集不存在。', 'danger')
        return redirect(url_for('datasets.list_datasets'))

    if dataset.owner_id != current_user.id and not current_user.is_admin:
        flash('您没有权限编辑此数据集。', 'danger')
        return redirect(url_for('datasets.list_datasets'))

    if request.method == 'POST':
        data = {
            'name': request.form.get('name', '').strip(),
            'description': request.form.get('description', '').strip() or None,
            'category': request.form.get('category'),
            'is_public': request.form.get('is_public') == 'on',
            'tags': [t.strip() for t in request.form.get('tags', '').split(',') if t.strip()],
        }

        success, error = DatasetService.update_dataset(dataset, data)
        if success:
            flash('数据集已更新。', 'success')
            return redirect(url_for('datasets.dataset_detail', dataset_id=dataset.id))
        else:
            flash(error, 'danger')

    return render_template('datasets/edit.html', dataset=dataset)


@datasets_bp.route('/<int:dataset_id>/delete', methods=['POST'])
@login_required
def delete_dataset(dataset_id):
    """删除数据集"""
    dataset = DatasetService.get_dataset_by_id(dataset_id)
    if not dataset:
        flash('数据集不存在。', 'danger')
        return redirect(url_for('datasets.list_datasets'))

    if dataset.owner_id != current_user.id and not current_user.is_admin:
        flash('您没有权限删除此数据集。', 'danger')
        return redirect(url_for('datasets.list_datasets'))

    success, error = DatasetService.delete_dataset(dataset)
    if success:
        flash('数据集已删除。', 'success')
    else:
        flash(error, 'danger')

    return redirect(url_for('datasets.list_datasets'))
