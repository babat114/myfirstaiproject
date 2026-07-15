"""
============================================
数据集 Web 路由
数据集管理的页面路由
============================================
"""
import contextlib
import json
import os

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from app.services.dataset_service import DatasetService

datasets_bp = Blueprint('datasets', __name__)


def _get_data_preview(dataset, max_rows=20):
    """读取数据集文件，返回前 N 行预览数据和列信息"""
    try:
        import pandas as pd

        file_path = dataset.file_path
        if not file_path or not os.path.exists(file_path):
            return None

        fmt = dataset.file_format.lower()
        if fmt == 'csv':
            df = pd.read_csv(file_path, nrows=max_rows)
        elif fmt in ('xlsx', 'xls'):
            df = pd.read_excel(file_path, nrows=max_rows)
        elif fmt == 'json':
            df = pd.read_json(file_path, nrows=max_rows)
        elif fmt == 'parquet':
            df = pd.read_parquet(file_path)
            df = df.head(max_rows)
        elif fmt == 'txt':
            df = pd.read_csv(file_path, sep='\t', nrows=max_rows)
        else:
            return None

        # 列类型信息
        columns_info = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            missing = int(df[col].isna().sum())
            unique = int(df[col].nunique())
            columns_info.append({
                'name': col,
                'dtype': dtype_str,
                'missing': missing,
                'unique': unique,
            })

        return {
            'columns': columns_info,
            'rows': df.values.tolist(),
            'total_preview_rows': len(df),
            'shape': (dataset.row_count, dataset.column_count) if dataset.row_count else (len(df), len(df.columns)),
        }
    except Exception as e:
        return {'error': str(e)}


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
    if not dataset.is_viewable_by(current_user):
        flash('您没有权限查看此数据集。', 'danger')
        return redirect(url_for('datasets.list_datasets'))

    # 获取数据预览
    preview = _get_data_preview(dataset)

    # 解析 summary_json
    summary = {}
    if dataset.summary_json:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            summary = json.loads(dataset.summary_json) if isinstance(dataset.summary_json, str) else dataset.summary_json

    return render_template(
        'datasets/detail.html',
        dataset=dataset,
        preview=preview,
        summary=summary,
    )


@datasets_bp.route('/<int:dataset_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_dataset(dataset_id):
    """编辑数据集"""
    dataset = DatasetService.get_dataset_by_id(dataset_id)
    if not dataset:
        flash('数据集不存在。', 'danger')
        return redirect(url_for('datasets.list_datasets'))

    if not dataset.is_editable_by(current_user):
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


@datasets_bp.route('/import', methods=['GET'])
@login_required
def import_datasets():
    """公开数据集导入页面"""
    from app.services.dataset_import_service import DatasetImportService
    page = request.args.get('page', 1, type=int)
    category = request.args.get('category')
    search = request.args.get('search', '').strip().lower()

    all_datasets = DatasetImportService.get_available_datasets(category=category or None)
    all_categories = DatasetImportService.get_categories()

    # 搜索过滤
    if search:
        all_datasets = [
            d for d in all_datasets
            if search in d['name'].lower() or search in d['description'].lower() or search in d['key'].lower()
        ]

    # 简单分页
    per_page = 12
    total = len(all_datasets)
    pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    end = start + per_page
    paginated = all_datasets[start:end]

    return render_template(
        'datasets/import.html',
        datasets=paginated,
        categories=all_categories,
        current_category=category,
        search_query=search,
        pagination={
            'total': total,
            'pages': pages,
            'current_page': page,
            'has_next': page < pages,
            'has_prev': page > 1,
        }
    )


@datasets_bp.route('/import/<dataset_key>', methods=['POST'])
@login_required
def do_import_dataset(dataset_key):
    """执行公开数据集导入"""
    from app.services.dataset_import_service import DatasetImportService
    name = request.form.get('name', '').strip() or None
    dataset, error = DatasetImportService.import_dataset(current_user, dataset_key, name=name)
    if error:
        flash(error, 'danger')
    else:
        flash(f'数据集 "{dataset.name}" 导入成功！({dataset.row_count} 行, {dataset.column_count} 列)', 'success')
    return redirect(url_for('datasets.import_datasets'))


@datasets_bp.route('/import/url', methods=['POST'])
@login_required
def import_from_url():
    """从 URL 导入数据集"""
    from app.services.dataset_import_service import DatasetImportService
    url = request.form.get('url', '').strip()
    name = request.form.get('name', '').strip()
    target_column = request.form.get('target_column', '').strip() or None
    description = request.form.get('description', '').strip() or None

    if not url or not name:
        flash('请填写URL和数据集名称。', 'danger')
        return redirect(url_for('datasets.import_datasets'))

    dataset, error = DatasetImportService.import_from_url(
        current_user, url, name, target_column=target_column, description=description
    )
    if error:
        flash(error, 'danger')
    else:
        flash(f'URL数据集 "{dataset.name}" 导入成功！({dataset.row_count} 行)', 'success')
    return redirect(url_for('datasets.import_datasets'))


@datasets_bp.route('/<int:dataset_id>/copy-and-train', methods=['POST'])
@login_required
def copy_and_train(dataset_id):
    """复制公开数据集到当前用户目录，然后跳转到训练创建页"""
    dataset = DatasetService.get_dataset_by_id(dataset_id)
    if not dataset:
        flash('数据集不存在。', 'danger')
        return redirect(url_for('datasets.list_datasets'))

    # 如果已经是自己的数据集，直接跳转训练
    if dataset.owner_id == current_user.id:
        return redirect(url_for('training.create_job', dataset_id=dataset.id))

    # 必须是公开数据集或有查看权限
    if not dataset.is_viewable_by(current_user):
        flash('您没有权限访问此数据集。', 'danger')
        return redirect(url_for('datasets.list_datasets'))

    # 复制到用户目录
    new_dataset, error = DatasetService.copy_dataset_to_user(dataset, current_user)
    if error:
        flash(f'复制数据集失败: {error}', 'danger')
        return redirect(url_for('datasets.dataset_detail', dataset_id=dataset.id))

    flash(f'数据集 "{dataset.name}" 已添加到您的目录！', 'success')
    return redirect(url_for('training.create_job', dataset_id=new_dataset.id))


@datasets_bp.route('/<int:dataset_id>/delete', methods=['POST'])
@login_required
def delete_dataset(dataset_id):
    """删除数据集"""
    dataset = DatasetService.get_dataset_by_id(dataset_id)
    if not dataset:
        flash('数据集不存在。', 'danger')
        return redirect(url_for('datasets.list_datasets'))

    if not dataset.is_editable_by(current_user):
        flash('您没有权限删除此数据集。', 'danger')
        return redirect(url_for('datasets.list_datasets'))

    success, error = DatasetService.delete_dataset(dataset)
    if success:
        flash('数据集已删除。', 'success')
    else:
        flash(error, 'danger')

    return redirect(url_for('datasets.list_datasets'))
