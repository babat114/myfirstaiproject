"""
============================================
仪表盘路由
首页和数据总览
============================================
"""
from flask import Blueprint, render_template, flash, redirect, url_for
from flask_login import login_required, current_user
from app.services.dataset_service import DatasetService
from app.services.model_service import ModelService
from app.services.training_service import TrainingService

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/')
@login_required
def index():
    """主仪表盘 - 显示用户的所有统计概览"""
    dataset_stats = DatasetService.get_dataset_statistics(user_id=current_user.id)
    model_stats = ModelService.get_model_statistics(user_id=current_user.id)
    job_stats = TrainingService.get_job_statistics(user_id=current_user.id)

    # 获取最近的记录
    recent_datasets = DatasetService.list_datasets(
        owner_id=current_user.id, per_page=5
    )['items']

    recent_models = ModelService.list_models(
        owner_id=current_user.id, per_page=5
    )['items']

    recent_jobs = TrainingService.list_jobs(
        owner_id=current_user.id, per_page=5
    )['items']

    return render_template(
        'dashboard.html',
        dataset_stats=dataset_stats,
        model_stats=model_stats,
        job_stats=job_stats,
        recent_datasets=recent_datasets,
        recent_models=recent_models,
        recent_jobs=recent_jobs,
    )


@dashboard_bp.route('/admin')
@login_required
def admin():
    """管理员面板 - 全局统计 (仅管理员)"""
    if not current_user.is_admin:
        flash('您没有管理员权限。', 'danger')
        return redirect(url_for('dashboard.index'))

    dataset_stats = DatasetService.get_dataset_statistics()
    model_stats = ModelService.get_model_statistics()
    job_stats = TrainingService.get_job_statistics()

    return render_template(
        'admin.html',
        dataset_stats=dataset_stats,
        model_stats=model_stats,
        job_stats=job_stats,
    )


@dashboard_bp.route('/admin/users')
@login_required
def admin_users():
    """用户管理页面 (仅管理员)"""
    if not current_user.is_admin:
        flash('您没有管理员权限。', 'danger')
        return redirect(url_for('dashboard.index'))

    from app.services.auth_service import AuthService
    users_result = AuthService.list_users(per_page=200)

    return render_template(
        'admin/users.html',
        users=users_result['users'],
    )
