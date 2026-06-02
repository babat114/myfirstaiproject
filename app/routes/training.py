"""
============================================
训练任务 Web 路由
训练任务管理的页面路由
============================================
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from app.services.training_service import TrainingService
from app.services.dataset_service import DatasetService

training_bp = Blueprint('training', __name__)


@training_bp.route('/')
@login_required
def list_jobs():
    """训练任务列表"""
    page = request.args.get('page', 1, type=int)
    status = request.args.get('status')
    search = request.args.get('search', '').strip() or None

    result = TrainingService.list_jobs(
        page=page,
        owner_id=current_user.id,
        status=status,
        search=search,
    )

    return render_template(
        'training/list.html',
        jobs=result['items'],
        pagination=result,
        current_status=status,
        search_query=search,
    )


@training_bp.route('/create', methods=['GET', 'POST'])
@login_required
def create_job():
    """创建新训练任务"""
    # 获取用户的数据集列表
    datasets_result = DatasetService.list_datasets(
        owner_id=current_user.id, per_page=100
    )
    datasets = datasets_result['items']

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip() or None
        dataset_id = request.form.get('dataset_id', type=int)
        task_type = request.form.get('task_type', 'training')
        framework = request.form.get('framework', '').strip() or None
        total_epochs = request.form.get('total_epochs', 0, type=int)
        total_steps = request.form.get('total_steps', 0, type=int)
        gpu_count = request.form.get('gpu_count', 0, type=int)
        cpu_cores = request.form.get('cpu_cores', 1, type=int)
        memory_gb = request.form.get('memory_gb', 4.0, type=float)

        if not name:
            flash('请输入任务名称。', 'danger')
            return render_template('training/create.html', datasets=datasets)

        job, error = TrainingService.create_job(
            user=current_user,
            name=name,
            dataset_id=dataset_id,
            description=description,
            task_type=task_type,
            framework=framework,
            total_epochs=total_epochs,
            total_steps=total_steps,
            gpu_count=gpu_count,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
        )

        if error:
            flash(error, 'danger')
            return render_template('training/create.html', datasets=datasets)

        flash(f'训练任务 "{job.name}" 创建成功！', 'success')
        return redirect(url_for('training.job_detail', job_id=job.id))

    return render_template('training/create.html', datasets=datasets)


@training_bp.route('/<int:job_id>')
@login_required
def job_detail(job_id):
    """训练任务详情"""
    job = TrainingService.get_job_by_id(job_id)
    if not job:
        flash('训练任务不存在。', 'danger')
        return redirect(url_for('training.list_jobs'))

    if job.owner_id != current_user.id and not current_user.is_admin:
        flash('您没有权限查看此任务。', 'danger')
        return redirect(url_for('training.list_jobs'))

    return render_template('training/detail.html', job=job)


@training_bp.route('/<int:job_id>/start', methods=['POST'])
@login_required
def start_job(job_id):
    """启动训练"""
    job = TrainingService.get_job_by_id(job_id)
    if not job or job.owner_id != current_user.id:
        flash('任务不存在或权限不足。', 'danger')
        return redirect(url_for('training.list_jobs'))

    success, error = TrainingService.start_job(job)
    if success:
        flash('训练任务已启动！', 'success')
    else:
        flash(error, 'danger')

    return redirect(url_for('training.job_detail', job_id=job.id))


@training_bp.route('/<int:job_id>/pause', methods=['POST'])
@login_required
def pause_job(job_id):
    """暂停训练"""
    job = TrainingService.get_job_by_id(job_id)
    if not job or job.owner_id != current_user.id:
        flash('任务不存在或权限不足。', 'danger')
        return redirect(url_for('training.list_jobs'))

    success, error = TrainingService.pause_job(job)
    if success:
        flash('训练已暂停。', 'info')
    else:
        flash(error, 'danger')

    return redirect(url_for('training.job_detail', job_id=job.id))


@training_bp.route('/<int:job_id>/cancel', methods=['POST'])
@login_required
def cancel_job(job_id):
    """取消训练"""
    job = TrainingService.get_job_by_id(job_id)
    if not job or job.owner_id != current_user.id:
        flash('任务不存在或权限不足。', 'danger')
        return redirect(url_for('training.list_jobs'))

    success, error = TrainingService.cancel_job(job)
    if success:
        flash('训练任务已取消。', 'info')
    else:
        flash(error, 'danger')

    return redirect(url_for('training.job_detail', job_id=job.id))


@training_bp.route('/<int:job_id>/delete', methods=['POST'])
@login_required
def delete_job(job_id):
    """删除训练任务"""
    job = TrainingService.get_job_by_id(job_id)
    if not job:
        flash('任务不存在。', 'danger')
        return redirect(url_for('training.list_jobs'))

    if job.owner_id != current_user.id and not current_user.is_admin:
        flash('您没有权限删除此任务。', 'danger')
        return redirect(url_for('training.list_jobs'))

    success, error = TrainingService.delete_job(job)
    if success:
        flash('训练任务已删除。', 'success')
    else:
        flash(error, 'danger')

    return redirect(url_for('training.list_jobs'))
