"""
============================================
训练任务 Web 路由
训练任务管理的页面路由
============================================
"""
import json
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

    # 解析数据集列信息 (从 summary_json 读取)
    datasets_with_cols = []
    for ds in datasets:
        cols = []
        if isinstance(ds, dict):
            summary_str = ds.get('summary_json', '{}')
            try:
                summary = json.loads(summary_str) if isinstance(summary_str, str) else summary_str
                cols = summary.get('columns', [])
            except (json.JSONDecodeError, TypeError):
                pass
            ds_copy = ds.copy()
        else:
            ds_copy = ds.to_dict()
        ds_copy['columns'] = cols
        datasets_with_cols.append(ds_copy)

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip() or None
        dataset_id = request.form.get('dataset_id', type=int)
        task_type = request.form.get('task_type', 'training')
        framework = request.form.get('framework', 'sklearn').strip() or 'sklearn'
        total_epochs = request.form.get('total_epochs', 10, type=int)
        total_steps = request.form.get('total_steps', 0, type=int)
        gpu_count = request.form.get('gpu_count', 0, type=int)
        cpu_cores = request.form.get('cpu_cores', 1, type=int)
        memory_gb = request.form.get('memory_gb', 4.0, type=float)

        # 新增：ML 任务参数
        ml_task_type = request.form.get('ml_task_type', 'classification')
        algorithm = request.form.get('algorithm', 'random_forest')
        target_column = request.form.get('target_column', '').strip() or None
        test_size = request.form.get('test_size', 0.2, type=float)
        start_immediately = request.form.get('start_immediately') == 'on'

        if not name:
            flash('请输入任务名称。', 'danger')
            return render_template('training/create.html', datasets=datasets_with_cols)

        if not dataset_id:
            flash('请选择数据集。', 'danger')
            return render_template('training/create.html', datasets=datasets_with_cols)

        # 根据 algorithm 自动调整 total_epochs
        if algorithm in ('random_forest', 'gradient_boosting', 'random_forest_regressor',
                         'gradient_boosting_regressor', 'svm', 'svr'):
            total_epochs = 1  # sklearn 的 fit 一次性
        elif total_epochs <= 0:
            total_epochs = 10

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
            ml_task_type=ml_task_type,
            algorithm=algorithm,
            target_column=target_column,
            test_size=test_size,
        )

        if error:
            flash(error, 'danger')
            return render_template('training/create.html', datasets=datasets_with_cols)

        if start_immediately:
            success, start_error = TrainingService.start_job(job)
            if start_error:
                flash(f'任务已创建，但启动失败: {start_error}', 'warning')
            else:
                flash(f'训练任务 "{job.name}" 已创建并启动！', 'success')
        else:
            flash(f'训练任务 "{job.name}" 创建成功！', 'success')

        return redirect(url_for('training.job_detail', job_id=job.id))

    return render_template('training/create.html', datasets=datasets_with_cols)


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


@training_bp.route('/<int:job_id>/resume', methods=['POST'])
@login_required
def resume_job(job_id):
    """恢复训练"""
    job = TrainingService.get_job_by_id(job_id)
    if not job or job.owner_id != current_user.id:
        flash('任务不存在或权限不足。', 'danger')
        return redirect(url_for('training.list_jobs'))

    success, error = TrainingService.resume_job(job)
    if success:
        flash('训练已恢复。', 'success')
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


@training_bp.route('/<int:job_id>/retrain', methods=['POST'])
@login_required
def retrain_job(job_id):
    """重新训练 (重置并启动) — 适用于 failed/paused/cancelled/completed 状态"""
    job = TrainingService.get_job_by_id(job_id)
    if not job:
        flash('任务不存在。', 'danger')
        return redirect(url_for('training.list_jobs'))

    if job.owner_id != current_user.id and not current_user.is_admin:
        flash('您没有权限操作此任务。', 'danger')
        return redirect(url_for('training.list_jobs'))

    success, error = TrainingService.retrain_job(job)
    if success:
        flash(f'训练任务 "{job.name}" 已重置并重新启动！', 'success')
    else:
        flash(error, 'danger')

    return redirect(url_for('training.job_detail', job_id=job.id))


@training_bp.route('/<int:job_id>/retrain-with-params', methods=['POST'])
@login_required
def retrain_job_with_params(job_id):
    """使用新参数重新训练 — 用户在线修改参数后重新训练"""
    job = TrainingService.get_job_by_id(job_id)
    if not job:
        flash('任务不存在。', 'danger')
        return redirect(url_for('training.list_jobs'))

    if job.owner_id != current_user.id and not current_user.is_admin:
        flash('您没有权限操作此任务。', 'danger')
        return redirect(url_for('training.list_jobs'))

    # 解析用户提交的新参数
    new_params = {}
    for key in ('learning_rate', 'batch_size', 'epochs', 'total_epochs',
                'test_size', 'dropout', 'weight_decay', 'algorithm',
                'ml_task_type', 'framework', 'hidden_layers_str'):
        val = request.form.get(key)
        if val is not None and val != '':
            new_params[key] = val

    # 转换数值类型
    for num_key in ('learning_rate', 'test_size', 'dropout', 'weight_decay'):
        if num_key in new_params:
            try:
                new_params[num_key] = float(new_params[num_key])
            except (ValueError, TypeError):
                flash(f'参数 {num_key} 格式无效。', 'danger')
                return redirect(url_for('training.job_detail', job_id=job.id))
    for int_key in ('batch_size', 'epochs', 'total_epochs'):
        if int_key in new_params:
            try:
                new_params[int_key] = int(new_params[int_key])
            except (ValueError, TypeError):
                flash(f'参数 {int_key} 格式无效。', 'danger')
                return redirect(url_for('training.job_detail', job_id=job.id))

    # 处理 hidden_layers (逗号分隔字符串 → 列表)
    if 'hidden_layers_str' in new_params:
        try:
            layers = [int(x.strip()) for x in new_params.pop('hidden_layers_str').split(',') if x.strip()]
            if layers:
                new_params['hidden_layers'] = layers
        except (ValueError, TypeError):
            flash('hidden_layers 格式无效，请使用逗号分隔的整数如: 128,64,32', 'danger')
            return redirect(url_for('training.job_detail', job_id=job.id))

    # total_epochs 合并
    if 'epochs' in new_params and 'total_epochs' not in new_params:
        new_params['total_epochs'] = new_params.pop('epochs')

    if not new_params:
        # 用户没有指定新参数，使用原参数重新训练
        success, error = TrainingService.retrain_job(job)
        msg = '已使用原参数重新训练！'
    else:
        success, error = TrainingService.retrain_job_with_params(job, new_params)
        msg = f'已使用新参数重新训练！'

    if success:
        flash(msg, 'success')
    else:
        flash(error, 'danger')

    return redirect(url_for('training.job_detail', job_id=job.id))


@training_bp.route('/tuning', methods=['GET', 'POST'])
@login_required
def hyperparameter_tuning():
    """超参数自动调优页面"""
    from app.services.hyperparameter_tuning import HyperparameterTuningService, SEARCH_SPACES

    datasets_result = DatasetService.list_datasets(owner_id=current_user.id, per_page=100)
    datasets = datasets_result['items']

    # 解析列信息
    datasets_with_cols = []
    for ds in datasets:
        cols = []
        ds_dict = ds.to_dict() if hasattr(ds, 'to_dict') else ds
        try:
            summary = json.loads(ds_dict.get('summary_json', '{}'))
            cols = summary.get('columns', [])
        except Exception:
            pass
        ds_dict['columns'] = cols
        datasets_with_cols.append(ds_dict)

    tuning_result = None
    error = None

    if request.method == 'POST':
        dataset_id = request.form.get('dataset_id', type=int)
        algorithm = request.form.get('algorithm', 'random_forest')
        task_type = request.form.get('ml_task_type', 'classification')
        target_column = request.form.get('target_column', '').strip() or None
        tuning_method = request.form.get('tuning_method', 'random')
        n_iter = request.form.get('n_iter', 30, type=int)
        cv = request.form.get('cv', 5, type=int)
        start_training = request.form.get('start_training') == 'on'

        if not dataset_id:
            error = '请选择数据集。'
        else:
            dataset = DatasetService.get_dataset_by_id(dataset_id)
            if not dataset:
                error = '数据集不存在。'
            else:
                if not target_column:
                    target_column = None

                job, tuning_result, err = HyperparameterTuningService.create_tuned_training(
                    user=current_user,
                    dataset=dataset,
                    algorithm=algorithm,
                    task_type=task_type,
                    target_column=target_column,
                    tuning_method=tuning_method,
                    n_iter=n_iter,
                    cv=cv,
                    epochs=0 if algorithm not in ('pytorch', 'mlp') else 10,
                )

                if err:
                    error = err
                elif job and start_training:
                    TrainingService.start_job(job)
                    flash(f'调优完成！最佳CV分数: {tuning_result["best_score"]:.4f}，训练任务已启动。', 'success')
                    return redirect(url_for('training.job_detail', job_id=job.id))
                elif job:
                    flash(f'调优完成！最佳CV分数: {tuning_result["best_score"]:.4f}，训练任务已创建。', 'success')
                    return redirect(url_for('training.job_detail', job_id=job.id))

    # 获取搜索空间供前端展示
    search_spaces = {k: list(v.keys()) for k, v in SEARCH_SPACES.items()}

    return render_template(
        'training/tuning.html',
        datasets=datasets_with_cols,
        tuning_result=tuning_result,
        error=error,
        search_spaces=search_spaces,
    )
