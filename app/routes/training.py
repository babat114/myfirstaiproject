"""
============================================
训练任务 Web 路由
训练任务管理的页面路由
============================================
"""

import contextlib
import json
import logging

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required

from app.services.dataset_service import DatasetService
from app.services.training_service import TrainingService
from app.utils.decorators import rate_limit
from app.utils.helpers import parse_form_params

training_bp = Blueprint('training', __name__)
logger = logging.getLogger(__name__)


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
    # 获取用户的数据集列表 + 公开数据集
    datasets_result = DatasetService.list_datasets(owner_id=current_user.id, per_page=100, include_public=True)
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

        # 防过拟合超参数 (PyTorch/TensorFlow，可选) — 委托给 Service 层统一解析
        extra_hparams = TrainingService.parse_extra_hyperparams(request.form.to_dict())

        if not name:
            flash('请输入任务名称。', 'danger')
            return render_template('training/create.html', datasets=datasets_with_cols)

        if not dataset_id:
            flash('请选择数据集。', 'danger')
            return render_template('training/create.html', datasets=datasets_with_cols)

        # === 超参数自动调优 ===
        # AutoML 模式: algorithm='auto' 时自动启用调优
        auto_tune = (
            request.form.get('auto_tune') == 'on'
            or request.form.get('auto_tune_hidden') == 'true'
            or algorithm == 'auto'
        )
        if auto_tune:
            # 统一构建 presets: 包含创建页面的全部字段, 避免跳转后丢失
            def _build_presets(algo):
                return {
                    'dataset_id': dataset_id,
                    'task_type': ml_task_type,
                    'target_column': target_column,
                    'tuning_method': request.form.get('tuning_method', 'random'),
                    'n_iter': request.form.get('n_iter', 30, type=int) if request.form.get('n_iter') else 30,
                    'cv': request.form.get('tuning_cv', 5, type=int) if request.form.get('tuning_cv') else 5,
                    'algorithm': algo,
                    'start_training': request.form.get('start_immediately') == 'on',
                    'job_name': name,
                    'description': description,
                    'framework': framework,
                    'test_size': test_size,
                    'total_epochs': total_epochs,
                    'total_steps': total_steps,
                    'gpu_count': gpu_count,
                    'cpu_cores': cpu_cores,
                    'memory_gb': memory_gb,
                    'extra_hparams_json': json.dumps(extra_hparams) if extra_hparams else '{}',
                }

            if algorithm == 'auto':
                session['_tuning_presets'] = _build_presets('auto')
                flash('已切换到 AutoML 智能调优模式 — 正在遍历所有适用算法, 请耐心等待实时进度...', 'info')
                return redirect(url_for('training.hyperparameter_tuning', auto_start='1'))

            session['_tuning_presets'] = _build_presets(algorithm)
            flash(f'已切换到 {algorithm} 自动调优模式 — 正在使用异步实时进度搜索最佳参数...', 'info')
            return redirect(url_for('training.hyperparameter_tuning', auto_start='1'))

        # === 正常创建流程 (无自动调优) ===
        if algorithm in (
            'random_forest',
            'gradient_boosting',
            'random_forest_regressor',
            'gradient_boosting_regressor',
            'svm',
            'svr',
            'logistic_regression',
            'linear_regression',
            'ridge',
            'knn',
            'knn_regressor',
            'decision_tree',
            'kmeans',
            'dbscan',
            'agglomerative',
            'minibatch_kmeans',
        ):
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
            hyperparameters=extra_hparams if extra_hparams else None,
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
@rate_limit(max_calls=60, period=60)  # 防 ID 枚举: 限制详情页访问频率
def job_detail(job_id):
    """训练任务详情"""
    job = TrainingService.get_job_by_id(job_id)
    if not job:
        flash('训练任务不存在。', 'danger')
        return redirect(url_for('training.list_jobs'))

    if not job.is_viewable_by(current_user):
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

    if not job.is_viewable_by(current_user):
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

    if not job.is_viewable_by(current_user):
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

    if not job.is_viewable_by(current_user):
        flash('您没有权限操作此任务。', 'danger')
        return redirect(url_for('training.list_jobs'))

    # 白名单字段类型 — 复用 parse_form_params 减少重复代码
    _STR_FIELDS = {
        'algorithm',
        'ml_task_type',
        'framework',
        'hidden_layers_str',
        'class_weight',
        'max_features',
        'criterion',
        'kernel',
        'penalty',
        'solver',
        'weights',
        'metric',
    }
    _FLOAT_FIELDS = {
        'learning_rate',
        'test_size',
        'dropout',
        'weight_decay',
        'C',
        'alpha',
        'gamma',
        'epsilon',
        'subsample',
        'learning_rate_init',
    }
    _INT_FIELDS = {
        'batch_size',
        'epochs',
        'total_epochs',
        'n_estimators',
        'max_depth',
        'min_samples_split',
        'min_samples_leaf',
        'n_neighbors',
        'max_iter',
        'n_clusters',
        'val_size',
        'early_stopping_patience',
    }

    new_params = parse_form_params(
        dict(request.form),
        int_fields=_INT_FIELDS,
        float_fields=_FLOAT_FIELDS,
        str_fields=_STR_FIELDS,
    )

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
        success, error = TrainingService.retrain_job(job, new_params=new_params)
        msg = '已使用新参数重新训练！'

    if success:
        flash(msg, 'success')
    else:
        flash(error, 'danger')

    return redirect(url_for('training.job_detail', job_id=job.id))


@training_bp.route('/<int:job_id>/gridsearch-retrain', methods=['POST'])
@login_required
def gridsearch_retrain(job_id):
    """使用 GridSearchCV 自动搜索最优参数并重新训练

    AJAX 请求: 启动后台调优, 返回 {tuning_id, redirect_url}
    传统表单: 同步执行, redirect到详情页
    """
    from flask import jsonify, request

    from app.services.hyperparameter_tuning import HyperparameterTuningService

    job = TrainingService.get_job_by_id(job_id)
    if not job:
        flash('任务不存在。', 'danger')
        return redirect(url_for('training.list_jobs'))

    if not job.is_viewable_by(current_user):
        flash('您没有权限操作此任务。', 'danger')
        return redirect(url_for('training.list_jobs'))

    if not job.dataset:
        flash('无法获取关联的数据集。', 'danger')
        return redirect(url_for('training.job_detail', job_id=job.id))

    hparams = job.model.hyperparameters_dict if job.model else {}
    algorithm = hparams.get('algorithm', 'random_forest')
    model_type = job.model.model_type if job.model else None
    if model_type in ('classification', 'regression', 'clustering'):
        ml_task_type = model_type
    else:
        ml_task_type = hparams.get('task_type') or hparams.get('ml_task_type') or 'classification'
    target_column = hparams.get('target_column')
    is_mlp = algorithm == 'mlp'
    is_ajax = (
        request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        or request.content_type == 'application/json'
        or request.args.get('async') == '1'
    )

    # ---- 异步模式 (AJAX): 后台线程 + SSE 进度 ----
    if is_ajax:
        tuning_id = HyperparameterTuningService.run_grid_search_async(
            dataset=job.dataset,
            algorithm=algorithm,
            task_type=ml_task_type,
            target_column=target_column,
            cv=3,
            n_jobs=2,
            random_state=None,
        )
        return jsonify(
            {
                'success': True,
                'tuning_id': tuning_id,
                'algorithm': algorithm,
                'task_type': ml_task_type,
                'is_mlp': is_mlp,
                'job_id': job.id,
                'stream_url': f'/api/v1/stream/tuning/{tuning_id}/stream',
                'complete_redirect': url_for('training.job_detail', job_id=job.id),
            }
        )

    # ---- 同步模式 (传统表单提交, 兼容旧行为) ----
    if is_mlp:
        flash('正在对 PyTorch MLP 运行超参数搜索 (sklearn代理)...', 'info')
    else:
        flash(f'正在对 {algorithm} 运行 GridSearchCV 自动调优...', 'info')

    tuning_result = None
    search_mode = 'GridSearch'
    try:
        tuning_result = HyperparameterTuningService.run_grid_search(
            dataset=job.dataset,
            algorithm=algorithm,
            task_type=ml_task_type,
            target_column=target_column,
            cv=3,
            n_jobs=2,
        )
    except Exception as e:
        logger.warning(f'GridSearch failed, falling back to RandomSearch: {e}')
        flash(f'GridSearchCV 搜索失败 ({str(e)})，已自动降级为 RandomSearch 快速搜索', 'warning')

    if not tuning_result or not tuning_result.get('success'):
        search_mode = 'RandomSearch'
        flash('正在使用 RandomSearch 进行快速参数搜索...', 'info')
        try:
            tuning_result = HyperparameterTuningService.run_random_search(
                dataset=job.dataset,
                algorithm=algorithm,
                task_type=ml_task_type,
                target_column=target_column,
                n_iter=20,
                cv=3,
                n_jobs=2,
            )
        except Exception as e:
            flash(f'超参数调优失败: {str(e)}', 'danger')
            return redirect(url_for('training.job_detail', job_id=job.id))

    if not tuning_result or not tuning_result.get('success'):
        flash(f'超参数调优失败: {tuning_result.get("error", "未知错误")}', 'danger')
        return redirect(url_for('training.job_detail', job_id=job.id))

    best_params = tuning_result['best_params']
    best_score = tuning_result['best_score']

    is_mlp = algorithm == 'mlp'
    retrain_params = TrainingService.build_retrain_params(best_params, hparams, algorithm, ml_task_type, is_mlp=is_mlp)

    success, error = TrainingService.retrain_job(job, new_params=retrain_params)

    if success:
        TrainingService.apply_tuning_hyperparameters(
            job,
            best_params,
            best_score,
            search_time=tuning_result.get('search_time'),
        )
        flash(
            f'{search_mode} 优化完成! 最佳CV分数: {best_score:.4f}, 最佳参数: {best_params}. 已使用最优参数重新训练!',
            'success',
        )
    else:
        flash(f'{search_mode} 调优完成({best_score:.4f}), 但重训失败: {error}', 'warning')

    return redirect(url_for('training.job_detail', job_id=job.id))


@training_bp.route('/<int:job_id>/apply-tuning', methods=['POST'])
@login_required
def apply_tuning_result(job_id):
    """异步调优完成后: 应用最佳参数并重新训练

    接收 JSON: {tuning_id, ...}
    从 TuningProgressTracker 读取结果, 应用到训练任务.
    """
    from flask import jsonify, request

    from app.services.hyperparameter_tuning import get_tuning_tracker

    job = TrainingService.get_job_by_id(job_id)
    if not job or not job.is_viewable_by(current_user):
        return jsonify({'success': False, 'error': '任务不存在或无权访问'}), 403

    tracker = get_tuning_tracker()
    tuning_id = request.json.get('tuning_id') if request.is_json else request.form.get('tuning_id')

    session = tracker.get(tuning_id) if tuning_id else None
    if not session or session['status'] != 'completed':
        return jsonify({'success': False, 'error': '调优尚未完成或已过期'}), 400

    result = session.get('result', {})
    best_params = result.get('best_params', session.get('best_params_so_far', {}))
    best_score = result.get('best_score', session.get('best_score_so_far', 0))

    if not best_params:
        return jsonify({'success': False, 'error': '无最佳参数可用'}), 400

    hparams = job.model.hyperparameters_dict if job.model else {}
    algorithm = hparams.get('algorithm', 'random_forest')
    ml_task_type = session.get('task_type', 'classification')
    is_mlp = algorithm == 'mlp'

    retrain_params = TrainingService.build_retrain_params(best_params, hparams, algorithm, ml_task_type, is_mlp=is_mlp)

    success, error = TrainingService.retrain_job(job, new_params=retrain_params)

    if success:
        TrainingService.apply_tuning_hyperparameters(
            job,
            best_params,
            best_score,
            search_time=session.get('elapsed_seconds'),
        )

    if success:
        return jsonify(
            {
                'success': True,
                'best_score': best_score,
                'best_params': best_params,
                'redirect_url': url_for('training.job_detail', job_id=job.id),
            }
        )
    else:
        return jsonify({'success': False, 'error': error}), 500


@training_bp.route('/tuning', methods=['GET', 'POST'])
@login_required
def hyperparameter_tuning():
    """超参数自动调优页面"""
    from flask import jsonify

    from app.services.hyperparameter_tuning import SEARCH_SPACES, HyperparameterTuningService

    datasets_result = DatasetService.list_datasets(owner_id=current_user.id, per_page=100, include_public=True)
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

    # ── 随机种子: 支持可复现调优 (form参数 random_seed) ──
    random_state = None
    try:
        random_state = request.form.get('random_seed', type=int)
        if random_state is not None and random_state < 0:
            random_state = None
    except (ValueError, TypeError):
        random_state = None

    # ── AutoML 自动启动: 从创建页面重定向过来, 预填参数并自动提交 ──
    auto_start = request.args.get('auto_start') == '1'
    presets = session.pop('_tuning_presets', None) if auto_start else None
    # 非AJAX回退产生的 tuning_id: 页面加载后自动连接 SSE 进度
    live_tuning_id = request.args.get('tuning_id')

    if request.method == 'POST':
        dataset_id = request.form.get('dataset_id', type=int)
        algorithm = request.form.get('algorithm', 'random_forest')
        task_type = request.form.get('ml_task_type', 'classification')
        target_column = request.form.get('target_column', '').strip() or None
        tuning_method = request.form.get('tuning_method', 'random')
        n_iter = request.form.get('n_iter', 30, type=int)
        cv = request.form.get('cv', 5, type=int)
        action = request.form.get('action', '')

        # ── 调优完成 → 创建训练任务 ──
        if action == 'create_job':
            from app.services.training_service import TrainingService

            tuning_id = request.form.get('tuning_id', '')
            tracker = HyperparameterTuningService.get_tuning_tracker()
            tsession = tracker.get(tuning_id) if tuning_id else None
            if not tsession or tsession['status'] != 'completed':
                error = '调优尚未完成或不存在。'
            else:
                best_params = tsession.get('result', {}).get('best_params', tsession.get('best_params_so_far', {}))
                best_score = tsession.get('result', {}).get('best_score', tsession.get('best_score_so_far'))
                job_name = request.form.get('job_name', '').strip() or f'{algorithm} (调优最佳)'
                description = request.form.get('description', '').strip() or f'调优自动创建, best_score={best_score}'
                framework = request.form.get('framework', 'sklearn').strip() or 'sklearn'
                test_size = request.form.get('test_size', 0.2, type=float)
                total_epochs = request.form.get('total_epochs', 10, type=int)
                total_steps = request.form.get('total_steps', 0, type=int)
                gpu_count = request.form.get('gpu_count', 0, type=int)
                cpu_cores = request.form.get('cpu_cores', 1, type=int)
                memory_gb = request.form.get('memory_gb', 4.0, type=float)
                extra_hparams = {}
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    extra_hparams = json.loads(request.form.get('extra_hparams_json', '{}'))

                if best_params and isinstance(extra_hparams, dict):
                    extra_hparams.update(best_params)

                job, job_error = TrainingService.create_job(
                    user=current_user,
                    name=job_name,
                    dataset_id=dataset_id,
                    description=description,
                    task_type='training',
                    framework=framework,
                    total_epochs=max(total_epochs, 1),
                    total_steps=total_steps,
                    gpu_count=gpu_count,
                    cpu_cores=cpu_cores,
                    memory_gb=memory_gb,
                    ml_task_type=task_type,
                    algorithm=algorithm,
                    target_column=target_column,
                    test_size=test_size,
                    hyperparameters=extra_hparams if extra_hparams else None,
                )
                if job_error:
                    error = job_error
                else:
                    success, start_error = TrainingService.start_job(job)
                    flash(f'调优完成！已自动创建训练任务 "{job.name}" (best_score={best_score})', 'success')
                    return redirect(url_for('training.job_detail', job_id=job.id))

        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.args.get('async') == '1'

        if not dataset_id:
            error = '请选择数据集。'
        else:
            dataset = DatasetService.get_dataset_by_id(dataset_id)
            if not dataset:
                error = '数据集不存在。'
            else:
                if not target_column:
                    target_column = None

                # ---- 异步模式: 后台调优 + SSE 进度 (推荐) ----
                if is_ajax:
                    if algorithm == 'auto':
                        tuning_id = HyperparameterTuningService.run_auto_tuning_async(
                            dataset=dataset,
                            task_type=task_type,
                            target_column=target_column,
                            cv=cv,
                            n_jobs=2,
                            random_state=random_state,
                        )
                    elif tuning_method == 'grid':
                        tuning_id = HyperparameterTuningService.run_grid_search_async(
                            dataset=dataset,
                            algorithm=algorithm,
                            task_type=task_type,
                            target_column=target_column,
                            cv=cv,
                            n_jobs=2,
                            random_state=random_state,
                        )
                    else:
                        tuning_id = HyperparameterTuningService.run_random_search_async(
                            dataset=dataset,
                            algorithm=algorithm,
                            task_type=task_type,
                            target_column=target_column,
                            n_iter=n_iter,
                            cv=cv,
                            n_jobs=2,
                            random_state=random_state,
                        )
                    return jsonify(
                        {
                            'success': True,
                            'tuning_id': tuning_id,
                            'stream_url': f'/api/v1/stream/tuning/{tuning_id}/stream',
                        }
                    )

                # ---- 非 AJAX 回退: 也走异步, 重定向到当前页并连接 SSE ----
                if algorithm == 'auto':
                    tuning_id = HyperparameterTuningService.run_auto_tuning_async(
                        dataset=dataset,
                        task_type=task_type,
                        target_column=target_column,
                        cv=cv,
                        n_jobs=2,
                        random_state=random_state,
                    )
                elif tuning_method == 'grid':
                    tuning_id = HyperparameterTuningService.run_grid_search_async(
                        dataset=dataset,
                        algorithm=algorithm,
                        task_type=task_type,
                        target_column=target_column,
                        cv=cv,
                        n_jobs=2,
                        random_state=random_state,
                    )
                else:
                    tuning_id = HyperparameterTuningService.run_random_search_async(
                        dataset=dataset,
                        algorithm=algorithm,
                        task_type=task_type,
                        target_column=target_column,
                        n_iter=n_iter,
                        cv=cv,
                        n_jobs=2,
                        random_state=random_state,
                    )
                flash('调优已在后台启动，下方将展示实时进度。', 'info')
                return redirect(url_for('training.hyperparameter_tuning', tuning_id=tuning_id))

    # 获取搜索空间供前端展示
    search_spaces = {k: list(v.keys()) for k, v in SEARCH_SPACES.items()}

    return render_template(
        'training/tuning.html',
        datasets=datasets_with_cols,
        tuning_result=tuning_result,
        error=error,
        search_spaces=search_spaces,
        auto_start=auto_start,
        presets=presets,
        live_tuning_id=live_tuning_id,
    )
