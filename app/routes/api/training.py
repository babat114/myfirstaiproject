"""
============================================
训练任务 API
RESTful JSON 接口
============================================
"""
import json
from flask import Blueprint, request, jsonify
from app.services.training_service import TrainingService
from app.services.auth_service import AuthService
from app.utils.decorators import api_login_required
from app.utils.auth_helpers import get_current_user

training_api_bp = Blueprint('training_api', __name__)


@training_api_bp.route('/', methods=['GET'])
@api_login_required
def list_jobs():
    """GET /api/training - 获取训练任务列表"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 15, type=int)
    status = request.args.get('status')
    search = request.args.get('search')

    user = get_current_user()
    result = TrainingService.list_jobs(
        page=page, per_page=per_page,
        status=status, search=search,
        owner_id=user.id,
    )

    return jsonify({'success': True, 'data': result})


@training_api_bp.route('/<string:job_uuid>', methods=['GET'])
@api_login_required
def get_job(job_uuid):
    """GET /api/training/<uuid> - 获取训练详情"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '训练任务不存在。'}), 404

    return jsonify({'success': True, 'data': job.to_dict()})


@training_api_bp.route('/', methods=['POST'])
@api_login_required
def create_job():
    """POST /api/training - 创建训练任务"""
    user = get_current_user()
    data = request.get_json(silent=True) or {}

    name = data.get('name')
    if not name:
        return jsonify({'success': False, 'message': '缺少任务名称。'}), 400

    job, error = TrainingService.create_job(
        user=user,
        name=name,
        dataset_id=data.get('dataset_id'),
        description=data.get('description'),
        task_type=data.get('task_type', 'training'),
        framework=data.get('framework', 'sklearn'),
        total_epochs=data.get('total_epochs', 10),
        total_steps=data.get('total_steps', 0),
        gpu_count=data.get('gpu_count', 0),
        cpu_cores=data.get('cpu_cores', 1),
        memory_gb=data.get('memory_gb', 4.0),
        ml_task_type=data.get('ml_task_type', 'classification'),
        algorithm=data.get('algorithm', 'random_forest'),
        target_column=data.get('target_column'),
        test_size=data.get('test_size', 0.2),
    )

    if error:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({
        'success': True,
        'message': '训练任务创建成功。',
        'data': job.to_dict(),
    }), 201


@training_api_bp.route('/<string:job_uuid>/start', methods=['POST'])
@api_login_required
def start_job(job_uuid):
    """POST /api/training/<uuid>/start - 启动训练"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    user = get_current_user()
    if job.owner_id != user.id:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    success, error = TrainingService.start_job(job)
    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({'success': True, 'message': '训练已启动。'})


@training_api_bp.route('/<string:job_uuid>/pause', methods=['POST'])
@api_login_required
def pause_job(job_uuid):
    """POST /api/training/<uuid>/pause - 暂停训练"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    user = get_current_user()
    if job.owner_id != user.id:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    success, error = TrainingService.pause_job(job)
    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({'success': True, 'message': '训练已暂停。'})


@training_api_bp.route('/<string:job_uuid>/resume', methods=['POST'])
@api_login_required
def resume_job(job_uuid):
    """POST /api/training/<uuid>/resume - 恢复训练"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    user = get_current_user()
    if job.owner_id != user.id:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    success, error = TrainingService.resume_job(job)
    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({'success': True, 'message': '训练已恢复。'})


@training_api_bp.route('/<string:job_uuid>/status', methods=['GET'])
@api_login_required
def job_status(job_uuid):
    """GET /api/training/<uuid>/status - 获取实时状态"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    status = TrainingService.get_job_status(job.id)
    if not status:
        return jsonify({'success': True, 'data': job.to_dict()})

    return jsonify({'success': True, 'data': status})


@training_api_bp.route('/<string:job_uuid>/progress', methods=['PUT'])
@api_login_required
def update_progress(job_uuid):
    """PUT /api/training/<uuid>/progress - 更新训练进度"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    user = get_current_user()
    if job.owner_id != user.id:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    data = request.get_json(silent=True) or {}
    epoch = data.get('epoch', 0)
    step = data.get('step', 0)
    metrics = data.get('metrics')

    success, error = TrainingService.update_progress(job, epoch, step, metrics)
    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({'success': True, 'message': '进度已更新。'})


@training_api_bp.route('/<string:job_uuid>/complete', methods=['POST'])
@api_login_required
def complete_job(job_uuid):
    """POST /api/training/<uuid>/complete - 完成训练"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    user = get_current_user()
    if job.owner_id != user.id:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    success, error = TrainingService.complete_job(job)
    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({'success': True, 'message': '训练已完成。'})


@training_api_bp.route('/<string:job_uuid>/fail', methods=['POST'])
@api_login_required
def fail_job(job_uuid):
    """POST /api/training/<uuid>/fail - 标记训练失败"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    user = get_current_user()
    if job.owner_id != user.id:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    data = request.get_json(silent=True) or {}
    error_msg = data.get('error', '未知错误')
    TrainingService.fail_job(job, error_msg)

    return jsonify({'success': True, 'message': '训练失败已记录。'})


@training_api_bp.route('/<string:job_uuid>/cancel', methods=['POST'])
@api_login_required
def cancel_job(job_uuid):
    """POST /api/training/<uuid>/cancel - 取消训练"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    user = get_current_user()
    if job.owner_id != user.id:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    success, error = TrainingService.cancel_job(job)
    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({'success': True, 'message': '训练已取消。'})


@training_api_bp.route('/<string:job_uuid>/retrain', methods=['POST'])
@api_login_required
def retrain_job(job_uuid):
    """POST /api/training/<uuid>/retrain - 重新训练 (重置并启动)"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    user = get_current_user()
    if job.owner_id != user.id:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    success, error = TrainingService.retrain_job(job)
    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({'success': True, 'message': '训练任务已重置并启动。'})


@training_api_bp.route('/<string:job_uuid>/retrain-with-params', methods=['POST'])
@api_login_required
def retrain_with_params(job_uuid):
    """POST /api/training/<uuid>/retrain-with-params - 使用新参数重新训练"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    user = get_current_user()
    if job.owner_id != user.id:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    data = request.get_json(silent=True) or {}

    # 数值转换
    new_params = {}
    str_fields = ('algorithm', 'ml_task_type', 'framework')
    float_fields = ('learning_rate', 'test_size', 'dropout', 'weight_decay')
    int_fields = ('batch_size', 'epochs', 'total_epochs')

    for k in str_fields:
        if data.get(k):
            new_params[k] = data[k]
    for k in float_fields:
        if data.get(k) is not None:
            try:
                new_params[k] = float(data[k])
            except (ValueError, TypeError):
                return jsonify({'success': False, 'message': f'参数 {k} 格式无效'}), 400
    for k in int_fields:
        if data.get(k) is not None:
            try:
                new_params[k] = int(data[k])
            except (ValueError, TypeError):
                return jsonify({'success': False, 'message': f'参数 {k} 格式无效'}), 400
    if data.get('hidden_layers'):
        try:
            if isinstance(data['hidden_layers'], str):
                new_params['hidden_layers'] = [int(x.strip()) for x in data['hidden_layers'].split(',') if x.strip()]
            elif isinstance(data['hidden_layers'], list):
                new_params['hidden_layers'] = [int(x) for x in data['hidden_layers']]
        except (ValueError, TypeError):
            return jsonify({'success': False, 'message': 'hidden_layers 格式无效'}), 400
    if 'epochs' in new_params and 'total_epochs' not in new_params:
        new_params['total_epochs'] = new_params.pop('epochs')

    if not new_params:
        success, error = TrainingService.retrain_job(job)
    else:
        success, error = TrainingService.retrain_job_with_params(job, new_params)

    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({
        'success': True,
        'message': '重新训练已启动',
        'data': {'params_used': new_params or '(使用原参数)'},
    })


# ============ 参数调整引导 API ============

@training_api_bp.route('/<string:job_uuid>/guidance', methods=['GET'])
@api_login_required
def training_guidance(job_uuid):
    """GET /api/training/<uuid>/guidance - 获取训练参数调整建议"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    from app.services.parameter_guidance_service import ParameterGuidanceService

    # 收集已有数据
    metrics_history = job.metrics_history or []
    final_metrics = {}
    if job.final_metrics_json:
        try:
            final_metrics = json.loads(job.final_metrics_json)
        except Exception:
            pass

    hyperparams = {}
    if job.model and job.model.hyperparameters_json:
        try:
            hyperparams = json.loads(job.model.hyperparameters_json)
        except Exception:
            pass

    guidance = ParameterGuidanceService.analyze_results(
        metrics_history=metrics_history,
        final_metrics=final_metrics,
        hyperparams=hyperparams,
    )

    return jsonify({'success': True, 'data': guidance})


@training_api_bp.route('/<string:job_uuid>', methods=['DELETE'])
@api_login_required
def delete_job(job_uuid):
    """DELETE /api/training/<uuid> - 删除训练任务"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    user = get_current_user()
    if job.owner_id != user.id and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    success, error = TrainingService.delete_job(job)
    if not success:
        return jsonify({'success': False, 'message': error}), 500

    return jsonify({'success': True, 'message': '任务已删除。'})


@training_api_bp.route('/tuning/search-space', methods=['GET'])
@api_login_required
def get_search_space():
    """GET /api/training/tuning/search-space - 获取算法的搜索空间"""
    from app.services.hyperparameter_tuning import HyperparameterTuningService
    algorithm = request.args.get('algorithm', 'random_forest')
    framework = request.args.get('framework', 'sklearn')
    space = HyperparameterTuningService.get_search_space(algorithm, framework)
    return jsonify({'success': True, 'data': space})


@training_api_bp.route('/tuning/run', methods=['POST'])
@api_login_required
def run_tuning():
    """POST /api/training/tuning/run - 运行超参数调优"""
    from app.services.hyperparameter_tuning import HyperparameterTuningService
    from app.services.dataset_service import DatasetService

    user = get_current_user()
    data = request.get_json(silent=True) or {}

    dataset_id = data.get('dataset_id')
    algorithm = data.get('algorithm', 'random_forest')
    task_type = data.get('ml_task_type', 'classification')
    target_column = data.get('target_column')
    tuning_method = data.get('tuning_method', 'random')
    n_iter = data.get('n_iter', 30)
    cv = data.get('cv', 5)
    start_training = data.get('start_training', False)

    dataset = DatasetService.get_dataset_by_id(dataset_id)
    if not dataset:
        return jsonify({'success': False, 'message': '数据集不存在。'}), 404

    job, tuning_result, error = HyperparameterTuningService.create_tuned_training(
        user=user,
        dataset=dataset,
        algorithm=algorithm,
        task_type=task_type,
        target_column=target_column,
        tuning_method=tuning_method,
        n_iter=n_iter,
        cv=cv,
    )

    if error:
        return jsonify({'success': False, 'message': error, 'tuning_result': tuning_result}), 400

    if start_training and job:
        TrainingService.start_job(job)

    return jsonify({
        'success': True,
        'message': '调优完成',
        'data': {
            'job': job.to_dict() if job else None,
            'tuning_result': tuning_result,
        }
    }), 201
