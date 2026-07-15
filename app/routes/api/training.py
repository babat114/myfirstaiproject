"""
============================================
训练任务 API
RESTful JSON 接口
============================================
"""

import contextlib
import json

from flask import Blueprint, jsonify, request

from app import logger
from app.services.training_service import TrainingService
from app.utils.auth_helpers import get_current_user
from app.utils.decorators import api_login_required, rate_limit

training_api_bp = Blueprint('training_api', __name__)


@training_api_bp.route('/', methods=['GET'])
@api_login_required
def list_jobs():
    """获取训练任务列表
    ---
    tags:
      - Training
    summary: 获取训练列表
    parameters:
      - in: query
        name: page
        schema:
          type: integer
          default: 1
      - in: query
        name: per_page
        schema:
          type: integer
          default: 15
      - in: query
        name: status
        schema:
          type: string
        description: queued/running/paused/completed/failed/cancelled
      - in: query
        name: search
        schema:
          type: string
    """
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 15, type=int)
    status = request.args.get('status')
    search = request.args.get('search')

    user = get_current_user()
    result = TrainingService.list_jobs(
        page=page,
        per_page=per_page,
        status=status,
        search=search,
        owner_id=user.id,
    )

    return jsonify({'success': True, 'data': result})


@training_api_bp.route('/<string:job_uuid>', methods=['GET'])
@api_login_required
def get_job(job_uuid):
    """获取训练详情
    ---
    tags:
      - Training
    summary: 获取训练详情
    parameters:
      - in: path
        name: job_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: 完整训练任务信息
      404:
        description: 任务不存在
    """
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '训练任务不存在。'}), 404

    return jsonify({'success': True, 'data': job.to_dict()})


@training_api_bp.route('/', methods=['POST'])
@api_login_required
@rate_limit(max_calls=30, period=60)  # 训练创建是耗时操作，限制调用频率
def create_job():
    """创建训练任务
    ---
    tags:
      - Training
    summary: 创建训练任务
    description: 频率限制 30次/分钟。
    requestBody:
      content:
        application/json:
          schema:
            type: object
            required: [name]
            properties:
              name:
                type: string
              dataset_id:
                type: integer
              description: {type: string}
              task_type:
                type: string
              framework:
                type: string
              ml_task_type:
                type: string
              algorithm:
                type: string
              target_column:
                type: string
              test_size:
                type: number
              total_epochs:
                type: integer
              batch_size:
                type: integer
    responses:
      201:
        description: 创建成功
      400:
        description: 缺少名称
      429:
        description: 频率超限
    """
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

    return jsonify(
        {
            'success': True,
            'message': '训练任务创建成功。',
            'data': job.to_dict(),
        }
    ), 201


@training_api_bp.route('/<string:job_uuid>/start', methods=['POST'])
@api_login_required
def start_job(job_uuid):
    """启动训练
    ---
    tags:
      - Training
    summary: 启动训练
    parameters:
      - in: path
        name: job_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: 已提交到训练引擎
      403:
        description: 非所有者
    """
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
    """暂停训练
    ---
    tags:
      - Training
    summary: 暂停训练
    parameters:
      - in: path
        name: job_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: 暂停信号已发送
      403:
        description: 非所有者
    """
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
    """恢复训练
    ---
    tags:
      - Training
    summary: 恢复训练
    parameters:
      - in: path
        name: job_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: 恢复信号已发送
      403:
        description: 非所有者
    """
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
    """获取实时训练状态
    ---
    tags:
      - Training
    summary: 训练实时状态
    parameters:
      - in: path
        name: job_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: 实时状态 (epoch_progress, current_loss, memory_usage)
      404:
        description: 任务不存在
    """
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
    """更新训练进度
    ---
    tags:
      - Training
    summary: 更新训练进度
    parameters:
      - in: path
        name: job_uuid
        required: true
        schema:
          type: string
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties: {epoch: {type: integer}, step: {type: integer}, metrics: {type: object}}
    responses:
      200:
        description: 进度已更新
      403:
        description: 非所有者
    """
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
    """完成训练
    ---
    tags:
      - Training
    summary: 标记完成
    parameters:
      - in: path
        name: job_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: 训练已完成
      403:
        description: 非所有者
    """
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
    """标记训练失败
    ---
    tags:
      - Training
    summary: 标记失败
    parameters:
      - in: path
        name: job_uuid
        required: true
        schema:
          type: string
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              error:
                type: string
                description: 错误详情
    responses:
      200:
        description: 已标记失败
      400:
        description: 状态不允许
    """
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    user = get_current_user()
    if job.owner_id != user.id and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    # 仅允许对 running/paused 状态的任务标记失败, 防止滥用
    if job.status not in ('running', 'paused', 'queued', 'preparing'):
        return jsonify(
            {
                'success': False,
                'message': f'无法将 {job.status} 状态的任务标记为失败。仅运行中的任务可标记。',
            }
        ), 400

    data = request.get_json(silent=True) or {}
    error_msg = data.get('error', '未知错误')
    TrainingService.fail_job(job, error_msg)
    logger.warning(f'用户 {user.username} 手动标记任务 {job.id} 为失败: {error_msg}')

    return jsonify({'success': True, 'message': '训练失败已记录。'})


@training_api_bp.route('/<string:job_uuid>/cancel', methods=['POST'])
@api_login_required
def cancel_job(job_uuid):
    """取消训练
    ---
    tags:
      - Training
    summary: 取消训练
    parameters:
      - in: path
        name: job_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: 已取消
      403:
        description: 非所有者
    """
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
    """重新训练 (使用原参数)
    ---
    tags:
      - Training
    summary: 重新训练
    parameters:
      - in: path
        name: job_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: 已重置并提交
      403:
        description: 非所有者
    """
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
    """使用新参数重新训练
    ---
    tags:
      - Training
    summary: 换参重训
    parameters:
      - in: path
        name: job_uuid
        required: true
        schema:
          type: string
    requestBody:
      content:
        application/json:
          schema:
            type: object
            description: 完整超参数字典 (算法/框架/ML任务类型 + 所有训练参数)
    responses:
      200:
        description: 已用新参数重置并提交
      403:
        description: 非所有者
    """
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    user = get_current_user()
    if job.owner_id != user.id:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    data = request.get_json(silent=True) or {}

    # 数值转换 — expanded to accept all sklearn hyperparameter keys
    new_params = {}
    # 已知安全参数白名单
    str_fields = (
        'algorithm',
        'ml_task_type',
        'framework',
        'class_weight',
        'max_features',
        'criterion',
        'kernel',
        'penalty',
        'solver',
        'weights',
        'metric',
        'hidden_layers_str',
    )
    float_fields = (
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
    )
    int_fields = (
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
    )

    # 已知安全的额外参数 (白名单, 防止参数注入)
    ALLOWED_EXTRA_PARAMS = {
        'fit_intercept',
        'positive',
        'n_init',
        'init',
        'degree',
        'p',
        'linkage',
        'eps',
        'min_samples',
        'batch_size_tune',
        'validation_fraction',
        'early_stopping',
        'hidden_layer_sizes',
        'tol',
        'verbose',
        'n_jobs',
        'warm_start',
    }

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
    # 白名单内的额外参数 (保持原始类型)
    for k, v in data.items():
        if k in ALLOWED_EXTRA_PARAMS:
            new_params[k] = v
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

    success, error = TrainingService.retrain_job(job, new_params=new_params or None)

    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify(
        {
            'success': True,
            'message': '重新训练已启动',
            'data': {'params_used': new_params or '(使用原参数)'},
        }
    )


# ============ 参数调整引导 API ============


@training_api_bp.route('/<string:job_uuid>/guidance', methods=['GET'])
@api_login_required
def training_guidance(job_uuid):
    """获取训练参数调整建议 (AI 诊断)
    ---
    tags:
      - Training
    summary: AI训练诊断
    parameters:
      - in: path
        name: job_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: 参数优化建议 (健康度评分/问题检测/参数建议)
    """
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    from app.services.parameter_guidance_service import ParameterGuidanceService

    # 收集已有数据
    metrics_history = job.metrics_history or []
    final_metrics = {}
    if job.final_metrics_json:
        with contextlib.suppress(Exception):
            final_metrics = json.loads(job.final_metrics_json)

    hyperparams = {}
    if job.model and job.model.hyperparameters_json:
        with contextlib.suppress(Exception):
            hyperparams = json.loads(job.model.hyperparameters_json)

    guidance = ParameterGuidanceService.analyze_results(
        metrics_history=metrics_history,
        final_metrics=final_metrics,
        hyperparams=hyperparams,
    )

    return jsonify({'success': True, 'data': guidance})


@training_api_bp.route('/<string:job_uuid>', methods=['DELETE'])
@api_login_required
def delete_job(job_uuid):
    """删除训练任务
    ---
    tags:
      - Training
    summary: 删除训练
    parameters:
      - in: path
        name: job_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: 已删除
      403:
        description: 非所有者或管理员
    """
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
    """获取算法搜索空间
    ---
    tags:
      - Training
    summary: 调优搜索空间
    parameters:
      - in: query
        name: algorithm
        required: true
        schema:
          type: string
      - in: query
        name: framework
        schema:
          type: string
    responses:
      200:
        description: 搜索空间 (param_grid 参数网格)
    """
    from app.services.hyperparameter_tuning import HyperparameterTuningService

    algorithm = request.args.get('algorithm', 'random_forest')
    framework = request.args.get('framework', 'sklearn')
    space = HyperparameterTuningService.get_search_space(algorithm, framework)
    return jsonify({'success': True, 'data': space})


@training_api_bp.route('/tuning/run', methods=['POST'])
@api_login_required
def run_tuning():
    """运行超参数调优
    ---
    tags:
      - Training
    summary: 运行调优
    description: GridSearchCV / RandomSearchCV / AutoML 三种模式。聚类特殊处理 (全量fit+子采样score)。SSE进度通过 /stream/tuning/<id>/stream 推送。
    requestBody:
      content:
        application/json:
          schema:
            type: object
            required: [dataset_id]
            properties:
              dataset_id:
                type: integer
              algorithm:
                type: string
              ml_task_type:
                type: string
              target_column:
                type: string
              tuning_method:
                type: string
                enum:
                  - grid
                  - random
                  - auto
                default: random
              n_iter:
                type: integer
              cv:
                type: integer
                default: 5
              start_training:
                type: boolean
    responses:
      201:
        description: 调优完成 (含 best_params + tuning_result)
      400:
        description: 参数无效
    """
    from app.services.dataset_service import DatasetService
    from app.services.hyperparameter_tuning import HyperparameterTuningService

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

    return jsonify(
        {
            'success': True,
            'message': '调优完成',
            'data': {
                'job': job.to_dict() if job else None,
                'tuning_result': tuning_result,
            },
        }
    ), 201
