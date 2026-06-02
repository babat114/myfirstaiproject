"""
============================================
训练任务 API
RESTful JSON 接口
============================================
"""
from flask import Blueprint, request, jsonify
from app.services.training_service import TrainingService
from app.services.auth_service import AuthService
from app.utils.decorators import api_login_required

training_api_bp = Blueprint('training_api', __name__)


def _get_current_user():
    api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
    if api_key:
        return AuthService.get_user_by_api_key(api_key)
    from flask_login import current_user
    return current_user if current_user.is_authenticated else None


@training_api_bp.route('/', methods=['GET'])
@api_login_required
def list_jobs():
    """GET /api/training - 获取训练任务列表"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 15, type=int)
    status = request.args.get('status')
    search = request.args.get('search')

    user = _get_current_user()
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
    user = _get_current_user()
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
        framework=data.get('framework'),
        total_epochs=data.get('total_epochs', 0),
        total_steps=data.get('total_steps', 0),
        gpu_count=data.get('gpu_count', 0),
        cpu_cores=data.get('cpu_cores', 1),
        memory_gb=data.get('memory_gb', 4.0),
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

    user = _get_current_user()
    if job.owner_id != user.id:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    success, error = TrainingService.start_job(job)
    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({'success': True, 'message': '训练已启动。'})


@training_api_bp.route('/<string:job_uuid>/progress', methods=['PUT'])
@api_login_required
def update_progress(job_uuid):
    """PUT /api/training/<uuid>/progress - 更新训练进度"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    user = _get_current_user()
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

    user = _get_current_user()
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

    user = _get_current_user()
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

    user = _get_current_user()
    if job.owner_id != user.id:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    success, error = TrainingService.cancel_job(job)
    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({'success': True, 'message': '训练已取消。'})


@training_api_bp.route('/<string:job_uuid>', methods=['DELETE'])
@api_login_required
def delete_job(job_uuid):
    """DELETE /api/training/<uuid> - 删除训练任务"""
    job = TrainingService.get_job_by_uuid(job_uuid)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在。'}), 404

    user = _get_current_user()
    if job.owner_id != user.id and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    success, error = TrainingService.delete_job(job)
    if not success:
        return jsonify({'success': False, 'message': error}), 500

    return jsonify({'success': True, 'message': '任务已删除。'})
