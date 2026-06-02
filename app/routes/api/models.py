"""
============================================
AI模型 API
RESTful JSON 接口
============================================
"""
from flask import Blueprint, request, jsonify, current_app
from app.services.model_service import ModelService
from app.services.auth_service import AuthService
from app.utils.decorators import api_login_required

models_api_bp = Blueprint('models_api', __name__)


def _get_current_user():
    api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
    if api_key:
        return AuthService.get_user_by_api_key(api_key)
    from flask_login import current_user
    return current_user if current_user.is_authenticated else None


@models_api_bp.route('/', methods=['GET'])
@api_login_required
def list_models():
    """GET /api/models - 获取模型列表"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 15, type=int)
    model_type = request.args.get('model_type')
    framework = request.args.get('framework')
    status = request.args.get('status')
    search = request.args.get('search')

    user = _get_current_user()
    result = ModelService.list_models(
        page=page, per_page=per_page,
        model_type=model_type, framework=framework,
        owner_id=user.id, status=status, search=search,
    )

    return jsonify({'success': True, 'data': result})


@models_api_bp.route('/<string:model_uuid>', methods=['GET'])
@api_login_required
def get_model(model_uuid):
    """GET /api/models/<uuid> - 获取模型详情"""
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    return jsonify({
        'success': True,
        'data': model.to_dict(include_files=True),
    })


@models_api_bp.route('/', methods=['POST'])
@api_login_required
def create_model():
    """POST /api/models - 注册新模型"""
    user = _get_current_user()
    data = request.get_json(silent=True) or {}

    name = data.get('name')
    if not name:
        return jsonify({'success': False, 'message': '缺少模型名称。'}), 400

    hyperparams = data.get('hyperparameters')
    model, error = ModelService.create_model(
        user=user,
        name=name,
        model_type=data.get('model_type', 'other'),
        framework=data.get('framework'),
        description=data.get('description'),
        version=data.get('version', '1.0.0'),
        hyperparameters=hyperparams,
        is_public=data.get('is_public', False),
    )

    if error:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({
        'success': True,
        'message': '模型注册成功。',
        'data': model.to_dict(),
    }), 201


@models_api_bp.route('/<string:model_uuid>', methods=['PUT'])
@api_login_required
def update_model(model_uuid):
    """PUT /api/models/<uuid> - 更新模型"""
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = _get_current_user()
    if model.owner_id != user.id and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    data = request.get_json(silent=True) or {}
    success, error = ModelService.update_model(model, data)

    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({
        'success': True,
        'message': '模型已更新。',
        'data': model.to_dict(),
    })


@models_api_bp.route('/<string:model_uuid>/upload', methods=['POST'])
@api_login_required
def upload_model_file(model_uuid):
    """POST /api/models/<uuid>/upload - 上传模型文件"""
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = _get_current_user()
    if model.owner_id != user.id:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    file = request.files.get('model_file')
    if not file:
        return jsonify({'success': False, 'message': '请选择文件。'}), 400

    success, error = ModelService.upload_model_file(
        model, file, upload_folder=current_app.config['UPLOAD_FOLDER']
    )

    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({'success': True, 'message': '文件上传成功。'})


@models_api_bp.route('/<string:model_uuid>/metrics', methods=['PUT'])
@api_login_required
def update_metrics(model_uuid):
    """PUT /api/models/<uuid>/metrics - 更新模型指标"""
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = _get_current_user()
    if model.owner_id != user.id and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    data = request.get_json(silent=True) or {}
    success, error = ModelService.update_metrics(model, data)

    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({'success': True, 'message': '指标已更新。'})


@models_api_bp.route('/<string:model_uuid>', methods=['DELETE'])
@api_login_required
def delete_model(model_uuid):
    """DELETE /api/models/<uuid> - 删除模型"""
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = _get_current_user()
    if model.owner_id != user.id and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    success, error = ModelService.delete_model(model)
    if not success:
        return jsonify({'success': False, 'message': error}), 500

    return jsonify({'success': True, 'message': '模型已删除。'})


@models_api_bp.route('/leaderboard', methods=['GET'])
@api_login_required
def leaderboard():
    """GET /api/models/leaderboard - 模型排行榜"""
    limit = request.args.get('limit', 10, type=int)
    metric = request.args.get('metric', 'accuracy')
    top = ModelService.get_top_models(limit=limit, metric=metric)
    return jsonify({'success': True, 'data': top})
