"""
============================================
数据集 API
RESTful JSON 接口
============================================
"""
from flask import Blueprint, request, jsonify
from app.services.dataset_service import DatasetService
from app.services.auth_service import AuthService
from app.utils.decorators import api_login_required, api_admin_required

datasets_api_bp = Blueprint('datasets_api', __name__)


def _get_current_user():
    """从 API 请求中获取当前用户"""
    api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
    if api_key:
        return AuthService.get_user_by_api_key(api_key)
    from flask_login import current_user
    return current_user if current_user.is_authenticated else None


@datasets_api_bp.route('/', methods=['GET'])
@api_login_required
def list_datasets():
    """GET /api/datasets - 获取数据集列表"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 15, type=int)
    category = request.args.get('category')
    search = request.args.get('search')

    user = _get_current_user()
    result = DatasetService.list_datasets(
        page=page, per_page=per_page,
        category=category, search=search,
        owner_id=user.id,
    )

    return jsonify({
        'success': True,
        'data': result,
    })


@datasets_api_bp.route('/<string:dataset_uuid>', methods=['GET'])
@api_login_required
def get_dataset(dataset_uuid):
    """GET /api/datasets/<uuid> - 获取数据集详情"""
    dataset = DatasetService.get_dataset_by_uuid(dataset_uuid)
    if not dataset:
        return jsonify({'success': False, 'message': '数据集不存在。'}), 404

    return jsonify({
        'success': True,
        'data': dataset.to_dict(include_file_path=True),
    })


@datasets_api_bp.route('/', methods=['POST'])
@api_login_required
def create_dataset():
    """POST /api/datasets - 创建新数据集"""
    user = _get_current_user()
    if not user.can_upload:
        return jsonify({'success': False, 'message': '没有上传权限。'}), 403

    name = request.form.get('name')
    file = request.files.get('file')

    if not name or not file:
        return jsonify({'success': False, 'message': '缺少必要字段。'}), 400

    from flask import current_app
    dataset, error = DatasetService.create_dataset(
        user=user,
        name=name,
        file=file,
        description=request.form.get('description'),
        category=request.form.get('category', 'other'),
        is_public=request.form.get('is_public', 'false').lower() == 'true',
        upload_folder=current_app.config['UPLOAD_FOLDER'],
    )

    if error:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({
        'success': True,
        'message': '数据集创建成功。',
        'data': dataset.to_dict(),
    }), 201


@datasets_api_bp.route('/<string:dataset_uuid>', methods=['PUT'])
@api_login_required
def update_dataset(dataset_uuid):
    """PUT /api/datasets/<uuid> - 更新数据集"""
    dataset = DatasetService.get_dataset_by_uuid(dataset_uuid)
    if not dataset:
        return jsonify({'success': False, 'message': '数据集不存在。'}), 404

    user = _get_current_user()
    if dataset.owner_id != user.id and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    data = request.get_json(silent=True) or {}
    success, error = DatasetService.update_dataset(dataset, data)

    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({
        'success': True,
        'message': '数据集已更新。',
        'data': dataset.to_dict(),
    })


@datasets_api_bp.route('/<string:dataset_uuid>', methods=['DELETE'])
@api_login_required
def delete_dataset(dataset_uuid):
    """DELETE /api/datasets/<uuid> - 删除数据集"""
    dataset = DatasetService.get_dataset_by_uuid(dataset_uuid)
    if not dataset:
        return jsonify({'success': False, 'message': '数据集不存在。'}), 404

    user = _get_current_user()
    if dataset.owner_id != user.id and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    success, error = DatasetService.delete_dataset(dataset)
    if not success:
        return jsonify({'success': False, 'message': error}), 500

    return jsonify({'success': True, 'message': '数据集已删除。'})


@datasets_api_bp.route('/stats', methods=['GET'])
@api_login_required
def dataset_stats():
    """GET /api/datasets/stats - 获取数据集统计"""
    user = _get_current_user()
    stats = DatasetService.get_dataset_statistics(user_id=user.id)
    return jsonify({'success': True, 'data': stats})
