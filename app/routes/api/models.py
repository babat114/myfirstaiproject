"""
============================================
AI模型 API
RESTful JSON 接口
============================================
"""
import os
from flask import Blueprint, request, jsonify, current_app
from app.services.model_service import ModelService
from app.utils.decorators import api_login_required
from app.utils.auth_helpers import get_current_user

models_api_bp = Blueprint('models_api', __name__)


@models_api_bp.route('/', methods=['GET'])
@api_login_required
def list_models():
    """GET /api/models - 获取模型列表 (支持排序: ?sort_by=accuracy&sort_order=desc)"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 15, type=int)
    model_type = request.args.get('model_type')
    framework = request.args.get('framework')
    status = request.args.get('status')
    search = request.args.get('search')
    sort_by = request.args.get('sort_by', 'created_at')
    sort_order = request.args.get('sort_order', 'desc')

    user = get_current_user()
    # 支持 is_public 查询参数: ?is_public=true 仅显示公开模型
    is_public_q = request.args.get('is_public')
    is_public = None if is_public_q is None else is_public_q.lower() == 'true'
    # 支持 owner_id 查询参数; 默认显示当前用户的模型
    owner_id = request.args.get('owner_id', type=int)
    if owner_id is None and is_public is None:
        owner_id = user.id  # 未指定筛选时默认显示自己的模型

    result = ModelService.list_models(
        page=page, per_page=per_page,
        model_type=model_type, framework=framework,
        owner_id=owner_id, status=status, search=search,
        is_public=is_public,
        sort_by=sort_by, sort_order=sort_order,
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
    user = get_current_user()
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

    user = get_current_user()
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

    user = get_current_user()
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

    user = get_current_user()
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

    user = get_current_user()
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


@models_api_bp.route('/<string:model_uuid>/predict', methods=['POST'])
@api_login_required
def predict(model_uuid):
    """POST /api/models/<uuid>/predict - 使用模型进行预测"""
    import pandas as pd

    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id and not model.is_public and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    # 支持两种输入方式：JSON 数据 或 上传文件
    if request.is_json:
        data = request.get_json()
        features = data.get('features', [])
        if not features:
            return jsonify({'success': False, 'message': '请提供特征数据。'}), 400

        # features 可以是 [[v1,v2,...], ...] 或 [{col1:v1, col2:v2}, ...]
        if isinstance(features[0], dict):
            df = pd.DataFrame(features)
        else:
            df = pd.DataFrame(features)
    else:
        file = request.files.get('file')
        if not file:
            return jsonify({'success': False, 'message': '请上传数据文件或提供JSON数据。'}), 400
        fmt = file.filename.rsplit('.', 1)[-1].lower()
        if fmt == 'csv':
            df = pd.read_csv(file)
        elif fmt in ('xlsx', 'xls'):
            df = pd.read_excel(file)
        elif fmt == 'json':
            df = pd.read_json(file)
        else:
            return jsonify({'success': False, 'message': f'不支持的文件格式: {fmt}'}), 400

    from app.services.inference_service import ModelInferenceService
    result = ModelInferenceService.predict(model, df)

    if not result['success']:
        return jsonify(result), 400

    return jsonify({'success': True, 'data': result})


@models_api_bp.route('/<string:model_uuid>/evaluate', methods=['POST'])
@api_login_required
def evaluate(model_uuid):
    """POST /api/models/<uuid>/evaluate - 完整评估模型"""
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id and not model.is_public and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    from app.services.inference_service import ModelInferenceService
    result = ModelInferenceService.test_model_with_split(model)

    if not result.get('success'):
        return jsonify(result), 400

    return jsonify({'success': True, 'data': result})


@models_api_bp.route('/<string:model_uuid>/feature-importance', methods=['GET'])
@api_login_required
def feature_importance(model_uuid):
    """GET /api/models/<uuid>/feature-importance - 获取特征重要性"""
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    from app.services.inference_service import ModelInferenceService
    result = ModelInferenceService.get_feature_importance(model)
    return jsonify(result)


# ============ 模型导出与部署 API ============

@models_api_bp.route('/<string:model_uuid>/export/info', methods=['GET'])
@api_login_required
def export_info(model_uuid):
    """GET /api/models/<uuid>/export/info - 获取模型导出状态"""
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    from app.services.model_export_service import ModelExportService
    info = ModelExportService.get_export_info(model)
    return jsonify({'success': True, 'data': info})


@models_api_bp.route('/<string:model_uuid>/export/onnx', methods=['POST'])
@api_login_required
def export_onnx(model_uuid):
    """POST /api/models/<uuid>/export/onnx - 导出模型为 ONNX 格式"""
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    from app.services.model_export_service import ModelExportService
    success, message, onnx_path = ModelExportService.export_onnx(model)

    if not success:
        return jsonify({'success': False, 'message': message}), 400

    return jsonify({
        'success': True,
        'message': message,
        'data': {
            'onnx_path': onnx_path,
            'filename': os.path.basename(onnx_path) if onnx_path else None,
        }
    })


@models_api_bp.route('/<string:model_uuid>/export/deploy', methods=['POST'])
@api_login_required
def export_deploy(model_uuid):
    """POST /api/models/<uuid>/export/deploy - 生成 Docker 部署包"""
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    from app.services.model_export_service import ModelExportService
    success, message, package_dir = ModelExportService.generate_deployment_package(model)

    if not success:
        return jsonify({'success': False, 'message': message}), 400

    # 生成 zip 打包文件
    import shutil
    zip_name = f'{model.name_slug}_deploy'
    zip_base = os.path.join('experiments', 'exports', model.uuid)
    zip_path = os.path.join(zip_base, zip_name)
    shutil.make_archive(zip_path, 'zip', package_dir)

    return jsonify({
        'success': True,
        'message': message,
        'data': {
            'package_dir': package_dir,
            'zip_file': zip_path + '.zip',
            'download_url': f'/api/models/{model_uuid}/export/download/{zip_name}.zip',
        }
    })


@models_api_bp.route('/<string:model_uuid>/export/download/<path:filename>', methods=['GET'])
@api_login_required
def export_download(model_uuid, filename):
    """GET /api/models/<uuid>/export/download/<filename> - 下载导出文件"""
    from flask import send_file

    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    export_dir = os.path.join('experiments', 'exports', model.uuid)
    file_path = os.path.join(export_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({'success': False, 'message': '文件不存在。'}), 404

    return send_file(
        os.path.abspath(file_path),
        as_attachment=True,
        download_name=filename,
    )
