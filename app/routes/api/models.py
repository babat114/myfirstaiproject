"""
============================================
AI模型 API
RESTful JSON 接口
============================================
"""

import contextlib
import ipaddress
import json
import os
from datetime import UTC
from urllib.parse import urlparse

from flask import Blueprint, current_app, jsonify, request

from app import db, logger
from app._timezone import localnow
from app.services.model_service import ModelService
from app.utils.auth_helpers import get_current_user
from app.utils.decorators import api_login_required

models_api_bp = Blueprint('models_api', __name__)


@models_api_bp.route('/', methods=['GET'])
@api_login_required
def list_models():
    """获取模型列表
    ---
    tags:
      - Models
    summary: 获取模型列表
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
        name: model_type
        schema:
          type: string
        description: classification/regression/clustering/nlp/...
      - in: query
        name: framework
        schema:
          type: string
        description: sklearn/pytorch/tensorflow/transformers/onnx
      - in: query
        name: status
        schema:
          type: string
        description: trained/training/queued/failed/...
      - in: query
        name: search
        schema:
          type: string
      - in: query
        name: sort_by
        schema:
          type: string
          default: created_at
      - in: query
        name: sort_order
        schema:
          type: string
          default: desc
      - in: query
        name: is_public
        schema:
          type: string
        description: true 仅公开模型
      - in: query
        name: owner_id
        schema:
          type: integer
        description: 筛选指定用户的模型
    responses:
      200:
        description: 模型列表
    """
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 15, type=int), 100)
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
        page=page,
        per_page=per_page,
        model_type=model_type,
        framework=framework,
        owner_id=owner_id,
        status=status,
        search=search,
        is_public=is_public,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    return jsonify({'success': True, 'data': result})


@models_api_bp.route('/<string:model_uuid>', methods=['GET'])
@api_login_required
def get_model(model_uuid):
    """获取模型详情
    ---
    tags:
      - Models
    summary: 获取模型详情
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: 模型完整信息
      404:
        description: 模型不存在
    """
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    return jsonify(
        {
            'success': True,
            'data': model.to_dict(include_files=True),
        }
    )


@models_api_bp.route('/', methods=['POST'])
@api_login_required
def create_model():
    """注册新模型
    ---
    tags:
      - Models
    summary: 注册模型
    requestBody:
      content:
        application/json:
          schema:
            type: object
            required: [name]
            properties:
              name:
                type: string
              model_type:
                type: string
              framework:
                type: string
              description: {type: string}
              version:
                type: string
              hyperparameters:
                type: object
              is_public:
                type: boolean
    responses:
      201:
        description: 注册成功
      400:
        description: 缺少名称
    """
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

    return jsonify(
        {
            'success': True,
            'message': '模型注册成功。',
            'data': model.to_dict(),
        }
    ), 201


@models_api_bp.route('/<string:model_uuid>', methods=['PUT'])
@api_login_required
def update_model(model_uuid):
    """更新模型信息
    ---
    tags:
      - Models
    summary: 更新模型
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              name:
                type: string
              description: {type: string}
              is_public:
                type: boolean
              status:
                type: string
    responses:
      200:
        description: 更新成功
      403:
        description: 权限不足
      404:
        description: 模型不存在
    """
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

    return jsonify(
        {
            'success': True,
            'message': '模型已更新。',
            'data': model.to_dict(),
        }
    )


@models_api_bp.route('/<string:model_uuid>/upload', methods=['POST'])
@api_login_required
def upload_model_file(model_uuid):
    """上传模型文件
    ---
    tags:
      - Models
    summary: 上传模型文件
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
    requestBody:
      content:
        multipart/form-data:
          schema:
            type: object
            required: [model_file]
            properties:
              model_file:
                type: string
                format: binary
                description: .pkl/.pt/.keras/.h5 文件
    responses:
      200:
        description: 上传成功
      403:
        description: 权限不足
      404:
        description: 模型不存在
    """
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    file = request.files.get('model_file')
    if not file:
        return jsonify({'success': False, 'message': '请选择文件。'}), 400

    success, error = ModelService.upload_model_file(model, file, upload_folder=current_app.config['UPLOAD_FOLDER'])

    if not success:
        return jsonify({'success': False, 'message': error}), 400

    return jsonify({'success': True, 'message': '文件上传成功。'})


@models_api_bp.route('/<string:model_uuid>/metrics', methods=['PUT'])
@api_login_required
def update_metrics(model_uuid):
    """更新模型指标
    ---
    tags:
      - Models
    summary: 更新模型指标
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              accuracy:
                type: number
              precision:
                type: number
              recall:
                type: number
              f1_score:
                type: number
              r2:
                type: number
    responses:
      200:
        description: 指标已更新
      403:
        description: 权限不足
    """
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
    """删除模型
    ---
    tags:
      - Models
    summary: 删除模型
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: 删除成功
      403:
        description: 权限不足
      404:
        description: 模型不存在
    """
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
    """模型排行榜
    ---
    tags:
      - Models
    summary: 模型排行榜
    parameters:
      - in: query
        name: limit
        schema:
          type: integer
        description: 返回前N名
      - in: query
        name: metric
        schema:
          type: string
          default: accuracy
        description: 排序指标 (accuracy/precision/recall/f1/r2)
    responses:
      200:
        description: 排行榜列表
    """
    limit = request.args.get('limit', 10, type=int)
    metric = request.args.get('metric', 'accuracy')
    top = ModelService.get_top_models(limit=limit, metric=metric)
    return jsonify({'success': True, 'data': top})


@models_api_bp.route('/<string:model_uuid>/predict', methods=['POST'])
@api_login_required
def predict(model_uuid):
    """使用模型进行批量预测
    ---
    tags:
      - Models
    summary: 批量预测
    description: 支持 JSON 特征数组、CSV/Excel/JSON 文件上传、图像文件上传 (自动CNN特征提取)。
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              features:
                type: array
                description: "[[...], ...] 特征数组"
        multipart/form-data:
          schema:
            type: object
            properties:
              file:
                type: string
                format: binary
                description: CSV/Excel/JSON/Image 文件
    responses:
      200:
        description: "{predictions, probabilities, task_type, num_samples}"
      400:
        description: 模型文件缺失 / 无输入
      404:
        description: 模型不存在
    """
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
        df = pd.DataFrame(features) if isinstance(features[0], dict) else pd.DataFrame(features)
    else:
        file = request.files.get('file')
        if not file:
            return jsonify({'success': False, 'message': '请上传数据文件或提供JSON数据。'}), 400

        # 检测文件类型 — 优先判断 MIME，其次扩展名
        mime = (file.content_type or '').lower()
        fmt = file.filename.rsplit('.', 1)[-1].lower() if '.' in (file.filename or '') else ''
        image_exts = {'jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'tiff', 'tif'}

        if mime.startswith('image/') or fmt in image_exts:
            # ── 图像文件 → CNN 特征提取 → DataFrame ──
            try:
                from app.services.feature_extractor import FeatureExtractor

                image_data = file.read()

                # 获取期望特征数 (从模型元数据)
                n_features = 100  # 默认值
                try:
                    from app.services.inference_service import ModelInferenceService

                    _, metadata, _, _ = ModelInferenceService.load_model(model)
                    if metadata and metadata.get('feature_names'):
                        n_features = max(len(metadata['feature_names']), 10)
                except Exception:
                    pass

                features, feat_error = FeatureExtractor.extract_image_features(image_data, n_features)
                if feat_error:
                    return jsonify({'success': False, 'message': feat_error}), 400

                # 生成列名 (对齐 feature_names)
                try:
                    _, metadata, _, _ = ModelInferenceService.load_model(model)
                    fnames = (metadata or {}).get('feature_names', [])
                except Exception:
                    fnames = []
                if fnames and len(fnames) >= features.shape[1]:
                    cols = fnames[: features.shape[1]]
                else:
                    cols = [f'feature_{i}' for i in range(features.shape[1])]

                df = pd.DataFrame(features, columns=cols)
            except Exception as e:
                return jsonify(
                    {
                        'success': False,
                        'message': f'图像特征提取失败: {str(e)}',
                    }
                ), 400

        elif fmt == 'csv':
            df = pd.read_csv(file)
        elif fmt in ('xlsx', 'xls'):
            df = pd.read_excel(file)
        elif fmt == 'json':
            df = pd.read_json(file)
        else:
            return jsonify(
                {
                    'success': False,
                    'message': f'不支持的文件格式: {fmt or "未知"}。支持 CSV/Excel/JSON 或图像 (JPG/PNG/WebP)。',
                }
            ), 400

    from app.services.inference_service import ModelInferenceService

    result = ModelInferenceService.predict(model, df)

    if not result.get('success'):
        return jsonify({'success': False, 'message': result.get('message', result.get('error', '预测失败。'))}), 400

    return jsonify({'success': True, 'data': result})


@models_api_bp.route('/<string:model_uuid>/predict-template', methods=['GET'])
@api_login_required
def predict_template(model_uuid):
    """下载 CSV 预测模板
    ---
    tags:
      - Models
    summary: 下载预测模板
    description: >
      从模型元数据获取特征名列, 生成 utf-8-sig (BOM) CSV 文件供用户填写预测数据。
      第 1 行: 特征列名 header, 第 2-4 行: 空行 (供用户填写数据)。
      降级策略: 若元数据无 feature_names 则尝试从 dataset.summary_json.columns 获取。
      文件名: {model_name_slug}_template.csv
    parameters:
      - in: query
        name: rows
        schema:
          type: integer
          default: 3
        description: 空行数 (最大 100)
    responses:
      200:
        description: CSV 模板文件
      404:
        description: 模型不存在
    """
    import csv
    import io
    import json as _json

    from flask import Response

    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id and not model.is_public and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    empty_rows = request.args.get('rows', 3, type=int)
    empty_rows = max(1, min(empty_rows, 100))

    feature_names = []

    # 策略 1: 从模型文件元数据获取
    try:
        from app.services.inference_service import ModelInferenceService

        _, metadata, _, _ = ModelInferenceService.load_model(model)
        if metadata and metadata.get('feature_names'):
            feature_names = list(metadata['feature_names'])
    except Exception:
        pass

    # 策略 2: 降级 — 从训练数据集 summary 获取
    if not feature_names and model.training_dataset and model.training_dataset.summary_json:
        try:
            summary = (
                _json.loads(model.training_dataset.summary_json)
                if isinstance(model.training_dataset.summary_json, str)
                else model.training_dataset.summary_json
            )
            cols = list(summary.get('columns', []))
            hp = model.hyperparameters_dict
            target_col = hp.get('target_column', cols[-1] if cols else None)
            feature_names = [c for c in cols if c != target_col]
        except Exception:
            pass

    # 策略 3: 完全无特征名 → 返回错误
    if not feature_names:
        return jsonify(
            {
                'success': False,
                'message': ('无法生成 CSV 模板: 该模型没有特征列信息。请先上传模型文件或关联训练数据集。'),
            }
        ), 404

    # 生成 CSV (utf-8-sig BOM → Excel 正确识别中文)
    buf = io.StringIO()
    buf.write('﻿')  # BOM
    writer = csv.writer(buf)
    writer.writerow(feature_names)
    for _ in range(empty_rows):
        writer.writerow([''] * len(feature_names))

    csv_content = buf.getvalue()
    buf.close()

    filename = f'{model.name_slug}_template.csv'

    return Response(
        csv_content,
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename="{filename}"',
            'Content-Type': 'text/csv; charset=utf-8',
        },
    )


@models_api_bp.route('/<string:model_uuid>/predict-export', methods=['POST'])
@api_login_required
def predict_export(model_uuid):
    """批量预测并导出结果
    ---
    tags:
      - Models
    summary: 预测+导出
    description: 上传 CSV/Excel/JSON 数据文件, 运行批量预测, 返回带预测结果的文件下载。支持 ?format=csv|json。

    Query params:
      ?format=csv  返回 CSV (默认, 原始列 + prediction 列)
      ?format=json 返回 JSON [{...original, prediction, ...probabilities}]

    认证: Session / JWT / API Key
    """
    import csv
    import io

    import pandas as pd
    from flask import Response

    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id and not model.is_public and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    export_format = request.args.get('format', 'csv').lower()
    if export_format not in ('csv', 'json'):
        return jsonify({'success': False, 'message': 'format 参数仅支持 csv 或 json。'}), 400

    file = request.files.get('file')
    if not file or not file.filename:
        return jsonify({'success': False, 'message': '请上传数据文件 (CSV/Excel/JSON)。'}), 400

    # 解析上传文件
    try:
        fmt = file.filename.rsplit('.', 1)[-1].lower()
        if fmt == 'csv':
            df = pd.read_csv(file)
        elif fmt in ('xlsx', 'xls'):
            df = pd.read_excel(file)
        elif fmt == 'json':
            df = pd.read_json(file)
        else:
            return jsonify({'success': False, 'message': f'不支持的文件格式: {fmt}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': f'文件解析失败: {str(e)}'}), 400

    if df.empty:
        return jsonify({'success': False, 'message': '上传文件为空。'}), 400

    # 运行预测
    from app.services.inference_service import ModelInferenceService

    result = ModelInferenceService.predict(model, df)

    if not result.get('success'):
        return jsonify({'success': False, 'message': result.get('message', result.get('error', '预测失败。'))}), 400

    predictions = result.get('predictions', [])
    probabilities = result.get('probabilities', [])

    if export_format == 'json':
        # JSON: 每个原始行 + prediction + probabilities
        output = []
        for i, (_, row) in enumerate(df.iterrows()):
            item = row.to_dict()
            item['_prediction'] = str(predictions[i]) if i < len(predictions) else None
            if i < len(probabilities) and probabilities[i]:
                item['_probabilities'] = probabilities[i]
                item['_confidence'] = probabilities[i][0].get('probability', 0) if probabilities[i] else 0
            output.append(item)

        json_str = json.dumps(output, ensure_ascii=False, indent=2, default=str)
        filename = f'{model.name_slug}_predictions.json'
        return Response(
            json_str,
            mimetype='application/json',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Type': 'application/json; charset=utf-8',
            },
        )
    else:
        # CSV: 原始列 + prediction + confidence + top-N prob 列
        out_columns = list(df.columns) + ['prediction']
        # 收集概率类别名
        prob_classes = []
        if probabilities and probabilities[0]:
            prob_classes = [p.get('class', f'class_{j}') for j, p in enumerate(probabilities[0][:5])]
            out_columns += ['confidence'] + [f'prob_{c}' for c in prob_classes]

        buf = io.StringIO()
        buf.write('﻿')  # BOM
        writer = csv.writer(buf)
        writer.writerow(out_columns)

        for i, (_, row) in enumerate(df.iterrows()):
            out_row = [row[c] if c in df.columns else '' for c in out_columns[: len(df.columns)]]
            out_row.append(str(predictions[i]) if i < len(predictions) else '')
            if prob_classes:
                probs_i = probabilities[i] if i < len(probabilities) else None
                conf = probs_i[0].get('probability', 0) if probs_i else 0
                out_row.append(f'{conf:.4f}')
                prob_map = {p.get('class'): p.get('probability', 0) for p in (probs_i or [])}
                for pc in prob_classes:
                    out_row.append(f'{prob_map.get(pc, 0):.4f}')
            writer.writerow(out_row)

        csv_content = buf.getvalue()
        buf.close()
        filename = f'{model.name_slug}_predictions.csv'
        return Response(
            csv_content,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Type': 'text/csv; charset=utf-8',
            },
        )


@models_api_bp.route('/<string:model_uuid>/evaluate', methods=['POST'])
@api_login_required
def evaluate(model_uuid):
    """完整评估模型
    ---
    tags:
      - Models
    summary: 评估模型
    description: |
      评估模型性能。支持两种模式:
      - 默认: 使用训练数据集评估 (in-distribution split)
      - 独立测试: 传入 test_dataset_uuid 使用独立测试集评估
      返回完整指标 (accuracy, precision, recall, f1, r2, confusion_matrix 等)。
      若使用独立测试集, 结果会自动保存到模型记录的 independent_* 字段。
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
      - in: body
        name: body
        schema:
          type: object
          properties:
            test_dataset_uuid:
              type: string
              description: 可选, 独立测试集的 UUID
    responses:
      200:
        description: 评估结果
      500:
        description: 评估失败
    """
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id and not model.is_public and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    data = request.get_json(silent=True) or {}
    test_dataset_uuid = data.get('test_dataset_uuid')
    test_dataset = None

    if test_dataset_uuid:
        from sqlalchemy import select

        from app.models.dataset import Dataset

        test_dataset = db.session.execute(select(Dataset).where(Dataset.uuid == test_dataset_uuid)).scalar_one_or_none()
        if not test_dataset:
            return jsonify({'success': False, 'message': '指定的测试数据集不存在。'}), 404

    from app.services.inference_service import ModelInferenceService

    result = ModelInferenceService.test_model_with_split(model, test_dataset=test_dataset)

    if not result.get('success'):
        return jsonify({'success': False, 'message': result.get('message', result.get('error', '模型评估失败。'))}), 400

    # 如果使用独立测试集, 自动保存结果到 ModelRecord
    if test_dataset and test_dataset.is_test_set and result.get('is_independent_test'):
        try:
            ind_metrics = {
                'ind_test_accuracy': result.get('accuracy'),
                'ind_test_f1_macro': result.get('f1_macro'),
                'ind_test_f1_weighted': result.get('f1_weighted'),
                'ind_test_precision_macro': result.get('precision_macro'),
                'ind_test_recall_macro': result.get('recall_macro'),
                'test_dataset_name': test_dataset.name,
                'test_dataset_uuid': test_dataset.uuid,
                'collection_method': test_dataset.collection_method,
            }
            model.set_independent_metrics(ind_metrics)
            model.independent_test_dataset_id = test_dataset.id
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            # 非致命: 评估成功但保存独立指标失败
            result['save_independent_metrics_error'] = str(e)

    return jsonify({'success': True, 'data': result})


@models_api_bp.route('/<string:model_uuid>/feature-importance', methods=['GET'])
@api_login_required
def feature_importance(model_uuid):
    """获取特征重要性
    ---
    tags:
      - Models
    summary: 特征重要性分析
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: 特征重要性排名
      404:
        description: 模型不存在或不可用
    """
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    from app.services.inference_service import ModelInferenceService

    result = ModelInferenceService.get_feature_importance(model)
    return jsonify(result)


@models_api_bp.route('/<string:model_uuid>/quick-predict', methods=['POST'])
@api_login_required
def quick_predict(model_uuid):
    """交互式快速预测
    ---
    tags:
      - Models
    summary: 快速预测
    description: >
      支持两种模式: (1) 文本输入: {"text": "..."} — NLP模型自动TF-IDF处理,
      (2) 特征值输入: {"features": {...}} — 表格模型。
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              text:
                type: string
                description: NLP文本输入
              features:
                type: object
                description: 特征键值对
    responses:
      200:
        description: "{prediction, confidence, probabilities, model_type, input_mode}"
    """

    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id and not model.is_public and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    if not request.is_json:
        return jsonify({'success': False, 'message': '请使用 JSON 格式发送数据。'}), 400

    data = request.get_json(silent=True) or {}
    text_input = (data.get('text') or '').strip()
    features_input = data.get('features')

    if not text_input and not features_input:
        return jsonify(
            {
                'success': False,
                'message': '请提供输入数据。文本: {"text": "..."} 或 特征: {"features": {...}}',
            }
        ), 400

    # ── 输入校验 ──
    if text_input:
        if len(text_input) > 5000:
            return jsonify(
                {
                    'success': False,
                    'message': '文本过长，最大支持5000字符。',
                }
            ), 400
        # 纯符号/数字文本检测
        import re

        cleaned = re.sub(r'[\s\d\W_]', '', text_input)
        if len(cleaned) == 0:
            return jsonify(
                {
                    'success': False,
                    'message': '无法分析纯符号/数字文本，请输入有意义的中文或英文内容。',
                }
            ), 400

    input_mode = 'text' if text_input else 'features'
    hp = model.hyperparameters_dict
    hp.get('algorithm', '')

    # ── 分发到专用处理函数 ──
    if text_input:
        return _handle_text_prediction(model, text_input, input_mode)
    elif features_input:
        return _handle_features_prediction(model, features_input, input_mode)
    else:
        return jsonify({'success': False, 'message': '未知输入模式。'}), 400


def _format_quick_result(result: dict, task_type: str, input_text: str, input_mode: str, note: str | None = None):
    """格式化快速预测结果为前端友好的 JSON"""
    predictions = result.get('predictions', [])
    probabilities = result.get('probabilities', [])

    pred = str(predictions[0]) if predictions else '?'
    conf = 0.0
    probs_list = []

    if probabilities and probabilities[0]:
        probs_list = probabilities[0][:5]
        conf = probs_list[0].get('probability', 0.0) if probs_list else 0.0
    elif not probabilities:
        note = (note + '; ' if note else '') + '模型输出无概率分布，置信度不可用'

    response_data = {
        'prediction': pred,
        'confidence': round(float(conf), 4),
        'probabilities': probs_list,
        'model_type': task_type,
        'input_mode': input_mode,
        'input_preview': input_text[:200],
        'num_samples': result.get('num_samples', 1),
    }
    if note:
        response_data['note'] = note

    return jsonify(
        {
            'success': True,
            'data': response_data,
        }
    )


def _handle_text_prediction(model, text_input, input_mode):
    """处理文本输入预测 — 分发到各 NLP/sklearn/关键词匹配路径"""
    import logging

    from app.services.inference_service import ModelInferenceService

    log = logging.getLogger(__name__)

    model_obj, metadata, tokenizer, load_err = ModelInferenceService.load_model(model)
    if load_err:
        return jsonify({'success': False, 'message': load_err}), 400

    task_type = (metadata or {}).get('task_type', model.model_type)
    label_encoders = (metadata or {}).get('label_encoders', {})
    target_le = label_encoders.get('__target__')
    framework = (metadata or {}).get('framework', 'sklearn')

    # 1a. Transformer NLP
    if framework == 'transformers' and tokenizer is not None:
        pred_result = ModelInferenceService.predict_single(model_obj, tokenizer, metadata, text_input)
        if pred_result:
            return jsonify(
                {
                    'success': True,
                    'data': {
                        'prediction': pred_result.get('label', str(pred_result.get('prediction', ''))),
                        'confidence': pred_result.get('confidence', 0),
                        'probabilities': pred_result.get('probabilities', []),
                        'model_type': task_type,
                        'input_mode': 'text',
                        'input_text': text_input[:200],
                    },
                }
            )
        return jsonify({'success': False, 'message': 'NLP 模型预测失败 — 请检查模型文件是否完整。'}), 500

    # 1b. sklearn 分类模型 — 尝试 vectorizer 或关键词匹配
    if task_type == 'classification':
        # 尝试从 metadata 获取 vectorizer
        result = _try_vectorizer_predict(model, metadata, text_input, task_type, input_mode, log)
        if result is not None:
            return result

        # 无 vectorizer → 关键词匹配
        result = _try_keyword_match_predict(target_le, metadata, text_input, task_type, input_mode)
        if result is not None:
            return result

    # 1c. NLP模型 — vectorizer优先 → 情感分析fallback
    if model.model_type == 'nlp':
        return _handle_nlp_sentiment_predict(model, metadata, text_input, task_type, input_mode)

    # 1d. 回归/聚类 — 文本输入不适用
    model_type_label = model.model_type
    extra_hint = ''
    if model_type_label == 'nlp':
        extra_hint = ' (NLP模型缺少文本向量化器, 请确认训练时保存了 vectorizer)'
    return jsonify(
        {
            'success': False,
            'message': (
                f'当前模型为 {task_type} 类型, 不支持文本直接输入。{extra_hint}'
                f'请使用特征值输入: {{"features": {{"col1": 1.0, "col2": 2.0, ...}}}}'
            ),
            'data': {
                'feature_names': (metadata or {}).get('feature_names', [])[:15],
                'model_type': task_type,
            },
        }
    ), 400


def _try_vectorizer_predict(model, metadata, text_input, task_type, input_mode, log):
    """尝试使用模型保存的 TF-IDF vectorizer 做文本→特征转换后预测。成功返回 Response, 失败返回 None。"""
    import numpy as np
    import pandas as pd

    from app.services.inference_service import ModelInferenceService

    vectorizer = (metadata or {}).get('vectorizer')
    if vectorizer is None:
        return None
    try:
        X_vec = vectorizer.transform([text_input])
        nnz = X_vec.nnz if hasattr(X_vec, 'nnz') else int(np.count_nonzero(X_vec.toarray()))
        X_dense = X_vec.toarray() if hasattr(X_vec, 'toarray') else np.array(X_vec)
        feat_names = (metadata or {}).get('feature_names', [])
        # 只使用 TF-IDF 列名, 其他列 (如 id 等元数据列) 由 predict() 自动补0
        tfidf_names = [c for c in feat_names if str(c).startswith('tfidf_')]
        if tfidf_names and len(tfidf_names) == X_dense.shape[1]:
            df = pd.DataFrame(X_dense, columns=tfidf_names)
        else:
            df = pd.DataFrame(X_dense)
        result = ModelInferenceService.predict(model, df)
        if result.get('success'):
            # 短文本/低特征数提示 (合并分词器下字符级特征保证 nnz>0)
            note = '使用训练时保存的TF-IDF向量化器'
            if nnz < 1:
                note += '; 警告: 未命中任何特征词，预测不可靠'
            elif nnz < 3:
                note += f'; 提示: 输入仅命中{nnz}个特征，建议输入更长文本'
            return _format_quick_result(result, task_type, text_input, input_mode, note=note)
    except Exception as e:
        log.warning(f'Vectorizer transform failed: {e}')
    return None


def _try_keyword_match_predict(target_le, metadata, text_input, task_type, input_mode):
    """无 vectorizer 时尝试用类别关键词做模糊匹配。成功返回 Response, 失败返回 None。"""
    if target_le is None or not hasattr(target_le, 'classes_'):
        return None

    classes = list(target_le.classes_)
    text_lower = text_input.lower()
    feature_names = (metadata or {}).get('feature_names', [])
    if feature_names:
        return None  # 有特征名但无 vectorizer → 跳过关键词匹配

    best_class = None
    best_score = 0
    for cls_name in classes:
        cls_str = str(cls_name).lower()
        score = 1.0 if cls_str in text_lower else 0.0
        if score == 0 and len(cls_str) > 1:
            for word in text_lower.split():
                if len(word) > 1 and word in cls_str:
                    score = 0.5
                    break
        if score > best_score:
            best_score = score
            best_class = str(cls_name)

    if best_class and best_score > 0:
        capped_score = min(best_score, 0.5)
        probs = [
            {
                'class': str(c),
                'probability': round(
                    capped_score if str(c) == best_class else (1 - capped_score) / (len(classes) - 1), 3
                ),
            }
            for c in classes
        ]
        probs.sort(key=lambda x: x['probability'], reverse=True)
        return jsonify(
            {
                'success': True,
                'data': {
                    'prediction': best_class,
                    'confidence': round(capped_score, 3),
                    'probabilities': probs[:5],
                    'model_type': task_type,
                    'input_mode': 'text',
                    'input_text': text_input[:200],
                    'note': '关键词匹配模式 (极低置信度，仅作参考)',
                },
            }
        )
    return None


def _handle_nlp_sentiment_predict(model, metadata, text_input, task_type, input_mode):
    """NLP 模型情感分析: vectorizer 优先, 失败则回退到内置情感词典"""
    import numpy as np
    import pandas as pd

    from app.services.feature_extractor import FeatureExtractor
    from app.services.inference_service import ModelInferenceService

    # 优先使用 vectorizer
    vectorizer = (metadata or {}).get('vectorizer')
    if vectorizer is not None:
        try:
            X_vec = vectorizer.transform([text_input])
            nnz = X_vec.nnz if hasattr(X_vec, 'nnz') else int(np.count_nonzero(X_vec.toarray()))
            X_dense = X_vec.toarray() if hasattr(X_vec, 'toarray') else np.array(X_vec)

            # nnz=0: 输入文本完全不在训练词汇表中, 模型预测不可靠
            if nnz == 0:
                logger.warning(
                    f'quick-predict: nnz=0 for text="{text_input[:50]}", falling back to sentiment dictionary'
                )
                raise ValueError('nnz=0, 回退到情感词典')

            feat_names = (metadata or {}).get('feature_names', [])
            if feat_names and len(feat_names) == X_dense.shape[1]:
                df = pd.DataFrame(X_dense, columns=feat_names)
            else:
                df = pd.DataFrame(X_dense)
            result = ModelInferenceService.predict(model, df)
            if result.get('success'):
                note = '使用训练时保存的TF-IDF向量化器'
                if nnz < 3:
                    note += f'; 提示: 输入仅命中{nnz}个特征，结果仅供参考'

                # nnz极低时 (<3): 交叉验证情感词典, 若冲突则降低置信度
                if nnz < 3:
                    sentiment = FeatureExtractor.analyze_sentiment(text_input)
                    sent_label = sentiment['label']
                    pred_label = str(result['predictions'][0]) if result.get('predictions') else ''
                    sent_pos = sentiment.get('positive_count', 0)
                    sent_neg = sentiment.get('negative_count', 0)
                    if sent_pos + sent_neg > 0 and sent_label != '中性' and sent_label != pred_label:
                        note += (
                            f'; 注意: 情感词典判定为「{sent_label}」'
                            f'(正{sent_pos}/负{sent_neg}), 与模型预测冲突, 建议输入更长文本'
                        )

                return _format_quick_result(result, task_type, text_input, input_mode, note=note)
        except Exception:
            pass

    # 回退: 内置情感词典
    sentiment = FeatureExtractor.analyze_sentiment(text_input)
    sent_label = sentiment['label']
    sent_conf = sentiment['confidence']
    sent_pos = sentiment.get('positive_count', 0)
    sent_neg = sentiment.get('negative_count', 0)

    if sent_pos + sent_neg > 0:
        total = sent_pos + sent_neg
        sent_probs = [
            {'class': '正面', 'probability': round(sent_pos / total, 3)},
            {'class': '负面', 'probability': round(sent_neg / total, 3)},
        ]
    else:
        sent_probs = [{'class': '中性', 'probability': 1.0}]

    return jsonify(
        {
            'success': True,
            'data': {
                'prediction': sent_label,
                'confidence': sent_conf,
                'probabilities': sent_probs,
                'model_type': 'nlp',
                'input_mode': 'text',
                'input_preview': text_input[:200],
                'num_samples': 1,
                'note': '关键词情感分析 (基于内置情感词典, 模型未保存向量化器)',
                'positive_count': sent_pos,
                'negative_count': sent_neg,
            },
        }
    )


def _handle_features_prediction(model, features_input, input_mode):
    """处理特征值输入预测"""
    import pandas as pd

    from app.services.inference_service import ModelInferenceService

    try:
        df = pd.DataFrame([features_input])
        result = ModelInferenceService.predict(model, df)
        if not result.get('success'):
            return jsonify({'success': False, 'message': result.get('error', '预测失败')}), 400
        return _format_quick_result(
            result, result.get('task_type', model.model_type), str(features_input)[:200], input_mode
        )
    except Exception as e:
        return jsonify({'success': False, 'message': f'特征解析失败: {str(e)}'}), 400


# ============ 模型文件直接下载 ============


@models_api_bp.route('/<string:model_uuid>/download', methods=['GET'])
@api_login_required
def download_model_file(model_uuid):
    """下载原始模型文件
    ---
    tags:
      - Models
    summary: 下载模型文件
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
      - in: query
        name: file
        schema:
          type: string
        description: "可选指定文件名 (sklearn: *.pkl, pytorch: *.pt, keras: *.keras)"
    responses:
      200:
        description: 文件下载
      404:
        description: 模型文件不存在
    """
    from flask import send_file

    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id and not model.is_public and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    # 确定要下载的文件
    file_name = request.args.get('file')
    if file_name:
        # 安全检查: 仅允许下载模型相关文件
        safe_name = os.path.basename(file_name)
        model_dir = os.path.dirname(model.model_file_path) if model.model_file_path else ''
        file_path = os.path.join(model_dir, safe_name) if model_dir else None
    else:
        file_path = model.model_file_path

    if not file_path or not os.path.exists(file_path):
        return jsonify({'success': False, 'message': '模型文件不存在。请先上传模型文件。'}), 404

    download_name = os.path.basename(file_path)
    return send_file(
        os.path.abspath(file_path),
        as_attachment=True,
        download_name=download_name,
    )


# ============ 模型导出与部署 API ============


@models_api_bp.route('/<string:model_uuid>/export/info', methods=['GET'])
@api_login_required
def export_info(model_uuid):
    """获取模型导出状态
    ---
    tags:
      - Models
    summary: 导出状态
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: ONNX/Docker 导出状态 + 可下载文件列表
    """
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    from app.services.model_export_service import ModelExportService

    info = ModelExportService.get_export_info(model)
    return jsonify({'success': True, 'data': info})


@models_api_bp.route('/<string:model_uuid>/export/onnx', methods=['POST'])
@api_login_required
def export_onnx(model_uuid):
    """导出模型为 ONNX 格式
    ---
    tags:
      - Models
    summary: ONNX导出
    description: 将 sklearn/PyTorch 模型转换为 ONNX 格式 (同步, 限时90s)。
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: ONNX 导出成功
      202:
        description: 任务已启动 (异步)
      400:
        description: 不支持ONNX的模型类型
    """
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

    return jsonify(
        {
            'success': True,
            'message': message,
            'data': {
                'onnx_path': onnx_path,
                'filename': os.path.basename(onnx_path) if onnx_path else None,
            },
        }
    )


@models_api_bp.route('/<string:model_uuid>/export/deploy', methods=['POST'])
@api_login_required
def export_deploy(model_uuid):
    """生成 Docker 部署包
    ---
    tags:
      - Models
    summary: Docker部署包
    description: 生成包含 serve.py + Dockerfile + docker-compose.yml + requirements.txt + 模型权重的完整部署包 (zip)。
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: 部署包已生成
      400:
        description: 生成失败
    """
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    from app.services.model_export_service import ModelExportService

    success, message, package_dir, zip_file = ModelExportService.generate_deployment_package(model)

    if not success:
        return jsonify({'success': False, 'message': message}), 400

    return jsonify(
        {
            'success': True,
            'message': message,
            'data': {
                'package_dir': package_dir,
                'zip_file': zip_file,
                'download_url': f'/api/models/{model_uuid}/export/download/{zip_file}',
            },
        }
    )


@models_api_bp.route('/<string:model_uuid>/export/download/<path:filename>', methods=['GET'])
@api_login_required
def export_download(model_uuid, filename):
    """下载导出文件
    ---
    tags:
      - Models
    summary: 下载导出文件
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
      - in: path
        name: filename
        required: true
        schema:
          type: string
        description: 导出文件名 (.onnx / .zip)
    responses:
      200:
        description: 文件下载
      404:
        description: 文件不存在
    """
    from flask import send_file

    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    # 权限检查 — 仅模型所有者/管理员/公开模型可下载
    user = get_current_user()
    if model.owner_id != user.id and not model.is_public and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    export_dir = os.path.join('experiments', 'exports', model.uuid)
    # 路径遍历防护: 确保解析后的绝对路径在导出目录内
    resolved_path = os.path.abspath(os.path.join(export_dir, filename))
    export_root = os.path.abspath(export_dir)
    if not resolved_path.startswith(export_root + os.sep) and resolved_path != export_root:
        logger.warning(f'路径遍历尝试: model={model_uuid}, filename={filename}')
        return jsonify({'success': False, 'message': '非法的文件路径。'}), 403

    if not os.path.exists(resolved_path):
        return jsonify({'success': False, 'message': '文件不存在。'}), 404

    return send_file(
        resolved_path,
        as_attachment=True,
        download_name=os.path.basename(filename),
    )


# ============ 异步导出进度跟踪 ============


@models_api_bp.route('/<string:model_uuid>/export/async/<string:export_type>', methods=['POST'])
@api_login_required
def export_async(model_uuid, export_type):
    """启动异步导出任务
    ---
    tags:
      - Models
    summary: 异步导出
    description: 后台执行 ONNX 转换或 Docker 部署包生成, 返回 task_id 供 GET /export/status 轮询进度。
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
      - in: path
        name: export_type
        required: true
        schema:
          type: string
          enum:
            - onnx
            - deploy
    responses:
      202:
        description: 任务已启动, 返回 task_id
      400:
        description: 无效的导出类型
    """
    if export_type not in ('onnx', 'deploy'):
        return jsonify({'success': False, 'message': 'export_type 仅支持 onnx 或 deploy。'}), 400

    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id and not model.is_public and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    from app.services.export_task_tracker import ExportTaskTracker
    from app.services.model_export_service import ModelExportService

    tracker = ExportTaskTracker()
    task_id = tracker.create_task(model_uuid, export_type)

    fn = ModelExportService.export_onnx if export_type == 'onnx' else ModelExportService.generate_deployment_package

    tracker.run_async(task_id, fn, model)

    return jsonify(
        {
            'success': True,
            'message': f'{export_type} 导出任务已启动',
            'data': {'task_id': task_id, 'export_type': export_type},
        }
    ), 202


# ============ HuggingFace 风格模型卡片 ============


@models_api_bp.route('/<string:model_uuid>/model-card', methods=['GET'])
@api_login_required
def model_card(model_uuid):
    """获取 HuggingFace 风格模型卡片
    ---
    tags:
      - Models
    summary: 模型卡片
    description: >
      返回完整的 HuggingFace 风格模型卡片: YAML 元数据头 / 模型描述 / 训练过程 / 评估结果 / 使用方法 / BibTeX 引用。
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
      - in: query
        name: format
        schema:
          type: string
          enum:
            - markdown
            - json
          default: markdown
        description: 输出格式
    responses:
      200:
        description: 模型卡片 (Markdown 或 JSON)
    """
    from flask import Response

    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id and not model.is_public and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    output_format = request.args.get('format', 'markdown').lower()

    markdown = ModelService.generate_model_card(model)

    if output_format == 'json':
        return jsonify(
            {
                'success': True,
                'data': {
                    'model_name': model.name,
                    'model_uuid': model.uuid,
                    'model_version': model.version,
                    'framework': model.framework,
                    'task_type': model.model_type,
                    'markdown': markdown,
                },
            }
        )

    # 默认返回 Markdown
    filename = f'{model.name_slug}_model_card.md'
    return Response(
        markdown,
        mimetype='text/markdown; charset=utf-8',
        headers={
            'Content-Disposition': f'inline; filename="{filename}"',
        },
    )


# ============ 直接模型服务端点 (镜像 Docker 容器 API) ============


@models_api_bp.route('/<string:model_uuid>/serve', methods=['POST'])
@api_login_required
def serve_model(model_uuid):
    """直接模型推理端点 (本地)
    ---
    tags:
      - Models
    summary: 本地推理
    description: 镜像 Docker serve.py 的 /predict 契约, 本地加载模型并推理。非 dict JSON body 或加载失败返回 503。
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [features]
            properties:
              features:
                type: array
                description: "[[...], ...] 特征数组"
    responses:
      200:
        description: "{predictions: [...], task_type: classification}"
      503:
        description: 模型加载失败 / JSON 格式错误
    """
    import pandas as pd

    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。', 'code': 404}), 404

    user = get_current_user()
    if model.owner_id != user.id and not model.is_public and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。', 'code': 403}), 403

    if not request.is_json:
        return jsonify({'success': False, 'message': '请求体必须是 JSON 格式。', 'code': 400}), 400

    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify(
            {
                'success': False,
                'message': '请求体必须是 JSON 对象 (dict)。',
                'code': 400,
            }
        ), 400
    features = data.get('features')

    if features is None:
        return jsonify(
            {
                'success': False,
                'message': '缺少 features 字段。请提供 {"features": [[...], ...]}',
                'code': 400,
            }
        ), 400

    if not isinstance(features, list) or len(features) == 0:
        return jsonify(
            {
                'success': False,
                'message': 'features 必须是非空数组。',
                'code': 400,
            }
        ), 400

    # 构建 DataFrame
    try:
        if isinstance(features[0], dict):
            df = pd.DataFrame(features)
        elif isinstance(features[0], list):
            df = pd.DataFrame(features)
            # 尝试使用模型特征名作为列名
            try:
                from app.services.inference_service import ModelInferenceService

                _, metadata, _, _ = ModelInferenceService.load_model(model)
                if metadata and metadata.get('feature_names'):
                    fnames = metadata['feature_names']
                    if len(fnames) == df.shape[1]:
                        df.columns = fnames
            except Exception:
                pass
        else:
            return jsonify(
                {
                    'success': False,
                    'message': 'features 每项必须是数组 [v1, v2, ...] 或字典 {col: val}。',
                    'code': 400,
                }
            ), 400
    except Exception as e:
        return jsonify({'success': False, 'message': f'features 解析失败: {str(e)}', 'code': 400}), 400

    # 执行预测
    try:
        from app.services.inference_service import ModelInferenceService

        result = ModelInferenceService.predict(model, df)
    except Exception as e:
        from app.utils.helpers import sanitize_service_error

        return jsonify(
            {
                'success': False,
                'message': sanitize_service_error(e, '模型推理异常'),
                'code': 500,
            }
        ), 500

    if not result.get('success'):
        error_msg = result.get('error', '模型预测失败')
        # 判断是否为模型加载错误 (503) vs 数据错误 (400)
        if '模型文件不存在' in error_msg or '无法加载' in error_msg or '加载失败' in error_msg or '不支持' in error_msg:
            return jsonify({'success': False, 'message': error_msg, 'code': 503}), 503
        return jsonify({'success': False, 'message': error_msg, 'code': 400}), 400

    return jsonify(
        {
            'predictions': result.get('predictions', []),
            'task_type': result.get('task_type', model.model_type),
        }
    )


# ============ 部署健康检查 ============

# SSRF 防护: 内部/私有 IP 段黑名单
_SSRF_BLOCKED_NETWORKS = [
    ipaddress.ip_network('127.0.0.0/8'),  # loopback
    ipaddress.ip_network('10.0.0.0/8'),  # private A
    ipaddress.ip_network('172.16.0.0/12'),  # private B
    ipaddress.ip_network('192.168.0.0/16'),  # private C
    ipaddress.ip_network('169.254.0.0/16'),  # link-local / AWS metadata
    ipaddress.ip_network('0.0.0.0/8'),  # current network
    ipaddress.ip_network('100.64.0.0/10'),  # CGNAT
    ipaddress.ip_network('198.18.0.0/15'),  # benchmark
    ipaddress.ip_network('224.0.0.0/4'),  # multicast
    ipaddress.ip_network('240.0.0.0/4'),  # reserved
]


def _validate_deployment_url(url: str) -> bool:
    """验证部署URL是否安全 (防SSRF: 仅允许公网IP/域名, 禁止内网地址)"""
    if not url:
        return True
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        # localhost 允许 (Docker 本地部署)
        if hostname in ('localhost', '127.0.0.1'):
            return True
        # DNS 解析 → IP 检查
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast:
            return False
        return all(addr not in net for net in _SSRF_BLOCKED_NETWORKS)
    except ValueError:
        # hostname 不是有效IP → 域名, 允许 (部署到公网)
        return True


@models_api_bp.route('/<string:model_uuid>/deploy/health', methods=['GET'])
@api_login_required
def deploy_health(model_uuid):
    """检查 Docker 部署健康状态
    ---
    tags:
      - Models
    summary: 部署健康检查
    description: >
      验证部署包是否存在, 尝试连接部署容器 /health 端点 (SSRF防护: 仅公网IP/域名)。
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
    responses:
      200:
        description: "{status: healthy|unreachable|not_deployed, deploy_exists, container_info}"
    """
    import json as _json
    import urllib.error
    import urllib.request

    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id and not model.is_public and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    # 检查部署包是否存在
    deploy_dir = os.path.join('experiments', 'exports', model.uuid, 'deploy')
    deploy_exists = os.path.isdir(deploy_dir)

    # 收集部署包文件信息
    package_files = []
    if deploy_exists:
        try:
            package_files = sorted(os.listdir(deploy_dir))
        except OSError:
            package_files = []

    # 确定健康检查 URL (优先 model.deployment_url, 其次默认 localhost:8000)
    health_url = None
    if model.deployment_url:
        if not _validate_deployment_url(model.deployment_url):
            return jsonify(
                {
                    'success': False,
                    'message': '部署URL不安全: 仅允许公网域名或 localhost。',
                    'code': 400,
                }
            ), 400
        health_url = model.deployment_url.rstrip('/') + '/health'
    elif deploy_exists:
        health_url = 'http://localhost:8000/health'

    # 尝试健康检查
    container_info = {
        'deploy_exists': deploy_exists,
        'package_files': package_files,
        'health_url': health_url,
    }
    status = 'not_deployed'

    if health_url and deploy_exists:
        try:
            req = urllib.request.Request(health_url, method='GET')
            req.add_header('User-Agent', 'AI-Platform-DeployHealthCheck/1.0')
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    body = resp.read().decode('utf-8')
                    try:
                        health_data = _json.loads(body)
                    except _json.JSONDecodeError:
                        health_data = {'raw_response': body[:500]}
                    status = 'healthy'
                    container_info['health_response'] = health_data
                    container_info['http_status'] = 200
                else:
                    status = 'unreachable'
                    container_info['http_status'] = resp.status
        except urllib.error.URLError as e:
            status = 'unreachable'
            container_info['error'] = f'连接失败: {str(e.reason)}'
        except urllib.error.HTTPError as e:
            status = 'unreachable'
            container_info['error'] = f'HTTP {e.code}: {e.reason}'
            container_info['http_status'] = e.code
        except Exception as e:
            status = 'unreachable'
            container_info['error'] = f'检查异常: {str(e)}'

    from datetime import datetime

    return jsonify(
        {
            'success': True,
            'data': {
                'status': status,
                'deploy_exists': deploy_exists,
                'container_info': container_info,
                'checked_at': datetime.now(UTC).isoformat(),
            },
        }
    )


@models_api_bp.route('/<string:model_uuid>/export/status', methods=['GET'])
@api_login_required
def export_status(model_uuid):
    """查询导出任务进度
    ---
    tags:
      - Models
    summary: 导出进度
    description: 轮询异步导出任务的实时进度 (配合 POST /export/async 使用)。
    parameters:
      - in: path
        name: model_uuid
        required: true
        schema:
          type: string
      - in: query
        name: task_id
        required: true
        schema:
          type: string
        description: 异步导出返回的 task_id
    responses:
      200:
        description: "{task_id, status: pending|running|completed|failed, progress, message, result}"
    """
    task_id = request.args.get('task_id', '').strip()
    if not task_id:
        return jsonify({'success': False, 'message': '缺少 task_id 参数。'}), 400

    # 模型存在性 + 权限检查
    model = ModelService.get_model_by_uuid(model_uuid)
    if not model:
        return jsonify({'success': False, 'message': '模型不存在。'}), 404

    user = get_current_user()
    if model.owner_id != user.id and not model.is_public and not user.is_admin:
        return jsonify({'success': False, 'message': '权限不足。'}), 403

    from app.services.export_task_tracker import ExportTaskTracker

    tracker = ExportTaskTracker()
    task = tracker.get_task(task_id)

    if not task:
        return jsonify({'success': False, 'message': '任务不存在或已过期。'}), 404

    if task.get('model_uuid') != model_uuid:
        return jsonify({'success': False, 'message': 'task_id 与模型不匹配。'}), 400

    return jsonify({'success': True, 'data': task})


# ============ 模型导入 (双向) ============
# 辅助函数 (模块级, 供 import_model_preview 使用)


def _model_class_name(model_obj) -> str:
    """获取模型对象的类名"""
    if model_obj is None:
        return ''
    if hasattr(model_obj, '__class__') and hasattr(model_obj.__class__, '__name__'):
        return model_obj.__class__.__name__
    return type(model_obj).__name__


def _extract_from_pkl(extracted: dict, pkl_path: str):
    """从 .pkl 模型文件中提取元数据"""
    import pickle

    try:
        with open(pkl_path, 'rb') as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict):
            extracted['algorithm'] = _model_class_name(bundle.get('model'))
            extracted['feature_names'] = bundle.get('feature_names', [])
            extracted['class_labels'] = bundle.get('class_labels', [])
            task_type = bundle.get('task_type')
            if task_type:
                extracted['model_type'] = task_type
            if bundle.get('vectorizer') is not None:
                extracted['model_type'] = extracted.get('model_type') or 'nlp'
            input_dim = bundle.get('input_dim')
            if input_dim:
                extracted['input_dimension'] = int(input_dim)
        else:
            extracted['algorithm'] = _model_class_name(bundle)
    except Exception:
        pass


@models_api_bp.route('/import/preview', methods=['POST'])
@api_login_required
def import_model_preview():
    """预览导入 — 上传模型文件, 自动提取元数据 + AI 推荐
    ---
    tags:
      - Models
    summary: 导入预览
    description: >
      上传 .pkl 或 .zip (部署包) 文件, 自动提取模型信息并生成 AI 推荐
      (名称/描述/版本)。前端展示后由 /import/confirm 确认创建。
    requestBody:
      content:
        multipart/form-data:
          schema:
            type: object
            required: [model_file]
            properties:
              model_file:
                type: string
                format: binary
                description: .pkl 模型文件 或 .zip 部署包
    responses:
      200:
        description: 提取信息 + AI 推荐
      400:
        description: 无效文件
    """
    file = request.files.get('model_file')
    if not file or not file.filename:
        return jsonify({'success': False, 'message': '请选择模型文件。'}), 400

    import shutil as _su
    import tempfile

    from app.services.model_recommender import ModelRecommender

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    temp_dir = tempfile.mkdtemp(prefix='model_import_')
    temp_path = os.path.join(temp_dir, filename)

    try:
        file.save(temp_path)

        extracted = {
            'filename': filename,
            'file_size': os.path.getsize(temp_path),
            'framework': None,
            'model_type': None,
            'algorithm': None,
            'feature_names': [],
            'class_labels': [],
            'input_dimension': None,
            'has_metadata_json': False,
            'has_model_file': False,
            'metadata_fields': [],
            'existing_metadata': {},
        }

        # ── 解析 ZIP 部署包 ──
        if ext == '.zip':
            import zipfile

            with zipfile.ZipFile(temp_path, 'r') as zf:
                namelist = zf.namelist()
                meta_members = [n for n in namelist if n.endswith('metadata.json')]
                if meta_members:
                    extracted['has_metadata_json'] = True
                    with zf.open(meta_members[0]) as mf:
                        meta_data = json.loads(mf.read().decode('utf-8'))
                    extracted['existing_metadata'] = meta_data
                    extracted['framework'] = meta_data.get('framework')
                    extracted['model_type'] = meta_data.get('model_type')
                    extracted['metadata_fields'] = [
                        k for k in meta_data if not k.startswith('_') and k != 'model_file_path'
                    ]
                    infer = meta_data.get('_inference_meta', {})
                    extracted['algorithm'] = infer.get('algorithm')
                    extracted['feature_names'] = infer.get('feature_names', [])
                    extracted['class_labels'] = infer.get('class_labels', [])

                model_exts = ('.pkl', '.pt', '.keras', '.h5', '.joblib')
                model_members = [
                    n
                    for n in namelist
                    if any(n.endswith(e) for e in model_exts) and os.path.basename(n).startswith('model')
                ]
                if model_members:
                    extracted['has_model_file'] = True
                    if not extracted['has_metadata_json']:
                        zf.extract(model_members[0], temp_dir)
                        pkl_path = os.path.join(temp_dir, model_members[0])
                        _extract_from_pkl(extracted, pkl_path)

        # ── 解析 .pkl / .pt / .joblib ──
        elif ext in ('.pkl', '.pt', '.pth', '.joblib', '.h5', '.keras'):
            extracted['has_model_file'] = True
            _extract_from_pkl(extracted, temp_path)
        else:
            return jsonify(
                {'success': False, 'message': f'不支持的文件格式 "{ext}"。支持: .pkl, .pt, .h5, .keras, .zip (部署包)'}
            ), 400

        # 推断 model_type
        if not extracted.get('model_type'):
            if extracted.get('class_labels'):
                extracted['model_type'] = 'classification'
            elif extracted.get('algorithm'):
                algo = extracted['algorithm']
                if any(k in algo for k in ('Cluster', 'Means', 'DBSCAN', 'Agglomerative')):
                    extracted['model_type'] = 'clustering'
                elif any(k in algo for k in ('Regressor', 'Regression')):
                    extracted['model_type'] = 'regression'
                else:
                    extracted['model_type'] = 'classification'

        # ── AI 推荐 ──
        recommendations = ModelRecommender.recommend(extracted)

        return jsonify(
            {
                'success': True,
                'data': {
                    'extracted': extracted,
                    'recommendations': recommendations,
                    'requires_manual': ['name', 'description', 'version'],
                },
            }
        )

    except Exception as e:
        logger.error(f'导入预览失败: {e}', exc_info=True)
        return jsonify({'success': False, 'message': f'解析失败: {str(e)}'}), 400
    finally:
        with contextlib.suppress(Exception):
            _su.rmtree(temp_dir, ignore_errors=True)


@models_api_bp.route('/import/confirm', methods=['POST'])
@api_login_required
def import_model_confirm():
    """确认导入 — 创建模型记录
    ---
    tags:
      - Models
    summary: 确认导入
    description: >
      使用 /import/preview 返回的元数据 + AI 推荐 (或用户修改),
      上传模型文件并创建完整的模型记录。
    requestBody:
      content:
        multipart/form-data:
          schema:
            type: object
            required: [model_file, name, model_type]
            properties:
              model_file:
                type: string
                format: binary
              name:
                type: string
              description:
                type: string
              version:
                type: string
              model_type:
                type: string
              framework:
                type: string
              metrics:
                type: string
                description: JSON 字符串
              hyperparameters:
                type: string
                description: JSON 字符串
              is_public:
                type: boolean
    responses:
      201:
        description: 导入成功
      400:
        description: 参数错误
    """
    file = request.files.get('model_file')
    if not file or not file.filename:
        return jsonify({'success': False, 'message': '请选择模型文件。'}), 400

    name = (request.form.get('name') or '').strip()
    if not name:
        return jsonify({'success': False, 'message': '模型名称不能为空。'}), 400

    user = get_current_user()
    model_type = request.form.get('model_type', 'other')
    framework = request.form.get('framework') or None
    description = request.form.get('description') or None
    version = request.form.get('version', '1.0.0').strip()

    metrics = None
    metrics_raw = request.form.get('metrics')
    if metrics_raw:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            metrics = json.loads(metrics_raw)

    hyperparameters = None
    hp_raw = request.form.get('hyperparameters')
    if hp_raw:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            hyperparameters = json.loads(hp_raw)

    is_public = request.form.get('is_public') in ('true', 'True', '1', 'on')

    if not framework:
        ext = os.path.splitext(file.filename)[1].lower()
        fw_map = {
            '.pt': 'pytorch',
            '.pth': 'pytorch',
            '.pkl': 'sklearn',
            '.joblib': 'sklearn',
            '.h5': 'tensorflow',
            '.keras': 'tensorflow',
        }
        framework = fw_map.get(ext)

    from werkzeug.utils import secure_filename

    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    model_dir = os.path.join(upload_folder, 'models')
    os.makedirs(model_dir, exist_ok=True)

    try:
        model, error = ModelService.import_model(
            user=user,
            name=name,
            model_type=model_type,
            framework=framework,
            description=description,
            version=version,
            hyperparameters=hyperparameters,
            metrics=metrics,
            is_public=is_public,
        )
        if error:
            return jsonify({'success': False, 'message': error}), 400

        original_name = secure_filename(file.filename)
        unique_name = f'{model.uuid}_{original_name}'
        file_path = os.path.join(model_dir, unique_name)
        file.save(file_path)
        file_size = os.path.getsize(file_path)

        model.model_file_path = file_path
        model.file_size = file_size
        model.updated_at = localnow()
        db.session.commit()

        logger.info(f'模型导入确认完成: {model.name} v{model.version} (uuid={model.uuid})')
        return jsonify(
            {
                'success': True,
                'message': '模型导入成功！',
                'data': model.to_dict(),
            }
        ), 201

    except Exception as e:
        db.session.rollback()
        logger.error(f'模型导入确认失败: {e}', exc_info=True)
        return jsonify({'success': False, 'message': f'导入失败: {str(e)}'}), 400
