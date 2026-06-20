"""
============================================
AI模型 API
RESTful JSON 接口
============================================
"""
import os
import json
import re
import ipaddress
from urllib.parse import urlparse
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

        # 检测文件类型 — 优先判断 MIME，其次扩展名
        mime = (file.content_type or '').lower()
        fmt = file.filename.rsplit('.', 1)[-1].lower() if '.' in (file.filename or '') else ''
        image_exts = {'jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'tiff', 'tif'}

        if mime.startswith('image/') or fmt in image_exts:
            # ── 图像文件 → CNN 特征提取 → DataFrame ──
            try:
                import numpy as np
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

                features, feat_error = FeatureExtractor.extract_image_features(
                    image_data, n_features
                )
                if feat_error:
                    return jsonify({'success': False, 'message': feat_error}), 400

                # 生成列名 (对齐 feature_names)
                try:
                    _, metadata, _, _ = ModelInferenceService.load_model(model)
                    fnames = (metadata or {}).get('feature_names', [])
                except Exception:
                    fnames = []
                if fnames and len(fnames) >= features.shape[1]:
                    cols = fnames[:features.shape[1]]
                else:
                    cols = [f'feature_{i}' for i in range(features.shape[1])]

                df = pd.DataFrame(features, columns=cols)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'图像特征提取失败: {str(e)}',
                }), 400

        elif fmt == 'csv':
            df = pd.read_csv(file)
        elif fmt in ('xlsx', 'xls'):
            df = pd.read_excel(file)
        elif fmt == 'json':
            df = pd.read_json(file)
        else:
            return jsonify({
                'success': False,
                'message': f'不支持的文件格式: {fmt or "未知"}。支持 CSV/Excel/JSON 或图像 (JPG/PNG/WebP)。'
            }), 400

    from app.services.inference_service import ModelInferenceService
    result = ModelInferenceService.predict(model, df)

    if not result['success']:
        return jsonify(result), 400

    return jsonify({'success': True, 'data': result})


@models_api_bp.route('/<string:model_uuid>/predict-template', methods=['GET'])
@api_login_required
def predict_template(model_uuid):
    """GET /api/models/<uuid>/predict-template — 下载 CSV 预测模板

    从模型元数据获取特征名列, 生成 utf-8-sig (BOM) CSV:
      - 第 1 行: 特征列名 header
      - 第 2-4 行: 空行 (供用户填写数据)

    降级策略: 若元数据无 feature_names → 尝试从 dataset.summary_json.columns
    文件名: {model_name_slug}_template.csv

    Query params:
      ?rows=10  指定空行数 (默认 3, 最大 100)
    """
    import io
    import csv
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
            summary = _json.loads(model.training_dataset.summary_json) if isinstance(
                model.training_dataset.summary_json, str
            ) else model.training_dataset.summary_json
            cols = list(summary.get('columns', []))
            hp = model.hyperparameters_dict
            target_col = hp.get('target_column', cols[-1] if cols else None)
            feature_names = [c for c in cols if c != target_col]
        except Exception:
            pass

    # 策略 3: 完全无特征名 → 返回错误
    if not feature_names:
        return jsonify({
            'success': False,
            'message': (
                '无法生成 CSV 模板: 该模型没有特征列信息。'
                '请先上传模型文件或关联训练数据集。'
            ),
        }), 404

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
        }
    )


@models_api_bp.route('/<string:model_uuid>/predict-export', methods=['POST'])
@api_login_required
def predict_export(model_uuid):
    """POST /api/models/<uuid>/predict-export — 批量预测并导出结果

    上传 CSV/Excel/JSON 数据文件, 运行批量预测, 返回带预测结果的文件下载。

    Query params:
      ?format=csv  返回 CSV (默认, 原始列 + prediction 列)
      ?format=json 返回 JSON [{...original, prediction, ...probabilities}]

    认证: Session / JWT / API Key
    """
    import io
    import csv
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
        return jsonify(result), 400

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
            }
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
            out_row = [row[c] if c in df.columns else '' for c in out_columns[:len(df.columns)]]
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
            }
        )


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


@models_api_bp.route('/<string:model_uuid>/quick-predict', methods=['POST'])
@api_login_required
def quick_predict(model_uuid):
    """POST /api/models/<uuid>/quick-predict — 交互式快速预测

    支持两种输入模式:
      1. 文本输入 (NLP/情感分析):
         {"text": "这部电影太棒了！"}
      2. 特征值输入 (表格数据):
         {"features": {"age": 35, "income": 50000}}

    返回:
      {success, prediction, confidence, probabilities, model_type, input_mode}
    """
    import pandas as pd
    import numpy as np

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
        return jsonify({
            'success': False,
            'message': '请提供输入数据。文本: {"text": "..."} 或 特征: {"features": {...}}',
        }), 400

    # ── 输入校验 ──
    if text_input:
        if len(text_input) > 5000:
            return jsonify({
                'success': False,
                'message': '文本过长，最大支持5000字符。',
            }), 400
        # 纯符号/数字文本检测
        import re
        cleaned = re.sub(r'[\s\d\W_]', '', text_input)
        if len(cleaned) == 0:
            return jsonify({
                'success': False,
                'message': '无法分析纯符号/数字文本，请输入有意义的中文或英文内容。',
            }), 400

    from app.services.inference_service import ModelInferenceService

    input_mode = 'text' if text_input else 'features'
    hp = model.hyperparameters_dict
    algo = hp.get('algorithm', '')

    # ── 分发到专用处理函数 ──
    if text_input:
        return _handle_text_prediction(model, text_input, input_mode)
    elif features_input:
        return _handle_features_prediction(model, features_input, input_mode)
    else:
        return jsonify({'success': False, 'message': '未知输入模式。'}), 400


def _format_quick_result(result: dict, task_type: str, input_text: str, input_mode: str,
                         note: str | None = None):
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

    return jsonify({
        'success': True,
        'data': response_data,
    })


def _handle_text_prediction(model, text_input, input_mode):
    """处理文本输入预测 — 分发到各 NLP/sklearn/关键词匹配路径"""
    import logging
    import pandas as pd
    import numpy as np
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
        pred_result = ModelInferenceService.predict_single(
            model_obj, tokenizer, metadata, text_input)
        if pred_result:
            return jsonify({
                'success': True,
                'data': {
                    'prediction': pred_result.get('label', str(pred_result.get('prediction', ''))),
                    'confidence': pred_result.get('confidence', 0),
                    'probabilities': pred_result.get('probabilities', []),
                    'model_type': task_type,
                    'input_mode': 'text',
                    'input_text': text_input[:200],
                }
            })
        return jsonify({
            'success': False,
            'message': 'NLP 模型预测失败 — 请检查模型文件是否完整。'
        }), 500

    # 1b. sklearn 分类模型 — 尝试 vectorizer 或关键词匹配
    if task_type == 'classification':
        # 尝试从 metadata 获取 vectorizer
        result = _try_vectorizer_predict(
            model, metadata, text_input, task_type, input_mode, log)
        if result is not None:
            return result

        # 无 vectorizer → 关键词匹配
        result = _try_keyword_match_predict(
            target_le, metadata, text_input, task_type, input_mode)
        if result is not None:
            return result

    # 1c. NLP模型 — vectorizer优先 → 情感分析fallback
    if model.model_type == 'nlp':
        return _handle_nlp_sentiment_predict(
            model, metadata, text_input, task_type, input_mode)

    # 1d. 回归/聚类 — 文本输入不适用
    model_type_label = model.model_type
    extra_hint = ''
    if model_type_label == 'nlp':
        extra_hint = ' (NLP模型缺少文本向量化器, 请确认训练时保存了 vectorizer)'
    return jsonify({
        'success': False,
        'message': (
            f'当前模型为 {task_type} 类型, 不支持文本直接输入。{extra_hint}'
            f'请使用特征值输入: {{"features": {{"col1": 1.0, "col2": 2.0, ...}}}}'
        ),
        'data': {
            'feature_names': (metadata or {}).get('feature_names', [])[:15],
            'model_type': task_type,
        }
    }), 400


def _try_vectorizer_predict(model, metadata, text_input, task_type, input_mode, log):
    """尝试使用模型保存的 TF-IDF vectorizer 做文本→特征转换后预测。成功返回 Response, 失败返回 None。"""
    import pandas as pd
    import numpy as np
    from app.services.inference_service import ModelInferenceService

    vectorizer = (metadata or {}).get('vectorizer')
    if vectorizer is None:
        return None
    try:
        X_vec = vectorizer.transform([text_input])
        X_dense = X_vec.toarray() if hasattr(X_vec, 'toarray') else np.array(X_vec)
        feat_names = (metadata or {}).get('feature_names', [])
        if feat_names and len(feat_names) == X_dense.shape[1]:
            df = pd.DataFrame(X_dense, columns=feat_names)
        else:
            df = pd.DataFrame(X_dense)
        result = ModelInferenceService.predict(model, df)
        if result.get('success'):
            return _format_quick_result(
                result, task_type, text_input, input_mode,
                note='使用训练时保存的TF-IDF向量化器'
            )
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
        probs = [{'class': str(c), 'probability': round(
            capped_score if str(c) == best_class else (1 - capped_score) / (len(classes) - 1), 3)}
            for c in classes]
        probs.sort(key=lambda x: x['probability'], reverse=True)
        return jsonify({
            'success': True,
            'data': {
                'prediction': best_class,
                'confidence': round(capped_score, 3),
                'probabilities': probs[:5],
                'model_type': task_type,
                'input_mode': 'text',
                'input_text': text_input[:200],
                'note': '关键词匹配模式 (极低置信度，仅作参考)',
            }
        })
    return None


def _handle_nlp_sentiment_predict(model, metadata, text_input, task_type, input_mode):
    """NLP 模型情感分析: vectorizer 优先, 失败则回退到内置情感词典"""
    import pandas as pd
    import numpy as np
    from app.services.inference_service import ModelInferenceService
    from app.services.feature_extractor import FeatureExtractor

    # 优先使用 vectorizer
    vectorizer = (metadata or {}).get('vectorizer')
    if vectorizer is not None:
        try:
            X_vec = vectorizer.transform([text_input])
            X_dense = X_vec.toarray() if hasattr(X_vec, 'toarray') else np.array(X_vec)
            feat_names = (metadata or {}).get('feature_names', [])
            if feat_names and len(feat_names) == X_dense.shape[1]:
                df = pd.DataFrame(X_dense, columns=feat_names)
            else:
                df = pd.DataFrame(X_dense)
            result = ModelInferenceService.predict(model, df)
            if result.get('success'):
                return _format_quick_result(
                    result, task_type, text_input, input_mode,
                    note='使用训练时保存的TF-IDF向量化器'
                )
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

    return jsonify({
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
        }
    })


def _handle_features_prediction(model, features_input, input_mode):
    """处理特征值输入预测"""
    import pandas as pd
    from app.services.inference_service import ModelInferenceService

    try:
        df = pd.DataFrame([features_input])
        result = ModelInferenceService.predict(model, df)
        if not result.get('success'):
            return jsonify({'success': False, 'message': result.get('error', '预测失败')}), 400
        return _format_quick_result(result, result.get('task_type', model.model_type),
                                    str(features_input)[:200], input_mode)
    except Exception as e:
        return jsonify({'success': False, 'message': f'特征解析失败: {str(e)}'}), 400


# ============ 模型文件直接下载 ============

@models_api_bp.route('/<string:model_uuid>/download', methods=['GET'])
@api_login_required
def download_model_file(model_uuid):
    """GET /api/models/<uuid>/download — 下载原始模型文件

    支持 ?file=model.pkl 指定文件名 (默认下载主模型文件)。
    """
    from flask import send_file, abort

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
        return jsonify({
            'success': False,
            'message': '模型文件不存在。请先上传模型文件。'
        }), 404

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


# ============ 异步导出进度跟踪 ============

@models_api_bp.route('/<string:model_uuid>/export/async/<string:export_type>', methods=['POST'])
@api_login_required
def export_async(model_uuid, export_type):
    """POST /api/models/<uuid>/export/async/<onnx|deploy> — 启动异步导出任务

    返回 task_id 供前端轮询进度。
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

    if export_type == 'onnx':
        fn = ModelExportService.export_onnx
    else:
        fn = ModelExportService.generate_deployment_package

    tracker.run_async(task_id, fn, model)

    return jsonify({
        'success': True,
        'message': f'{export_type} 导出任务已启动',
        'data': {'task_id': task_id, 'export_type': export_type},
    }), 202


# ============ HuggingFace 风格模型卡片 ============

@models_api_bp.route('/<string:model_uuid>/model-card', methods=['GET'])
@api_login_required
def model_card(model_uuid):
    """GET /api/v1/models/<uuid>/model-card — 获取 HuggingFace 风格模型卡片

    Query params:
        ?format=markdown  返回纯 Markdown 文本 (默认)
        ?format=json      返回 JSON: {success, data: {markdown, model_name, ...}}

    返回的模型卡片包含:
        - YAML 元数据头
        - 模型描述与用途
        - 训练过程与超参数
        - 评估指标
        - 使用方法 (Python/curl/Docker)
        - 局限性声明
        - BibTeX 引用
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
        return jsonify({
            'success': True,
            'data': {
                'model_name': model.name,
                'model_uuid': model.uuid,
                'model_version': model.version,
                'framework': model.framework,
                'task_type': model.model_type,
                'markdown': markdown,
            }
        })

    # 默认返回 Markdown
    filename = f'{model.name_slug}_model_card.md'
    return Response(
        markdown,
        mimetype='text/markdown; charset=utf-8',
        headers={
            'Content-Disposition': f'inline; filename="{filename}"',
        }
    )


# ============ 直接模型服务端点 (镜像 Docker 容器 API) ============

@models_api_bp.route('/<string:model_uuid>/serve', methods=['POST'])
@api_login_required
def serve_model(model_uuid):
    """POST /api/v1/models/<uuid>/serve — 直接模型推理端点

    镜像 Docker 部署容器中 serve.py 的 /predict 契约, 方便在部署前本地测试。

    请求体:
        {"features": [[1.0, 2.5, 3.0], [4.0, 5.5, 6.0]]}

    响应 200:
        {"predictions": ["class_a", "class_b"], "task_type": "classification"}

    错误响应:
        400 — 输入格式错误
        404 — 模型不存在
        503 — 模型文件未加载
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
        return jsonify({
            'success': False,
            'message': '请求体必须是 JSON 对象 (dict)。',
            'code': 400,
        }), 400
    features = data.get('features')

    if features is None:
        return jsonify({
            'success': False,
            'message': '缺少 features 字段。请提供 {"features": [[...], ...]}',
            'code': 400,
        }), 400

    if not isinstance(features, list) or len(features) == 0:
        return jsonify({
            'success': False,
            'message': 'features 必须是非空数组。',
            'code': 400,
        }), 400

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
            return jsonify({
                'success': False,
                'message': 'features 每项必须是数组 [v1, v2, ...] 或字典 {col: val}。',
                'code': 400,
            }), 400
    except Exception as e:
        return jsonify({'success': False, 'message': f'features 解析失败: {str(e)}', 'code': 400}), 400

    # 执行预测
    try:
        from app.services.inference_service import ModelInferenceService
        result = ModelInferenceService.predict(model, df)
    except Exception as e:
        from app.utils.helpers import sanitize_service_error
        return jsonify({
            'success': False,
            'message': sanitize_service_error(e, '模型推理异常'),
            'code': 500,
        }), 500

    if not result.get('success'):
        error_msg = result.get('error', '模型预测失败')
        # 判断是否为模型加载错误 (503) vs 数据错误 (400)
        if '模型文件不存在' in error_msg or '无法加载' in error_msg or '加载失败' in error_msg or '不支持' in error_msg:
            return jsonify({'success': False, 'message': error_msg, 'code': 503}), 503
        return jsonify({'success': False, 'message': error_msg, 'code': 400}), 400

    return jsonify({
        'predictions': result.get('predictions', []),
        'task_type': result.get('task_type', model.model_type),
    })


# ============ 部署健康检查 ============

# SSRF 防护: 内部/私有 IP 段黑名单
_SSRF_BLOCKED_NETWORKS = [
    ipaddress.ip_network('127.0.0.0/8'),        # loopback
    ipaddress.ip_network('10.0.0.0/8'),         # private A
    ipaddress.ip_network('172.16.0.0/12'),      # private B
    ipaddress.ip_network('192.168.0.0/16'),     # private C
    ipaddress.ip_network('169.254.0.0/16'),     # link-local / AWS metadata
    ipaddress.ip_network('0.0.0.0/8'),          # current network
    ipaddress.ip_network('100.64.0.0/10'),      # CGNAT
    ipaddress.ip_network('198.18.0.0/15'),      # benchmark
    ipaddress.ip_network('224.0.0.0/4'),        # multicast
    ipaddress.ip_network('240.0.0.0/4'),        # reserved
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
        for net in _SSRF_BLOCKED_NETWORKS:
            if addr in net:
                return False
        return True
    except ValueError:
        # hostname 不是有效IP → 域名, 允许 (部署到公网)
        return True


@models_api_bp.route('/<string:model_uuid>/deploy/health', methods=['GET'])
@api_login_required
def deploy_health(model_uuid):
    """GET /api/v1/models/<uuid>/deploy/health — 检查 Docker 部署健康状态

    检查逻辑:
        1. 验证部署包是否存在 (experiments/exports/<uuid>/deploy/)
        2. 尝试 GET <deployment_url>/health 或 http://localhost:8000/health
        3. 返回容器健康状态和元信息

    返回:
        {success, data: {status, deploy_exists, container_info, checked_at}}

    status 取值:
        - "healthy"      容器响应正常
        - "unreachable"  部署包存在但容器无响应
        - "not_deployed" 尚未生成部署包
    """
    import urllib.request
    import urllib.error
    import json as _json

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
            return jsonify({
                'success': False,
                'message': '部署URL不安全: 仅允许公网域名或 localhost。',
                'code': 400,
            }), 400
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

    from datetime import datetime, timezone
    return jsonify({
        'success': True,
        'data': {
            'status': status,
            'deploy_exists': deploy_exists,
            'container_info': container_info,
            'checked_at': datetime.now(timezone.utc).isoformat(),
        }
    })


@models_api_bp.route('/<string:model_uuid>/export/status', methods=['GET'])
@api_login_required
def export_status(model_uuid):
    """GET /api/models/<uuid>/export/status?task_id=xxx — 查询导出任务进度

    返回:
        {success, data: {task_id, status, progress, message, result, error}}
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
