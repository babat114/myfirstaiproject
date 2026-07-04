"""
============================================
模型推理服务
加载已训练模型并对新数据进行预测
支持 sklearn / PyTorch / HuggingFace Transformers 模型
============================================
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from flask import current_app
from app import db, logger
from app.models.model_record import ModelRecord


class ModelInferenceService:
    """模型推理服务 — 加载已训练的模型文件并执行预测/评估/特征分析

    支持三个主要框架:
        - sklearn: pickle/joblib 格式, predict + predict_proba
        - PyTorch: .pt 权重 + _config.pkl 配置, torch.softmax 输出概率
        - Transformers: HuggingFace _nlp_model/ 目录, AutoModel + AutoTokenizer

    四大功能:
        1. predict()        — 对新数据执行预测 (支持 CSV/JSON/手动输入/文本)
        2. predict_single() — 单文本快速预测 (NLP transformer 模型)
        3. test_model_with_split() — 使用原始数据集完整评估 (混淆矩阵/分类报告/回归指标)
        4. get_feature_importance() — 特征重要性分析 (树模型/线性模型)
    """

    @staticmethod
    def load_model(model: ModelRecord) -> Tuple[Optional[object], Optional[dict], Optional[object], Optional[str]]:
        """
        加载模型文件及其元数据

        Returns:
            (model_obj, metadata, tokenizer, error_message)
            - model_obj: 模型对象 (sklearn/PyTorch/Transformers)
            - metadata: 包含 scaler, label_encoders, feature_names 等
            - tokenizer: HuggingFace tokenizer (仅 NLP 模型, 其他为 None)
            - error_message: 错误信息字符串 (成功时为 None)
        """
        model_path = model.model_file_path
        if not model_path or not os.path.exists(model_path):
            # 尝试从 experiments 目录加载
            training_job = model.training_job
            if training_job:
                exp_path = os.path.join('experiments', training_job.uuid, 'model.pkl')
                if os.path.exists(exp_path):
                    model_path = exp_path
                else:
                    exp_path_pt = os.path.join('experiments', training_job.uuid, 'model.pt')
                    if os.path.exists(exp_path_pt):
                        model_path = exp_path_pt

        if not model_path or not os.path.exists(model_path):
            # HuggingFace NLP 模型检测: 查找 _nlp_model/ 目录
            nlp_dir = None
            # 1) 在 model_path 同级查找
            if model_path:
                base = os.path.splitext(model_path)[0]
                candidate = base + '_nlp_model'
                if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, 'config.json')):
                    nlp_dir = candidate
            # 2) 在 experiment 目录扫描
            if not nlp_dir and model.training_job:
                exp_dir = os.path.join('experiments', model.training_job.uuid)
                if os.path.isdir(exp_dir):
                    for entry in os.listdir(exp_dir):
                        if entry.endswith('_nlp_model'):
                            candidate = os.path.join(exp_dir, entry)
                            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, 'config.json')):
                                nlp_dir = candidate
                                break
            # 3) 扫描整个 experiments 目录 (fallback for models whose training_job is unset)
            if not nlp_dir:
                for exp_root in ['experiments']:
                    if os.path.isdir(exp_root):
                        for exp_name in os.listdir(exp_root):
                            exp_path = os.path.join(exp_root, exp_name)
                            if os.path.isdir(exp_path):
                                for entry in os.listdir(exp_path):
                                    if entry.endswith('_nlp_model'):
                                        candidate = os.path.join(exp_path, entry)
                                        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, 'config.json')):
                                            nlp_dir = candidate
                                            break
                            if nlp_dir:
                                break
                    if nlp_dir:
                        break

            if nlp_dir:
                try:
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer
                    hf_model = AutoModelForSequenceClassification.from_pretrained(nlp_dir)
                    tokenizer = AutoTokenizer.from_pretrained(nlp_dir)
                    meta_path = os.path.join(nlp_dir, 'metadata.json')
                    metadata = {}
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    metadata['framework'] = 'transformers'
                    metadata['task_type'] = metadata.get('task_type', 'classification')
                    metadata['algorithm'] = metadata.get('model_name', 'transformer')
                    logger.info(f"HuggingFace 模型已加载: {model.name} ({nlp_dir})")
                    return hf_model, metadata, tokenizer, None
                except ImportError:
                    return None, None, None, (
                        'HuggingFace transformers 未安装，无法加载 NLP 模型。'
                        '请运行: pip install transformers'
                    )
                except Exception as e:
                    logger.error(f"加载 HuggingFace 模型失败: {e}")
                    return None, None, None, f'HuggingFace NLP 模型加载失败: {str(e)}'

            return None, None, None, '模型文件不存在。请先上传模型文件或完成训练。'

        ext = os.path.splitext(model_path)[1].lower()

        try:
            if ext == '.pkl' or ext == '.joblib':
                # sklearn 模型
                with open(model_path, 'rb') as f:
                    bundle = pickle.load(f)
                if isinstance(bundle, dict):
                    model_obj = bundle.get('model')
                    metadata = {
                        'scaler': bundle.get('scaler'),
                        'label_encoders': bundle.get('label_encoders', {}),
                        'feature_names': bundle.get('feature_names', []),
                        'task_type': bundle.get('task_type', model.model_type),
                        'algorithm': bundle.get('algorithm', ''),
                        'vectorizer': bundle.get('vectorizer'),  # NLP TF-IDF
                        'class_labels': bundle.get('class_labels', []),  # 可读标签
                    }
                else:
                    model_obj = bundle
                    metadata = {'task_type': model.model_type}

                logger.info(f"sklearn 模型已加载: {model.name} ({model_path})")
                return model_obj, metadata, None, None

            elif ext == '.pt' or ext == '.pth':
                # PyTorch 模型 — 加载配置和权重
                config_path = model_path.replace('.pt', '_config.pkl').replace('.pth', '_config.pkl')
                metadata = {}
                if os.path.exists(config_path):
                    with open(config_path, 'rb') as f:
                        metadata = pickle.load(f)
                try:
                    from app.executor.trainers.pytorch_trainer import load_mlp_model
                    model_obj, metadata_pt = load_mlp_model(model_path, config_path)
                    metadata.update(metadata_pt)
                    metadata['framework'] = 'pytorch'
                    logger.info(f"PyTorch 模型已加载: {model.name} ({model_path})")
                    return model_obj, metadata, None, None
                except ImportError:
                    return None, metadata, None, 'PyTorch 未安装，无法加载模型。请运行: pip install torch'
                except Exception as e:
                    logger.error(f"加载 PyTorch 模型失败: {e}")
                    return None, metadata, None, f'PyTorch 模型加载失败: {str(e)}'

            elif ext == '.keras' or ext == '.h5':
                # TensorFlow/Keras 模型 — SavedModel 或 HDF5 格式
                try:
                    import tensorflow as tf
                except ImportError:
                    return None, metadata, None, 'TensorFlow 未安装，无法加载 .keras 模型。请运行: pip install tensorflow'
                model_obj_tf = tf.keras.models.load_model(model_path)

                # Load config from model_config.pkl (same dir as model file)
                config_path = os.path.join(os.path.dirname(model_path), 'model_config.pkl')
                metadata = {}
                if os.path.exists(config_path):
                    with open(config_path, 'rb') as f:
                        metadata = pickle.load(f)
                # Normalize: keras_trainer 存 'TensorFlow', 统一为 'tensorflow'
                metadata['framework'] = 'tensorflow'
                if 'scaler' not in metadata:
                    metadata['scaler'] = None
                if 'label_encoders' not in metadata:
                    metadata['label_encoders'] = {}
                if 'feature_names' not in metadata:
                    metadata['feature_names'] = []
                if 'task_type' not in metadata:
                    metadata['task_type'] = model.model_type

                logger.info(f"TensorFlow 模型已加载: {model.name} ({model_path})")
                return model_obj_tf, metadata, None, None

            else:
                return None, None, None, f'不支持的模型格式: {ext}'

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None, None, None, f'模型加载失败: {str(e)}'

    @staticmethod
    def predict(model: ModelRecord, data: pd.DataFrame) -> dict:
        """
        对输入数据进行预测

        Args:
            model: ModelRecord 实例
            data: 输入特征 DataFrame (表格数据或包含文本列的 NLP 数据)

        Returns:
            {
                'success': bool,
                'predictions': [...],
                'probabilities': [...] (分类任务),
                'task_type': str,
                'error': str or None,
            }
        """
        model_obj, metadata, tokenizer, error = ModelInferenceService.load_model(model)
        if error:
            return {'success': False, 'error': error, 'predictions': [], 'probabilities': []}

        task_type = metadata.get('task_type', model.model_type) if metadata else model.model_type

        try:
            # 预处理：对齐特征列
            feature_names = metadata.get('feature_names', []) if metadata else []
            if feature_names:
                # 确保数据包含所有训练时的特征列
                missing = [c for c in feature_names if c not in data.columns]
                extra = [c for c in data.columns if c not in feature_names]

                # 自动 TF-IDF 向量化: 如果模型保存了 vectorizer 且输入为原始文本列
                vectorizer = metadata.get('vectorizer') if metadata else None
                if vectorizer is not None and missing:
                    non_tfidf_cols = [c for c in data.columns if not str(c).startswith('tfidf_')]
                    if non_tfidf_cols and len(non_tfidf_cols) == len(data.columns):
                        try:
                            text_data = data[non_tfidf_cols[0]].astype(str).fillna('')
                            X_vec = vectorizer.transform(text_data)
                            vec_names = [f'tfidf_{i}' for i in range(vectorizer.get_feature_names_out().shape[0])]
                            data = pd.DataFrame(X_vec.toarray(), columns=vec_names, index=data.index)
                            missing = [c for c in feature_names if c not in data.columns]
                        except Exception as e:
                            logger.warning(f"TF-IDF 向量化自动转换失败: {e}")

                # NLP TF-IDF path: 智能匹配 — 仅要求 tfidf_ 列存在, 其余列填0
                tfidf_cols = [c for c in feature_names if c.startswith('tfidf_')]
                if tfidf_cols and len(tfidf_cols) == len(data.columns):
                    # 确认 data 的列名确实是 tfidf_* (而非未命名的整数列)
                    data_col_strs = [str(c) for c in data.columns]
                    is_tfidf_input = any(c.startswith('tfidf_') for c in data_col_strs)
                    if is_tfidf_input:
                        # 纯TF-IDF输入 (如quick-predict): 自动补齐非TF-IDF列
                        for c in feature_names:
                            if c not in data.columns:
                                data[c] = 0.0
                        missing = []

                if missing:
                    return {'success': False, 'error': f'缺少特征列: {missing}',
                            'predictions': [], 'probabilities': []}
                if extra:
                    logger.warning(f"多余的特征列将被忽略: {extra}")
                data = data[feature_names]

            # 处理缺失值
            _means = data.mean(numeric_only=True)
            if len(_means) > 0:
                data = data.fillna(_means)
            else:
                data = data.fillna(0)
            for col in data.select_dtypes(include=['object']).columns:
                data[col] = data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 'unknown')

            # 编码分类特征
            label_encoders = metadata.get('label_encoders', {}) if metadata else {}
            for col, le in label_encoders.items():
                if col in data.columns and col != '__target__':
                    try:
                        data[col] = data[col].astype(str)
                        # 处理未知类别
                        known_classes = set(le.classes_)
                        unknown_vals = [x for x in data[col].unique() if x not in known_classes]
                        if unknown_vals:
                            logger.warning(
                                f'未知类别值被映射到默认值 "{le.classes_[0]}": '
                                f'{unknown_vals[:5]}... (列={col})'
                            )
                        data[col] = data[col].apply(
                            lambda x: x if x in known_classes else le.classes_[0]
                        )
                        data[col] = le.transform(data[col])
                    except Exception as e:
                        logger.warning(f"编码列 {col} 失败: {e}")

            # 标准化
            scaler = metadata.get('scaler') if metadata else None
            if scaler is not None:
                num_cols = data.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    expected = getattr(scaler, 'n_features_in_', len(num_cols))
                    if len(num_cols) != expected:
                        return {'success': False,
                                'error': f'Scaler维度不匹配: 模型期望{expected}个数值列, 输入有{len(num_cols)}个',
                                'predictions': [], 'probabilities': []}
                    data[num_cols] = scaler.transform(data[num_cols])

            # 预测 - 根据框架类型选择不同路径
            framework = (metadata or {}).get('framework', 'sklearn')

            # ── Transformers NLP 路径 ──
            if framework == 'transformers' and tokenizer is not None:
                import torch

                # 从 DataFrame 提取文本列
                text_column = metadata.get('text_column', 'text')
                if text_column in data.columns:
                    texts = data[text_column].astype(str).tolist()
                else:
                    str_cols = data.select_dtypes(include=['object']).columns
                    if len(str_cols) > 0:
                        texts = data[str_cols[0]].astype(str).tolist()
                    else:
                        return {
                            'success': False,
                            'error': 'NLP 模型需要文本输入，但数据中未找到文本列。',
                            'predictions': [], 'probabilities': [],
                        }

                max_length = metadata.get('max_length', 256)
                inputs = tokenizer(
                    texts,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt',
                )

                model_obj.eval()
                with torch.no_grad():
                    outputs = model_obj(**inputs)
                    logits = outputs.logits
                    if task_type == 'classification':
                        probs = torch.softmax(logits, dim=1)
                        _, pred_indices = torch.max(logits, 1)
                        predictions = pred_indices.cpu().tolist()
                        proba = probs.cpu().numpy()
                    else:
                        raw_preds = logits.cpu().numpy()
                        predictions = raw_preds.ravel().tolist()
                        proba = None

                # id2label 映射
                id2label = metadata.get('id2label', {})
                if id2label:
                    predictions = [
                        id2label.get(str(p), id2label.get(p, str(p)))
                        for p in predictions
                    ]

            elif framework == 'pytorch':
                X = data.values.astype('float32')
                import torch
                model_obj.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X)
                    outputs = model_obj(X_tensor)
                    if task_type == 'classification':
                        probs = torch.softmax(outputs, dim=1)
                        _, pred_indices = torch.max(outputs, 1)
                        predictions = pred_indices.cpu().tolist()
                        proba = probs.cpu().tolist()
                    else:
                        # 回归: 反标准化预测值到原始尺度
                        raw_preds = outputs.cpu().numpy().reshape(-1, 1)
                        y_scaler = (metadata or {}).get('y_scaler')
                        if y_scaler is not None:
                            raw_preds = y_scaler.inverse_transform(raw_preds)
                        predictions = raw_preds.ravel().tolist()
                        proba = None

            elif framework == 'tensorflow':
                # TensorFlow/Keras 模型预测
                import tensorflow as tf
                X = data.values.astype('float32')
                # Keras predict() — 分类模型末层含 softmax, 输出概率; 回归模型输出原始值
                outputs = model_obj.predict(X, verbose=0)
                if task_type == 'classification':
                    # outputs shape: (n_samples, n_classes) — 已是 softmax 概率
                    proba = outputs
                    predictions = np.argmax(outputs, axis=1).tolist()
                else:
                    # 回归: 反标准化预测值到原始尺度
                    raw_preds = outputs.reshape(-1, 1)
                    y_scaler = (metadata or {}).get('y_scaler')
                    if y_scaler is not None:
                        raw_preds = y_scaler.inverse_transform(raw_preds)
                    predictions = raw_preds.ravel().tolist()
                    proba = None
            else:
                X = data.values.astype('float32')
                predictions = model_obj.predict(X)
                proba = None
                if task_type in ('classification',) and hasattr(model_obj, 'predict_proba'):
                    try:
                        proba = model_obj.predict_proba(X)
                    except Exception as e:
                        logger.warning(f"predict_proba 失败 (仅返回预测类别): {e}")

            # 格式化预测结果
            pred_list = []
            for p in predictions:
                if isinstance(p, (np.integer,)):
                    pred_list.append(int(p))
                elif isinstance(p, (np.floating,)):
                    pred_list.append(round(float(p), 4))
                elif isinstance(p, np.ndarray):
                    pred_list.append(p.tolist())
                else:
                    pred_list.append(str(p))

            # 如果目标是标签编码的，尝试反向解码 (必须在概率提取之前)
            target_le = label_encoders.get('__target__') if label_encoders else None
            decoded_labels = None  # 用于概率展示
            if target_le is not None and task_type == 'classification' and framework != 'transformers':
                try:
                    # 兼容 numpy int64 等类型
                    decoded = []
                    for p in predictions:
                        if isinstance(p, (int, float, np.integer, np.floating)):
                            decoded.append(int(p))
                        else:
                            decoded.append(p)
                    pred_list = [str(target_le.inverse_transform([d])[0]) for d in decoded]
                    # 构建类别名列表供概率展示使用
                    decoded_labels = [str(c) for c in target_le.classes_]
                except Exception as e:
                    logger.warning(
                        f'Label inverse_transform failed (labels may show as numbers): {e}'
                    )

            # 提取概率 (分类任务)
            probabilities = None
            if task_type in ('classification',) and proba is not None:
                try:
                    top_indices = np.argsort(-proba, axis=1)[:, :3]
                    probabilities = []
                    for i in range(len(predictions)):
                        probs_list = []
                        for j, idx in enumerate(top_indices[i]):
                            # 优先使用解码后的标签名, 回退到索引
                            class_label = (
                                decoded_labels[int(idx)]
                                if decoded_labels is not None and int(idx) < len(decoded_labels)
                                else str(idx)
                            )
                            probs_list.append({
                                'class': class_label,
                                'probability': round(float(proba[i][idx]), 4),
                            })
                        probabilities.append(probs_list)
                except Exception:
                    pass

            return {
                'success': True,
                'predictions': pred_list,
                'probabilities': probabilities,
                'task_type': task_type,
                'num_samples': len(pred_list),
                'error': None,
            }

        except Exception as e:
            logger.error(f"预测失败: {e}", exc_info=True)
            return {'success': False, 'error': f'预测失败: {str(e)}',
                    'predictions': [], 'probabilities': []}

    @staticmethod
    def predict_single(
        model_obj,
        tokenizer,
        metadata: dict,
        text: str,
    ) -> Optional[dict]:
        """单文本快速预测 — 专用于 Transformer NLP 模型

        在 quick-predict API 中替代不存在 TransformersNLPTrainer.predict_single(),
        直接使用已加载的 model_obj + tokenizer 进行推理。

        Args:
            model_obj: HuggingFace AutoModelForSequenceClassification 实例
            tokenizer: HuggingFace AutoTokenizer 实例
            metadata: 模型元数据 (id2label, max_length 等)
            text: 输入文本字符串

        Returns:
            {'label': str, 'confidence': float, 'probabilities': [...]} 或 None
        """
        import torch

        if not text or not text.strip():
            return None

        try:
            max_length = metadata.get('max_length', 256)
            inputs = tokenizer(
                text.strip(),
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt',
            )

            model_obj.eval()
            with torch.no_grad():
                outputs = model_obj(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).squeeze(0)

            top_probs, top_indices = torch.topk(probs, min(5, len(probs)))
            id2label = metadata.get('id2label', {})

            probabilities = []
            for i, idx_val in enumerate(top_indices.tolist()):
                label = id2label.get(str(idx_val), id2label.get(idx_val, str(idx_val)))
                probabilities.append({
                    'class': str(label),
                    'probability': round(float(top_probs[i]), 4),
                })

            best_label = probabilities[0]['class'] if probabilities else 'unknown'
            best_conf = probabilities[0]['probability'] if probabilities else 0.0

            return {
                'label': best_label,
                'confidence': best_conf,
                'probabilities': probabilities,
            }

        except Exception as e:
            logger.error(f"predict_single 失败: {e}", exc_info=True)
            return None

    @staticmethod
    def test_model_with_split(model: ModelRecord, test_dataset=None) -> dict:
        """
        评估模型 — 支持独立测试集

        Args:
            model: 已训练的 ModelRecord
            test_dataset: 可选, 独立测试集 Dataset 对象。
                         若提供则使用独立测试集评估;
                         若未提供则使用训练数据集 (向后兼容)

        返回完整的评估报告，包含混淆矩阵数据等
        """
        import pandas as pd
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report,
            mean_squared_error, mean_absolute_error, r2_score
        )
        from app.models.dataset import Dataset

        model_obj, metadata, tokenizer, error = ModelInferenceService.load_model(model)
        if error:
            return {'success': False, 'error': error}

        # 确定使用哪个数据集进行评估
        if test_dataset is not None:
            eval_dataset = test_dataset
            is_independent = True
        else:
            eval_dataset = model.training_dataset
            is_independent = False

        if not eval_dataset or not os.path.exists(eval_dataset.file_path):
            ds_type = '独立测试' if is_independent else '原始训练'
            return {'success': False, 'error': f'{ds_type}数据集不存在，无法评估。'}

        try:
            # 加载数据集
            from app.utils.data_io import load_dataframe
            df = load_dataframe(eval_dataset.file_path, eval_dataset.file_format.lower())
            if df is None:
                return {'success': False, 'error': f'不支持的数据格式或文件已损坏'}

            # 获取目标列: 优先用测试集的summary_json, 再回退到训练超参数
            hyperparams = model.hyperparameters_dict
            target_col = None

            # 尝试从测试集的 summary_json 中读取 target_column
            if is_independent and eval_dataset.summary_json:
                try:
                    import json as _json
                    summary = _json.loads(eval_dataset.summary_json)
                    target_col = summary.get('target_column')
                except Exception:
                    pass

            # 回退到训练超参数中的 target_column
            if not target_col:
                target_col = hyperparams.get('target_column')

            # 最后回退: 使用最后一列
            if not target_col:
                target_col = df.columns[-1]

            if target_col not in df.columns:
                return {'success': False, 'error': f'目标列 "{target_col}" 不存在'}

            y_true = df[target_col]
            X = df.drop(columns=[target_col])

            # 预处理并预测
            result = ModelInferenceService.predict(model, X)
            if not result['success']:
                return result

            y_pred = result['predictions']
            task_type = result.get('task_type', model.model_type)

            # 计算评估指标
            report = {
                'success': True,
                'task_type': task_type,
                'num_samples': len(y_pred),
                'is_independent_test': is_independent,
                'test_dataset_name': eval_dataset.name,
                'test_dataset_uuid': eval_dataset.uuid,
            }
            if is_independent:
                report['collection_method'] = getattr(eval_dataset, 'collection_method', None)

            if task_type == 'classification':
                # 确保 y_true 和 y_pred 在同一个编码空间中比较
                from sklearn.preprocessing import LabelEncoder
                import numpy as np
                if y_true.dtype == 'object' or isinstance(y_true.iloc[0], str):
                    le = LabelEncoder()
                    all_labels = np.unique(list(y_true.astype(str)) + [str(p) for p in y_pred])
                    le.fit(all_labels)
                    y_true_enc = le.transform(y_true.astype(str))
                    try:
                        y_pred_enc = le.transform([str(p) for p in y_pred])
                    except ValueError as e:
                        return {'success': False, 'error': f'预测标签包含训练集未见的类别: {e}'}
                else:
                    y_true_enc = y_true.values.astype('int64')
                    try:
                        y_pred_enc = [int(p) for p in y_pred]
                    except (ValueError, TypeError):
                        return {'success': False, 'error': '预测值包含非整数标签，但真实标签为整数'}

                report['accuracy'] = round(float(accuracy_score(y_true_enc, y_pred_enc)), 4)
                try:
                    # weighted 平均
                    report['precision_weighted'] = round(float(precision_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0)), 4)
                    report['recall_weighted'] = round(float(recall_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0)), 4)
                    report['f1_weighted'] = round(float(f1_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0)), 4)
                    # macro 平均 (各类别等权)
                    report['precision_macro'] = round(float(precision_score(y_true_enc, y_pred_enc, average='macro', zero_division=0)), 4)
                    report['recall_macro'] = round(float(recall_score(y_true_enc, y_pred_enc, average='macro', zero_division=0)), 4)
                    report['f1_macro'] = round(float(f1_score(y_true_enc, y_pred_enc, average='macro', zero_division=0)), 4)
                except Exception:
                    pass

                # 混淆矩阵
                try:
                    cm = confusion_matrix(y_true_enc, y_pred_enc)
                    report['confusion_matrix'] = cm.tolist()
                    labels = sorted(set(list(y_true_enc) + list(y_pred_enc)))
                    report['confusion_matrix_labels'] = [str(l) for l in labels]
                except Exception:
                    pass

                report['classification_report'] = classification_report(
                    y_true_enc, y_pred_enc, zero_division=0, output_dict=True
                )

            else:
                # 回归指标
                y_true_num = y_true.values.astype(float)
                y_pred_num = np.array(y_pred, dtype=float)

                report['mse'] = round(float(mean_squared_error(y_true_num, y_pred_num)), 4)
                report['mae'] = round(float(mean_absolute_error(y_true_num, y_pred_num)), 4)
                report['rmse'] = round(float(np.sqrt(mean_squared_error(y_true_num, y_pred_num))), 4)
                report['r2'] = round(float(r2_score(y_true_num, y_pred_num)), 4)

                # 残差数据 (用于绘图)
                residuals = (y_true_num - y_pred_num).tolist()
                report['residuals'] = residuals[:500]  # 最多 500 个点
                report['predictions_scatter'] = [
                    {'true': float(y_true_num[i]), 'pred': float(y_pred_num[i])}
                    for i in range(min(500, len(y_pred_num)))
                ]

            logger.info(f"模型测试完成: {model.name}, 准确率: {report.get('accuracy', report.get('r2', 'N/A'))}")
            return report

        except Exception as e:
            logger.error(f"模型测试失败: {e}", exc_info=True)
            return {'success': False, 'error': f'测试失败: {str(e)}'}

    @staticmethod
    def get_feature_importance(model: ModelRecord) -> dict:
        """
        获取特征重要性 (支持树模型/线性模型; Transformer NLP 模型返回友好提示)
        """
        model_obj, metadata, tokenizer, error = ModelInferenceService.load_model(model)
        if error:
            return {'success': False, 'error': error}

        # Transformer NLP 模型: 不支持传统特征重要性, 返回友好提示
        framework = (metadata or {}).get('framework', 'sklearn')
        if framework == 'transformers':
            return {
                'success': False,
                'error': (
                    '该模型为 Transformer NLP 模型，不支持传统特征重要性分析。'
                    '请使用预测结果概率分布评估模型表现，或通过注意力权重热力图进行解释。'
                ),
            }

        feature_names = (metadata or {}).get('feature_names', [])
        try:
            # 树模型
            if hasattr(model_obj, 'feature_importances_'):
                importances = model_obj.feature_importances_
                if len(feature_names) == 0:
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
                # 排序
                sorted_idx = np.argsort(importances)[::-1]
                return {
                    'success': True,
                    'features': [feature_names[i] for i in sorted_idx],
                    'importances': [round(float(importances[i]), 4) for i in sorted_idx],
                }

            # 线性模型
            if hasattr(model_obj, 'coef_'):
                coef = model_obj.coef_
                if coef.ndim > 1:
                    coef = coef[0]
                if len(feature_names) == 0:
                    feature_names = [f'feature_{i}' for i in range(len(coef))]
                sorted_idx = np.argsort(np.abs(coef))[::-1]
                return {
                    'success': True,
                    'features': [feature_names[i] for i in sorted_idx],
                    'importances': [round(float(coef[i]), 4) for i in sorted_idx],
                }

            return {'success': False, 'error': '该模型不支持特征重要性分析。'}
        except Exception as e:
            return {'success': False, 'error': f'分析失败: {str(e)}'}
