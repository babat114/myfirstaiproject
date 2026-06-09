"""
============================================
模型推理服务
加载已训练模型并对新数据进行预测
支持 sklearn 和 PyTorch 模型
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

    支持两个主要框架:
        - sklearn: pickle/joblib 格式, predict + predict_proba
        - PyTorch: .pt 权重 + _config.pkl 配置, torch.softmax 输出概率

    三大功能:
        1. predict()        — 对新数据执行预测 (支持 CSV/JSON/手动输入)
        2. test_model_with_split() — 使用原始数据集完整评估 (混淆矩阵/分类报告/回归指标)
        3. get_feature_importance() — 特征重要性分析 (树模型/线性模型)
    """

    @staticmethod
    def load_model(model: ModelRecord) -> Tuple[Optional[object], Optional[dict], Optional[str]]:
        """
        加载模型文件及其元数据

        Returns:
            (model_obj, metadata, error_message)
            metadata 包含 scaler, label_encoders, feature_names 等
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
            return None, None, '模型文件不存在。请先上传模型文件或完成训练。'

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
                    }
                else:
                    model_obj = bundle
                    metadata = {'task_type': model.model_type}

                logger.info(f"sklearn 模型已加载: {model.name} ({model_path})")
                return model_obj, metadata, None

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
                    return model_obj, metadata, None
                except ImportError:
                    return None, metadata, 'PyTorch 未安装，无法加载模型。请运行: pip install torch'
                except Exception as e:
                    logger.error(f"加载 PyTorch 模型失败: {e}")
                    return None, metadata, f'PyTorch 模型加载失败: {str(e)}'

            else:
                return None, None, f'不支持的模型格式: {ext}'

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None, None, f'模型加载失败: {str(e)}'

    @staticmethod
    def predict(model: ModelRecord, data: pd.DataFrame) -> dict:
        """
        对输入数据进行预测

        Args:
            model: ModelRecord 实例
            data: 输入特征 DataFrame

        Returns:
            {
                'success': bool,
                'predictions': [...],
                'probabilities': [...] (分类任务),
                'task_type': str,
                'error': str or None,
            }
        """
        model_obj, metadata, error = ModelInferenceService.load_model(model)
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
                if missing:
                    return {'success': False, 'error': f'缺少特征列: {missing}',
                            'predictions': [], 'probabilities': []}
                if extra:
                    logger.warning(f"多余的特征列将被忽略: {extra}")
                data = data[feature_names]

            # 处理缺失值
            data = data.fillna(data.mean(numeric_only=True))
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
                    data[num_cols] = scaler.transform(data[num_cols])

            # 预测 - 根据框架类型选择不同路径
            framework = (metadata or {}).get('framework', 'sklearn')
            X = data.values.astype('float32')

            if framework in ('pytorch', 'tensorflow'):
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
            else:
                predictions = model_obj.predict(X)
                proba = None
                if task_type in ('classification',) and hasattr(model_obj, 'predict_proba'):
                    try:
                        proba = model_obj.predict_proba(X)
                    except Exception:
                        pass

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

            # 提取概率 (分类任务)
            probabilities = None
            if task_type in ('classification',) and proba is not None:
                try:
                    top_indices = np.argsort(-proba, axis=1)[:, :3]
                    probabilities = []
                    for i in range(len(X)):
                        probs_list = []
                        for j, idx in enumerate(top_indices[i]):
                            probs_list.append({
                                'class': str(idx),
                                'probability': round(float(proba[i][idx]), 4),
                            })
                        probabilities.append(probs_list)
                except Exception:
                    pass

            # 如果目标是标签编码的，尝试反向解码
            target_le = label_encoders.get('__target__') if label_encoders else None
            if target_le is not None and task_type == 'classification':
                try:
                    pred_list = [target_le.inverse_transform([int(p) if isinstance(p, (int, float)) else p])[0]
                                 for p in predictions]
                    pred_list = [str(p) for p in pred_list]
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
    def test_model_with_split(model: ModelRecord) -> dict:
        """
        使用训练时的测试集评估模型 (需要模型文件 + 原始数据集)

        返回完整的评估报告，包含混淆矩阵数据等
        """
        import pandas as pd
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report,
            mean_squared_error, mean_absolute_error, r2_score
        )

        model_obj, metadata, error = ModelInferenceService.load_model(model)
        if error:
            return {'success': False, 'error': error}

        # 需要原始数据集
        train_dataset = model.training_dataset
        if not train_dataset or not os.path.exists(train_dataset.file_path):
            return {'success': False, 'error': '原始训练数据集不存在，无法评估。'}

        try:
            # 加载数据集
            from app.utils.data_io import load_dataframe
            df = load_dataframe(train_dataset.file_path, train_dataset.file_format.lower())
            if df is None:
                return {'success': False, 'error': f'不支持的数据格式或文件已损坏'}

            # 获取超参数中的 target_column
            hyperparams = model.hyperparameters_dict
            target_col = hyperparams.get('target_column')
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
            report = {'success': True, 'task_type': task_type, 'num_samples': len(y_pred)}

            if task_type == 'classification':
                # 确保标签一致
                from sklearn.preprocessing import LabelEncoder
                if y_true.dtype == 'object':
                    le = LabelEncoder()
                    y_true_enc = le.fit_transform(y_true.astype(str))
                else:
                    y_true_enc = y_true.values

                try:
                    y_pred_enc = [int(p) for p in y_pred]
                except (ValueError, TypeError):
                    y_pred_enc = y_pred

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
        获取特征重要性 (仅支持树模型和线性模型)
        """
        model_obj, metadata, error = ModelInferenceService.load_model(model)
        if error:
            return {'success': False, 'error': error}

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
