"""
============================================
模型导出与部署服务
支持 ONNX 格式转换、Docker 部署配置生成
============================================
"""
import os
import pickle
import json
import shutil
import tempfile
from datetime import datetime
from app._timezone import localnow
from typing import Optional, Tuple
from flask import current_app

from app import db, logger
from app.models.model_record import ModelRecord


class ModelExportService:
    """模型导出服务 — ONNX 转换 / Docker 部署文件生成 / 模型下载

    支持框架:
        - sklearn → ONNX (skl2onnx)
        - PyTorch → ONNX (torch.onnx)
        - TensorFlow → ONNX (tf2onnx)

    导出产物:
        - ONNX 模型文件 (.onnx)
        - 推理脚本 (serve.py)
        - Dockerfile + docker-compose.yml
        - 环境依赖文件 (requirements.txt)
    """

    # 推理服务模板 — FastAPI 微服务
    INFERENCE_SERVER_TEMPLATE = '''"""
Auto-generated model serving script
Model: {model_name}
Framework: {framework}
Task: {task_type}
Generated: {timestamp}
"""
import numpy as np
import pickle
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="{model_name}", version="1.0.0")

# ── 加载模型 ─────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "./model")
MODEL = None
SCALER = None
LABEL_ENCODERS = {{}}
FEATURE_NAMES = []
TASK_TYPE = "{task_type}"
FRAMEWORK = "{framework}"

def load_model():
    global MODEL, SCALER, LABEL_ENCODERS, FEATURE_NAMES
    if FRAMEWORK == "onnx":
        import onnxruntime as ort
        MODEL = ort.InferenceSession(f"{{MODEL_PATH}}.onnx")
    elif FRAMEWORK in ("pytorch", "tensorflow"):
        import torch
        from app.executor.trainers.pytorch_trainer import load_mlp_model
        MODEL, config = load_mlp_model(
            f"{{MODEL_PATH}}.pt", f"{{MODEL_PATH}}_config.pkl"
        )
        SCALER = config.get("scaler")
        LABEL_ENCODERS = config.get("label_encoders", {{}})
        FEATURE_NAMES = config.get("feature_names", [])
    else:
        with open(f"{{MODEL_PATH}}.pkl", "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict):
            MODEL = bundle.get("model")
            SCALER = bundle.get("scaler")
            LABEL_ENCODERS = bundle.get("label_encoders", {{}})
            FEATURE_NAMES = bundle.get("feature_names", [])
        else:
            MODEL = bundle


class PredictRequest(BaseModel):
    features: list[list[float]]

class PredictResponse(BaseModel):
    predictions: list
    task_type: str


@app.on_event("startup")
async def startup():
    load_model()
    print(f"✓ Model loaded: {model_name}")


@app.get("/health")
async def health():
    return {{"status": "healthy", "model": "{model_name}"}}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")

    X = np.array(req.features, dtype='float32')

    if SCALER is not None:
        X = SCALER.transform(X)

    if FRAMEWORK == "onnx":
        input_name = MODEL.get_inputs()[0].name
        outputs = MODEL.run(None, {{input_name: X}})[0]
    elif FRAMEWORK in ("pytorch", "tensorflow"):
        import torch
        MODEL.eval()
        with torch.no_grad():
            outputs = MODEL(torch.tensor(X, dtype=torch.float32)).numpy()
    else:
        outputs = MODEL.predict(X)

    if TASK_TYPE == "classification":
        preds = outputs.argmax(axis=1).tolist() if outputs.ndim > 1 else outputs.tolist()
    else:
        preds = outputs.ravel().tolist()

    return PredictResponse(predictions=preds, task_type=TASK_TYPE)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
'''

    DOCKERFILE_TEMPLATE = '''# Auto-generated Dockerfile for {model_name}
# Framework: {framework} | Task: {task_type}

FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制模型文件和推理脚本
COPY model.pkl model.pt model.keras model.onnx serve.py ./
COPY *_config.pkl ./

ENV PORT=8000
ENV MODEL_PATH=./model

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "serve.py"]
'''

    DOCKER_COMPOSE_TEMPLATE = '''version: "3.8"
services:
  {service_name}:
    build: .
    container_name: {container_name}
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - MODEL_PATH=./model
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 10s
'''

    @staticmethod
    def export_onnx(model: ModelRecord) -> Tuple[bool, str, Optional[str]]:
        """将模型导出为 ONNX 格式

        Returns:
            (success, message, onnx_file_path)
        """
        from app.services.inference_service import ModelInferenceService

        model_obj, metadata, error = ModelInferenceService.load_model(model)
        if error:
            return False, f'无法加载模型: {error}', None

        if not metadata:
            metadata = {}

        framework = metadata.get('framework', 'sklearn')
        task_type = metadata.get('task_type', model.model_type)
        feature_count = metadata.get('input_dim', 10)

        export_dir = os.path.join('experiments', 'exports', model.uuid)
        os.makedirs(export_dir, exist_ok=True)
        onnx_path = os.path.join(export_dir, f'{model.name_slug}.onnx')

        try:
            if framework == 'sklearn':
                # sklearn → ONNX (skl2onnx)
                return ModelExportService._export_sklearn_onnx(
                    model_obj, metadata, feature_count, onnx_path
                )
            elif framework == 'pytorch':
                # PyTorch → ONNX
                return ModelExportService._export_pytorch_onnx(
                    model_obj, metadata, feature_count, onnx_path
                )
            elif framework in ('tensorflow', 'keras', 'tf'):
                # TensorFlow → ONNX (tf2onnx)
                return ModelExportService._export_tf_onnx(
                    model, metadata, feature_count, onnx_path
                )
            else:
                return False, f'框架 "{framework}" 暂不支持 ONNX 导出', None

        except ImportError as e:
            return False, f'缺少依赖: {str(e)}。请运行: pip install skl2onnx onnx', None
        except Exception as e:
            logger.error(f"ONNX 导出失败: {e}", exc_info=True)
            return False, f'导出失败: {str(e)}', None

    @staticmethod
    def _export_sklearn_onnx(model_obj, metadata, feature_count, onnx_path):
        """sklearn → ONNX"""
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import numpy as np

        initial_type = [('float_input', FloatTensorType([None, feature_count]))]

        try:
            onx = convert_sklearn(model_obj, initial_types=initial_type)
        except Exception:
            # 如果 convert_sklearn 不支持，尝试使用通用的 to_onnx
            from skl2onnx import to_onnx
            onx = to_onnx(model_obj, initial_types=initial_type)

        with open(onnx_path, 'wb') as f:
            f.write(onx.SerializeToString())

        logger.info(f"sklearn 模型已导出为 ONNX: {onnx_path}")
        return True, f'ONNX 导出成功: {os.path.basename(onnx_path)}', onnx_path

    @staticmethod
    def _export_pytorch_onnx(model_obj, metadata, feature_count, onnx_path):
        """PyTorch → ONNX"""
        import torch

        model_obj.eval()
        model_obj = model_obj.cpu()

        dummy_input = torch.randn(1, feature_count)

        torch.onnx.export(
            model_obj,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['float_input'],
            output_names=['output'],
            dynamic_axes={
                'float_input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            }
        )

        logger.info(f"PyTorch 模型已导出为 ONNX: {onnx_path}")
        return True, f'ONNX 导出成功: {os.path.basename(onnx_path)}', onnx_path

    @staticmethod
    def _export_tf_onnx(model, metadata, feature_count, onnx_path):
        """TensorFlow/Keras → ONNX (tf2onnx)"""
        try:
            import tf2onnx
            import tensorflow as tf

            # 加载 Keras 模型
            if hasattr(model, 'model_file_path'):
                keras_path = model.model_file_path
            else:
                # 从 experiments 目录查找
                from app.services.inference_service import ModelInferenceService
                model_obj, _, _ = ModelInferenceService.load_model(model)
                # 重新从 .keras 文件导出
                keras_path = (model.model_file_path or '').replace('.keras', '') + '.keras'
                if not os.path.exists(keras_path):
                    return False, 'Keras 模型文件不存在', None

            loaded_model = tf.keras.models.load_model(keras_path)
            spec = (tf.TensorSpec((None, feature_count), tf.float32, name='float_input'),)
            output_path = onnx_path

            tf2onnx.convert.from_keras(loaded_model, input_signature=spec, output_path=output_path)

            logger.info(f"TensorFlow 模型已导出为 ONNX: {onnx_path}")
            return True, f'ONNX 导出成功: {os.path.basename(onnx_path)}', onnx_path
        except ImportError:
            return False, '缺少依赖: pip install tf2onnx', None

    @staticmethod
    def generate_deployment_package(model: ModelRecord) -> Tuple[bool, str, Optional[str]]:
        """生成完整部署包 (Dockerfile + serve.py + requirements.txt + docker-compose.yml)

        将所有文件打包到一个目录，方便一键部署。

        Returns:
            (success, message, package_dir_path)
        """
        from app.services.inference_service import ModelInferenceService

        model_obj, metadata, error = ModelInferenceService.load_model(model)
        if error:
            return False, f'无法加载模型: {error}', None

        task_type = (metadata or {}).get('task_type', model.model_type)
        framework = (metadata or {}).get('framework', 'sklearn')

        # 导出目录
        package_dir = os.path.join('experiments', 'exports', model.uuid, 'deploy')
        os.makedirs(package_dir, exist_ok=True)

        # 1. 推理脚本
        serve_py = ModelExportService.INFERENCE_SERVER_TEMPLATE.format(
            model_name=model.name,
            framework=framework,
            task_type=task_type,
            timestamp=localnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
        )
        with open(os.path.join(package_dir, 'serve.py'), 'w', encoding='utf-8') as f:
            f.write(serve_py)

        # 2. 复制模型文件
        model_path = model.model_file_path
        if model_path and os.path.exists(model_path):
            ext = os.path.splitext(model_path)[1]
            shutil.copy2(model_path, os.path.join(package_dir, f'model{ext}'))
            # 复制配置文件
            config_path = model_path + '_config.pkl'
            if os.path.exists(config_path):
                shutil.copy2(config_path, os.path.join(package_dir, f'model{ext}_config.pkl'))
            # 复制 PyTorch .pt
            pt_path = model_path.replace('.pkl', '.pt')
            if os.path.exists(pt_path) and pt_path != model_path:
                shutil.copy2(pt_path, os.path.join(package_dir, 'model.pt'))
            # 复制 Keras .keras
            keras_path = model_path.replace('.pkl', '.keras')
            if os.path.exists(keras_path) and keras_path != model_path:
                shutil.copy2(keras_path, os.path.join(package_dir, 'model.keras'))

        # 3. requirements.txt (根据框架生成)
        if framework in ('pytorch', 'tensorflow'):
            reqs = 'fastapi>=0.100.0\nuvicorn>=0.23.0\nnumpy>=1.24.0\n'
            if framework == 'pytorch':
                reqs += 'torch>=2.0.0\n'
            else:
                reqs += 'tensorflow>=2.12.0\n'
            reqs += 'scikit-learn>=1.3.0\npython-multipart>=0.0.6\n'
        else:
            # sklearn / ONNX / Other
            reqs = 'fastapi>=0.100.0\nuvicorn>=0.23.0\nnumpy>=1.24.0\n'
            reqs += 'scikit-learn>=1.3.0\npython-multipart>=0.0.6\n'
            if framework == 'onnx':
                reqs += 'onnxruntime>=1.15.0\n'
        with open(os.path.join(package_dir, 'requirements.txt'), 'w', encoding='utf-8') as f:
            f.write(reqs)

        # 4. Dockerfile
        dockerfile = ModelExportService.DOCKERFILE_TEMPLATE.format(
            model_name=model.name,
            framework=framework,
            task_type=task_type,
        )
        with open(os.path.join(package_dir, 'Dockerfile'), 'w', encoding='utf-8') as f:
            f.write(dockerfile)

        # 5. docker-compose.yml
        safe_name = model.name.lower().replace(' ', '-').replace('_', '-')
        compose = ModelExportService.DOCKER_COMPOSE_TEMPLATE.format(
            service_name=safe_name,
            container_name=f'{safe_name}-serve',
        )
        with open(os.path.join(package_dir, 'docker-compose.yml'), 'w', encoding='utf-8') as f:
            f.write(compose)

        logger.info(f"部署包已生成: {package_dir}")
        return True, f'部署包已生成 (5个文件)', package_dir

    @staticmethod
    def get_export_info(model: ModelRecord) -> dict:
        """获取模型的导出信息和状态"""
        export_dir = os.path.join('experiments', 'exports', model.uuid)
        info = {
            'onnx_available': False,
            'deploy_available': False,
            'onnx_path': None,
            'deploy_dir': None,
            'framework': model.framework or 'sklearn',
            'supported_formats': [],
        }

        # 检查 ONNX 导出
        if os.path.exists(export_dir):
            onnx_files = [f for f in os.listdir(export_dir)
                         if f.endswith('.onnx')]
            if onnx_files:
                info['onnx_available'] = True
                info['onnx_path'] = os.path.join(export_dir, onnx_files[0])

            deploy_dir = os.path.join(export_dir, 'deploy')
            if os.path.exists(deploy_dir):
                info['deploy_available'] = True
                info['deploy_dir'] = deploy_dir

        # 支持导出的格式
        framework = info['framework']
        if framework in ('sklearn', 'pytorch'):
            info['supported_formats'].append('onnx')
        if framework in ('tensorflow', 'keras', 'tf'):
            info['supported_formats'].append('onnx-tf')
        info['supported_formats'].append('docker')

        return info
