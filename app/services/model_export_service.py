"""
============================================
模型导出与部署服务
支持 ONNX 格式转换、Docker 部署配置生成
============================================
"""

import json
import os
import shutil

from app import logger
from app._timezone import localnow
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

    DOCKERFILE_TEMPLATE = """# Auto-generated Dockerfile for {model_name}
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

# 复制所有模型文件和推理脚本 (动态生成)
{copy_instructions}

ENV PORT=8000
ENV MODEL_PATH=./model

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "serve.py"]
"""

    DOCKER_COMPOSE_TEMPLATE = """version: "3.8"
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
"""

    @staticmethod
    def export_onnx(model: ModelRecord) -> tuple[bool, str, str | None]:
        """将模型导出为 ONNX 格式

        Returns:
            (success, message, onnx_file_path)
        """
        from app.services.inference_service import ModelInferenceService

        model_obj, metadata, _, error = ModelInferenceService.load_model(model)
        if error:
            return False, f'无法加载模型: {error}', None

        if not metadata:
            metadata = {}

        framework = metadata.get('framework', 'sklearn')
        metadata.get('task_type', model.model_type)
        feature_count = metadata.get('input_dim', 10)

        export_dir = os.path.join('experiments', 'exports', model.uuid)
        os.makedirs(export_dir, exist_ok=True)
        onnx_path = os.path.join(export_dir, f'{model.name_slug}.onnx')

        try:
            if framework == 'sklearn':
                # sklearn → ONNX (skl2onnx)
                return ModelExportService._export_sklearn_onnx(model_obj, metadata, feature_count, onnx_path)
            elif framework == 'pytorch':
                # PyTorch → ONNX
                return ModelExportService._export_pytorch_onnx(model_obj, metadata, feature_count, onnx_path)
            elif framework in ('tensorflow', 'keras', 'tf'):
                # TensorFlow → ONNX (tf2onnx)
                return ModelExportService._export_tf_onnx(model, metadata, feature_count, onnx_path)
            else:
                return False, f'框架 "{framework}" 暂不支持 ONNX 导出', None

        except ImportError as e:
            return False, f'缺少依赖: {str(e)}。请运行: pip install skl2onnx onnx', None
        except Exception as e:
            logger.error(f'ONNX 导出失败: {e}', exc_info=True)
            return False, f'导出失败: {str(e)}', None

    @staticmethod
    def _export_sklearn_onnx(model_obj, metadata, feature_count, onnx_path):
        """sklearn → ONNX"""
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [('float_input', FloatTensorType([None, feature_count]))]

        try:
            onx = convert_sklearn(model_obj, initial_types=initial_type)
        except Exception:
            # 如果 convert_sklearn 不支持，尝试使用通用的 to_onnx
            from skl2onnx import to_onnx

            onx = to_onnx(model_obj, initial_types=initial_type)

        with open(onnx_path, 'wb') as f:
            f.write(onx.SerializeToString())

        logger.info(f'sklearn 模型已导出为 ONNX: {onnx_path}')
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
            },
        )

        logger.info(f'PyTorch 模型已导出为 ONNX: {onnx_path}')
        return True, f'ONNX 导出成功: {os.path.basename(onnx_path)}', onnx_path

    @staticmethod
    def _export_tf_onnx(model, metadata, feature_count, onnx_path):
        """TensorFlow/Keras → ONNX (tf2onnx)"""
        try:
            import tensorflow as tf
            import tf2onnx

            # 加载 Keras 模型文件
            keras_path = model.model_file_path
            if not keras_path or not os.path.exists(keras_path):
                return False, f'Keras 模型文件不存在: {keras_path}', None

            loaded_model = tf.keras.models.load_model(keras_path)
            spec = (tf.TensorSpec((None, feature_count), tf.float32, name='float_input'),)
            output_path = onnx_path

            tf2onnx.convert.from_keras(loaded_model, input_signature=spec, output_path=output_path)

            logger.info(f'TensorFlow 模型已导出为 ONNX: {onnx_path}')
            return True, f'ONNX 导出成功: {os.path.basename(onnx_path)}', onnx_path
        except ImportError:
            return False, '缺少依赖: pip install tf2onnx', None

    @staticmethod
    def generate_deployment_package(model: ModelRecord) -> tuple[bool, str, str | None, str | None]:
        """生成完整部署包 (Dockerfile + serve.py + requirements.txt + docker-compose.yml + README.md)

        所有文件打包到一个目录，方便一键部署。

        Returns:
            (success, message, package_dir_path, zip_filename)
        """
        from app.services.inference_service import ModelInferenceService

        model_obj, metadata, _, error = ModelInferenceService.load_model(model)
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

        # 2. 复制模型文件 (动态收集所有相关文件)
        copied_files = ['serve.py']
        model_path = model.model_file_path
        if model_path and os.path.exists(model_path):
            ext = os.path.splitext(model_path)[1]
            model_dest = os.path.join(package_dir, f'model{ext}')
            shutil.copy2(model_path, model_dest)
            copied_files.append(f'model{ext}')

            # 复制配置文件
            config_path = model_path + '_config.pkl'
            if os.path.exists(config_path):
                dest = os.path.join(package_dir, f'model{ext}_config.pkl')
                shutil.copy2(config_path, dest)
                copied_files.append(f'model{ext}_config.pkl')

            # 复制 PyTorch .pt
            _base, _ = os.path.splitext(model_path)
            pt_path = _base + '.pt'
            if os.path.exists(pt_path) and pt_path != model_path:
                shutil.copy2(pt_path, os.path.join(package_dir, 'model.pt'))
                copied_files.append('model.pt')

            # 复制 Keras .keras
            keras_path = _base + '.keras'
            if os.path.exists(keras_path) and keras_path != model_path:
                shutil.copy2(keras_path, os.path.join(package_dir, 'model.keras'))
                copied_files.append('model.keras')

            # 复制 .h5
            h5_path = _base + '.h5'
            if os.path.exists(h5_path) and h5_path != model_path:
                shutil.copy2(h5_path, os.path.join(package_dir, 'model.h5'))
                copied_files.append('model.h5')

            # 复制 ONNX (如果已导出)
            onnx_path = _base + '.onnx'
            if os.path.exists(onnx_path) and onnx_path != model_path:
                shutil.copy2(onnx_path, os.path.join(package_dir, 'model.onnx'))
                copied_files.append('model.onnx')

        # 3. requirements.txt (根据框架生成)
        reqs = ModelExportService._generate_requirements(framework)
        with open(os.path.join(package_dir, 'requirements.txt'), 'w', encoding='utf-8') as f:
            f.write(reqs)
        copied_files.append('requirements.txt')

        # 4. Dockerfile (动态生成 COPY 指令)
        copy_lines = '\n'.join(f'COPY {f} ./' for f in sorted(copied_files))
        # 若有 config 文件，通配符 COPY (零匹配时 Docker 不报错)
        has_config = any(f.endswith('_config.pkl') for f in copied_files)
        if has_config:
            copy_lines += '\nCOPY *_config.pkl ./'

        dockerfile = ModelExportService.DOCKERFILE_TEMPLATE.format(
            model_name=model.name,
            framework=framework,
            task_type=task_type,
            copy_instructions=copy_lines,
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

        # 6. README.md (Task 13)
        readme = ModelExportService.generate_model_readme(model, framework, task_type, metadata)
        with open(os.path.join(package_dir, 'README.md'), 'w', encoding='utf-8') as f:
            f.write(readme)

        # 7. metadata.json — 完整模型元数据 (支持双向导入, 保留所有DB字段 + 推理信息)
        meta_export = model.to_dict(include_files=True)
        meta_export['exported_at'] = localnow().isoformat()
        meta_export['export_tool_version'] = '2.0'
        # 补充推理元数据 (from InferenceService load_model)
        if metadata:
            meta_export['_inference_meta'] = {
                'task_type': metadata.get('task_type'),
                'algorithm': metadata.get('algorithm'),
                'feature_names': metadata.get('feature_names', []),
                'class_labels': metadata.get('class_labels', []),
                'has_scaler': metadata.get('scaler') is not None,
                'has_vectorizer': metadata.get('vectorizer') is not None,
                'label_encoders_keys': list((metadata.get('label_encoders') or {}).keys()),
            }
        # 移除不适合序列化的对象
        meta_export.pop('metrics', None)
        meta_export.pop('hyperparameters', None)
        meta_export['metrics'] = model.metrics_dict
        meta_export['hyperparameters'] = model.hyperparameters_dict
        with open(os.path.join(package_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(meta_export, f, ensure_ascii=False, default=str, indent=2)
        copied_files.append('metadata.json')

        # 8. 打包为 .zip (供下载)
        import re as _re

        safe_name = _re.sub(r'[^\w\-]', '_', model.name)
        zip_base = os.path.join('experiments', 'exports', model.uuid, f'{safe_name}_deploy')
        shutil.make_archive(zip_base, 'zip', package_dir)
        zip_file = f'{safe_name}_deploy.zip'
        logger.info(f'部署包 zip 已生成: {zip_base}.zip')

        file_count = len(os.listdir(package_dir))
        logger.info(f'部署包已生成: {package_dir} ({file_count} 个文件)')
        return True, f'部署包已生成 ({file_count} 个文件)', package_dir, zip_file

    @staticmethod
    def _generate_requirements(framework: str) -> str:
        """根据框架生成 requirements.txt"""
        reqs = 'fastapi>=0.100.0\nuvicorn>=0.23.0\nnumpy>=1.24.0\n'
        if framework == 'pytorch':
            reqs += 'torch>=2.0.0\n'
        elif framework in ('tensorflow', 'keras', 'tf'):
            reqs += 'tensorflow>=2.12.0\n'
        elif framework == 'transformers':
            reqs += 'torch>=2.0.0\ntransformers>=4.30.0\n'
        reqs += 'scikit-learn>=1.3.0\npython-multipart>=0.0.6\n'
        if framework == 'onnx':
            reqs += 'onnxruntime>=1.15.0\n'
        return reqs

    @staticmethod
    def generate_model_readme(
        model: ModelRecord, framework: str = None, task_type: str = None, metadata: dict = None
    ) -> str:
        """生成模型部署 README.md (Task 13)

        Args:
            model: 模型记录
            framework: 框架名 (sklearn/pytorch/tensorflow/onnx)
            task_type: 任务类型 (classification/regression/clustering/nlp)
            metadata: 模型元数据 (feature_names, label_encoders, etc.)

        Returns:
            README.md 内容字符串
        """
        framework = framework or model.framework or 'sklearn'
        task_type = task_type or model.model_type
        metadata = metadata or {}

        feature_names = metadata.get('feature_names', [])
        label_encoders = metadata.get('label_encoders', {})
        target_le = label_encoders.get('__target__')

        # API 端点示例
        api_example = """```bash
# 健康检查
curl http://localhost:8000/health

# 单条预测
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [[1.0, 2.5, 3.0]]}'

# 批量预测
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [[1.0, 2.5], [3.0, 4.5], [5.0, 6.5]]}'
```"""

        # 类别标签 (分类模型)
        classes_section = ''
        if target_le and hasattr(target_le, 'classes_'):
            classes = list(target_le.classes_)
            classes_section = f"""
## 类别标签

| 索引 | 类别 |
|------|------|
{chr(10).join(f'| {i} | `{c}` |' for i, c in enumerate(classes))}
"""

        # 特征列表
        features_section = ''
        if feature_names:
            max_show = 20
            shown = feature_names[:max_show]
            features_section = f"""
## 输入特征

共 **{len(feature_names)}** 个特征:

| # | 特征名 |
|---|--------|
{chr(10).join(f'| {i + 1} | `{name}` |' for i, name in enumerate(shown))}
"""
            if len(feature_names) > max_show:
                features_section += f'\n*... 还有 {len(feature_names) - max_show} 个特征未列出*\n'

        readme = f'''# {model.name} — 模型部署包

## 模型信息

| 属性 | 值 |
|------|-----|
| 名称 | {model.name} |
| 版本 | {model.version} |
| 框架 | {framework} |
| 任务类型 | {task_type} |
| 描述 | {model.description or '-'} |
| 生成时间 | {localnow().strftime('%Y-%m-%d %H:%M:%S')} |

## 性能指标

| 指标 | 值 |
|------|-----|
| Accuracy | {f'{model.accuracy * 100:.2f}%' if model.accuracy is not None else '-'} |
| Precision | {f'{model.precision:.4f}' if model.precision is not None else '-'} |
| Recall | {f'{model.recall:.4f}' if model.recall is not None else '-'} |
| F1 Score | {f'{model.f1_score:.4f}' if model.f1_score is not None else '-'} |
| R² | {f'{model.r2:.4f}' if model.r2 is not None else '-'} |
| MSE | {f'{model.mse:.4f}' if model.mse is not None else '-'} |
{features_section}
{classes_section}
## 快速部署

### Docker (推荐)

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

### 手动部署

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python serve.py
```

服务默认监听 `http://0.0.0.0:8000`

## API 文档

### GET /health

健康检查端点。

**响应 200:**
```json
{{"status": "healthy", "model": "{model.name}"}}
```

### POST /predict

模型推理端点。

**请求体:**
```json
{{"features": [[1.0, 2.5, 3.0]]}}
```

**响应 200:**
```json
{{"predictions": ["class_a"], "task_type": "{task_type}"}}
```

### 示例

{api_example}

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `8000` | 服务端口 |
| `MODEL_PATH` | `./model` | 模型文件路径 (不含扩展名) |

## 文件说明

| 文件 | 说明 |
|------|------|
| `serve.py` | FastAPI 推理服务脚本 |
| `Dockerfile` | Docker 镜像构建文件 |
| `docker-compose.yml` | Docker Compose 编排文件 |
| `requirements.txt` | Python 依赖清单 |
| `model.*` | 模型权重文件 |
| `README.md` | 本文件 |
'''

        return readme

    @staticmethod
    def get_export_info(model: ModelRecord) -> dict:
        """获取模型的导出信息和状态"""
        export_dir = os.path.join('experiments', 'exports', model.uuid)
        info = {
            'onnx_available': False,
            'deploy_available': False,
            'metadata_available': False,
            'onnx_path': None,
            'deploy_dir': None,
            'framework': model.framework or 'sklearn',
            'supported_formats': [],
        }

        # 检查 ONNX 导出
        if os.path.exists(export_dir):
            onnx_files = [f for f in os.listdir(export_dir) if f.endswith('.onnx')]
            if onnx_files:
                info['onnx_available'] = True
                info['onnx_path'] = os.path.join(export_dir, onnx_files[0])

            # 检查 metadata.json
            meta_path = os.path.join(export_dir, 'metadata.json')
            if os.path.exists(meta_path):
                info['metadata_available'] = True
            else:
                # 老版本可能在 deploy/ 内
                deploy_dir = os.path.join(export_dir, 'deploy')
                meta_path2 = os.path.join(deploy_dir, 'metadata.json') if os.path.exists(deploy_dir) else None
                if meta_path2 and os.path.exists(meta_path2):
                    info['metadata_available'] = True

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
