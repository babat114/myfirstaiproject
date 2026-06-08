# myfirstaiproject 开发 Skill

## 项目概览

AI模型训练管理平台 — Flask + MySQL Web 应用，支持数据集管理、多框架模型训练(sklearn/PyTorch/TensorFlow/ONNX)、模型注册/测试/对比/导出、超参数自动调优(GridSearchCV+RandomSearch+AutoML)。

## 启动

```bash
cd ~/Desktop/myfirstaiproject
python run.py
```
- URL: http://127.0.0.1:5000
- 默认账户: admin / Admin123456
- 普通用户: researcher1 / User123456

## 项目结构

```
myfirstaiproject/
├── app/
│   ├── __init__.py          # Flask app 工厂, CSRF, 蓝图注册
│   ├── models/              # SQLAlchemy 模型 (Dataset, ModelRecord, TrainingJob, User)
│   ├── routes/              # Web 路由 (models, training, datasets, dashboard, auth)
│   │   └── api/             # REST API (models, training, datasets, users, auth, stream)
│   ├── services/            # 业务逻辑层
│   │   ├── model_service.py
│   │   ├── training_service.py
│   │   ├── dataset_service.py
│   │   ├── auth_service.py
│   │   ├── inference_service.py
│   │   ├── hyperparameter_tuning.py   # 超参数调优 (GridSearchCV+Random+AutoML)
│   │   ├── dataset_recommendation_service.py
│   │   ├── dataset_import_service.py
│   │   ├── model_export_service.py
│   │   └── parameter_guidance_service.py
│   ├── executor/            # 训练执行引擎
│   │   ├── engine.py        # 单例训练引擎 (含algorithm纠错)
│   │   └── trainers/        # 训练器 (sklearn, pytorch, keras)
│   ├── templates/           # Jinja2 模板
│   │   └── training/
│   │       ├── detail.html              # 训练详情 (含AI智能优化面板)
│   │       ├── tuning.html              # 超参数调优页 (含AutoML)
│   │       └── _tuning_progress_modal.html  # SSE进度弹窗组件
│   ├── static/              # CSS, JS
│   └── utils/               # 装饰器, JWT, helpers
├── config.py                # 配置 (DevConfig, ProdConfig)
├── run.py                   # 入口
├── scripts/                 # 种子数据, 批量训练脚本
├── tests/                   # pytest 测试 (77个)
├── experiments/             # 训练实验输出
└── uploads/                 # 上传的数据集和模型文件
```

## 关键认证方式

| 优先级 | 方式 | 用途 |
|--------|------|------|
| 1 | Session (Flask-Login) | Web UI |
| 2 | JWT Bearer Token | API |
| 3 | API Key (X-API-Key) | API 兼容 |

## GPU 环境

- **硬件**: NVIDIA RTX 4060 Laptop (8GB VRAM, CUDA 13.2)
- **PyTorch**: 2.6.0+cu124 (CUDA 12.4), `torch.cuda.is_available() == True`
- **TensorFlow**: 2.20.0 (Windows pip 包为 CPU-only, GPU 需 WSL2)
- **安装**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
- 训练器自动检测 `torch.cuda.is_available()` → GPU 训练, 无需手动配置

## 训练器支持

| 框架 | 支持算法 | 备注 |
|------|---------|------|
| **sklearn** | RandomForest, GradientBoosting, SVM, KNN, LogisticRegression, LinearRegression, Ridge, DecisionTree, KMeans, DBSCAN, Agglomerative, MiniBatchKMeans | 分类/回归/聚类全覆盖 |
| **PyTorch** | MLP 分类/回归 | GPU加速, AdamW+CosineAnnealing, BatchNorm+Dropout |
| **TensorFlow/Keras** | MLP (Dense+ReLU+BatchNorm+Dropout) | .keras 保存, CPU训练 |

## 超参数调优系统 (v8)

### 三种调优模式

| 模式 | 方法 | 适用场景 |
|------|------|---------|
| **GridSearchCV** | 穷举搜索 + CV交叉验证 | 精确搜索, 参数组合少时使用 |
| **RandomSearchCV** | 随机采样 + CV | 快速探索, 参数空间大时推荐 |
| **AutoML** | 遍历所有适用算法, 每算法快速RandomSearch | 不确定用哪个算法时, 自动找最优 |

### 聚类特殊处理

聚类不使用 CV (交叉验证对无监督学习无意义)。改为:
- 遍历 `ParameterGrid`, 对每组参数: `fit(全量数据)` → `silhouette_score(子样本)`
- 子采样阈值: 3000 样本 (silhouette_score 是 O(n²), 大数据集自动降采样 ~90x 加速)
- 总步数 = 参数组合数 (不乘 CV, 比之前快 3x)

### AutoML 算法覆盖

| 任务类型 | 自动遍历的算法 |
|----------|---------------|
| classification | Random Forest → GBDT → Logistic Reg → SVM → KNN → Decision Tree → PyTorch MLP |
| regression | RF Reg → GBDT Reg → Linear → Ridge → SVR → KNN Reg → PyTorch MLP |
| clustering | K-Means → MiniBatch K-Means → Agglomerative → DBSCAN |

### KMeans algorithm 参数冲突

KMeans 的 `algorithm` 参数 (值: `lloyd`/`elkan`) 与 ML 算法名 `kmeans` 同名。
- **三层防护**: training.py 合并 → training_service.py 检测修正 → engine.py 启动前兜底
- 修改 `best_params` 时 **必须跳过冲突键**: `algorithm`, `ml_task_type`, `task_type`, `framework`

### SSE 实时进度

- `TuningProgressTracker` (单例): 线程安全进度状态, 无数据库轮询
- `GET /api/v1/stream/tuning/<tuning_id>/stream`: SSE 端点, 500ms 推送
- 进度不再卡 99%: 允许自然到达 100%
- AutoML 模式: 进度条 + 当前算法名 + 日志行 + 全局最佳

### 关键 API

| 端点 | 用途 |
|------|------|
| `GET /api/training/tuning/search-space?algorithm=x` | 获取搜索空间 |
| `POST /api/training/tuning/run` | 同步运行调优 (含 AutoML) |
| `GET /api/v1/stream/tuning/<id>/stream` | SSE 进度流 |
| `POST /training/<id>/gridsearch-retrain?async=1` | 异步调优+重训 |
| `POST /training/<id>/apply-tuning` | 应用调优结果 |

## AI 智能优化面板 (训练详情页)

整合于 `app/templates/training/detail.html`，按钮区:

```
[重新训练] [🤖 AI 智能优化] [手动调参] [查看模型]
```

三个 Tab:
- **AI 诊断**: 自动分析训练结果 (健康度评分/问题检测/参数建议), 调用 `/api/training/<uuid>/guidance`
- **GridSearchCV 单算法调优**: SSE 实时进度, 完成后一键应用最佳参数
- **AutoML 全算法对比**: 遍历所有算法排名, 可单独选用任一算法

## 数据集类别感知 (auto-config)

`GET /api/datasets/<id>/auto-config` 根据 dataset.category 发出警告:
- `classification`/`regression` → 强制匹配
- `synthetic` → ⚠ 合成数据 target 可能是随机噪声
- `nlp`/`vision`/`clustering`/`time_series` → ⚠ 标准分类/回归不适用
- 后备检测: >70% 特征名为 `latent_*` → 识别为生成式隐空间数据

## 前端系统

### 居中通知 (替代浏览器 alert/confirm)
- `showCenteredMessage(msg, type, dur)` — 屏幕正中浮动通知
- `showConfirmModal(msg, title, btnText)` — Promise-based 确认弹窗
- `[data-confirm="消息"]` — HTML属性自动拦截表单/按钮

### Flash 消息流
1. Flask `flash(message, category)` → Jinja2 渲染 alert
2. `base.html` 添加 `data-flash-category` + `data-flash-message` 属性
3. `el-enhance.js` `upgradeAlertsToElMessage()` 读取 → `showCenteredMessage()`

### 消息体系优先级
1. `showCenteredMessage()` — 居中浮动通知 (新)
2. `$message.success/error/warning/info()` — el-enhance 全局 API
3. `showToast()` — 兼容旧代码 (自动路由到居中通知)

## 测试

```bash
cd ~/Desktop/myfirstaiproject
python -m pytest tests/ -v    # 77 个测试
```

## 已修复的关键问题

1. **删除模型外键约束**: `ModelService.delete_model()` 先执行 `TrainingJob.query.filter_by(model_id=model.id).update({model_id: None})`
2. **公开模型权限**: `public_models()` 添加 `is_public=True` 过滤; `test_model()` 添加 `model.is_public` 检查
3. **numpy 序列化**: `auto_config` API 使用 `_py()` 转换 numpy bool/float 为 Python 原生类型
4. **名称重复**: 回归算法名含"回归"时不再追加 task_cn
5. **login_id vs username**: 登录表单字段是 `login_id` 不是 `username`
6. **聚类GridSearchCV卡99%**: 改用手动ParameterGrid遍历+全量数据silhouette, 不再用CV
7. **聚类silhouette极慢**: 自动子采样到3000点, O(n²)→O(k²), ~90x加速
8. **KMeans algorithm=lloyd冲突**: algorithm是KMeans参数值也是ML算法名, 三层防护修正
9. **PyTorch GPU**: 从 2.4.0+cpu 升级到 2.6.0+cu124, RTX 4060 可用
10. **agglomerative/DBSCAN非法参数**: 修复搜索空间中 ward+l1/cosine+kd_tree 不兼容组合
11. **RandomizedSearchCV error_score**: 改为 `error_score=0` 跳过非法组合而非崩溃

## 开发注意事项

- **Windows GBK**: 避免在 Python 源码中使用 emoji, 用 ASCII 替代
- **SQLAlchemy 2.0**: 使用 `db.session.get(Model, id)` 替代 `Model.query.get(id)`
- **Flask 测试客户端**: 用 `follow_redirects=True` 跟随重定向, 注意 session 隔离
- **numpy 兼容**: 涉及 torch/numpy 转换时显式指定 dtype, 用 `.tolist()` 替代 `.numpy()`
- **CSRF**: 测试时使用 `app.test_client()` 绕过 CSRF; API 蓝图已全局豁免
- **聚类调优**: 不要对无监督学习使用 CV; 用 `_manual_clustering_search()` 全量fit+子样本score
- **KMeans参数冲突**: best_params 合并到 hyperparams 时必须跳过 `algorithm` 等冲突键
- **调优进度**: 总步数聚类=参数组合数, 监督学习=组合数×CV; 进度可自然到达100%

## 常用模式

### 添加新 Web 路由
```python
@xxx_bp.route('/path')
@login_required
def view():
    ...
```

### 添加新 API
```python
@xxx_api_bp.route('/path', methods=['GET'])
@api_login_required  # 支持 Session/JWT/API Key
def api():
    ...
```

### 前端表单确认
```html
<form method="POST" data-confirm="确定删除？" action="...">...</form>
```

### JS 异步确认
```javascript
const ok = await showConfirmModal('确定删除？', '删除确认', '确认删除');
if (ok) { /* 执行 */ }
```

### 聚类手动搜索 (服务端)
```python
from app.services.hyperparameter_tuning import _manual_clustering_search
result = _manual_clustering_search(base_model, param_grid, X, progress_callback=cb)
# result: {success, best_score, best_params, cv_results, n_combinations}
```

### AutoML 异步调优
```python
from app.services.hyperparameter_tuning import HyperparameterTuningService
tuning_id = HyperparameterTuningService.run_auto_tuning_async(
    dataset=ds, task_type='clustering', target_column=None, cv=3)
# 前端连接 SSE: /api/v1/stream/tuning/<tuning_id>/stream
```
