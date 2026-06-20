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
│   │   └── api/             # REST API (models, training, datasets, users, auth, stream, comments)
│   ├── services/            # 业务逻辑层
│   │   ├── model_service.py         # 模型注册/统计 + 模型卡片生成
│   │   ├── training_service.py
│   │   ├── dataset_service.py
│   │   ├── auth_service.py
│   │   ├── inference_service.py     # 模型加载/预测/评估
│   │   ├── hyperparameter_tuning.py # 超参数调优 (GridSearchCV+Random+AutoML)
│   │   ├── dataset_recommendation_service.py
│   │   ├── dataset_import_service.py
│   │   ├── model_export_service.py  # ONNX导出/Docker部署包
│   │   ├── export_task_tracker.py   # 异步导出进度跟踪
│   │   ├── feature_extractor.py     # 图像/文本特征提取
│   │   ├── comment_service.py       # 评论区管理
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
├── scripts/                 # 种子数据, 批量训练, NLP重训练, e2e验证
├── tests/                   # pytest 测试 (124个)
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
| **sklearn** | RandomForest, GradientBoosting, SVM, KNN, LogisticRegression, LinearRegression, Ridge, DecisionTree, KMeans, DBSCAN, Agglomerative, MiniBatchKMeans | 分类/回归/聚类全覆盖, NLP TF-IDF自动管道 |
| **PyTorch** | MLP 分类/回归 | GPU加速, AdamW+CosineAnnealing, BatchNorm+Dropout |
| **TensorFlow/Keras** | MLP (Dense+ReLU+BatchNorm+Dropout) | .keras 保存, CPU训练 |

## 超参数调优系统 (v9 — 真随机修复)

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

### random_state 控制 (v9)

所有调优方法默认 `random_state=None` (真随机), 如需可复现结果可传 `int` 种子:

```python
# 真随机 — 每次运行不同结果
result = HyperparameterTuningService.run_random_search(dataset, algo, 'classification', target)

# 可复现 — 固定种子
result = HyperparameterTuningService.run_random_search(
    dataset, algo, 'classification', target, random_state=42)
```

影响范围: `_create_model()` → `RandomizedSearchCV` → `_manual_clustering_search()` → CV split

表单参数: `<input name="random_seed" type="number">` (可选, 空=真随机)

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

## 模型测试与导出 (Phase 1-3 新增)

### 模型交互测试 API

| 端点 | 用途 |
|------|------|
| `POST /api/v1/models/<uuid>/predict` | 批量预测 (支持 CSV/Excel/JSON/图像上传) |
| `POST /api/v1/models/<uuid>/quick-predict` | 快速预测 (文本 NLP 或 JSON 特征值) |
| `GET /api/v1/models/<uuid>/predict-template` | 下载 CSV 预测模板 (含特征列名) |
| `POST /api/v1/models/<uuid>/predict-export` | 批量预测 + 结果导出 (CSV/JSON) |
| `POST /api/v1/models/<uuid>/evaluate` | 完整模型评估 (train/test split) |
| `GET /api/v1/models/<uuid>/feature-importance` | 特征重要性分析 |

### 模型导出与部署 API

| 端点 | 用途 |
|------|------|
| `GET /api/v1/models/<uuid>/download` | 下载原始模型文件 |
| `GET /api/v1/models/<uuid>/export/info` | 获取导出状态 (ONNX/Docker) |
| `POST /api/v1/models/<uuid>/export/onnx` | 导出 ONNX 格式 (同步) |
| `POST /api/v1/models/<uuid>/export/deploy` | 生成 Docker 部署包 (同步) |
| `POST /api/v1/models/<uuid>/export/async/<type>` | 启动异步导出任务 (返回 task_id) |
| `GET /api/v1/models/<uuid>/export/status?task_id=x` | 查询异步导出进度 |
| `GET /api/v1/models/<uuid>/export/download/<filename>` | 下载导出文件 (zip) |

### 部署与服务 API (Phase 3)

| 端点 | 用途 |
|------|------|
| `GET /api/v1/models/<uuid>/deploy/health` | Docker 部署健康检查 (healthy/unreachable/not_deployed) |
| `POST /api/v1/models/<uuid>/serve` | 直接模型推理端点 (镜像 Docker serve.py /predict 契约) |
| `GET /api/v1/models/<uuid>/model-card` | HuggingFace 风格模型卡片 (Markdown, 支持 ?format=json) |

### 模型卡片 (`generate_model_card()`)

`ModelService.generate_model_card(model)` 生成 HuggingFace 风格的完整模型卡片:

- **YAML 元数据头** — language, tags, license, framework, task_type, pipeline_tag
- **模型描述** — 描述文本 + 用途说明
- **预期用途与局限** — Use Cases / Out-of-Scope
- **训练过程** — 超参数表 + 训练时长
- **评估结果** — 指标表 (分类/回归/聚类自适应)
- **输入特征** — 特征列表 (最多20个, 自适应列名)
- **类别标签** — 分类模型类别映射
- **使用方法** — Python API / cURL / Docker 三示例
- **局限性声明** — 5 条标准局限 + 框架特定说明
- **BibTeX 引用** — 自动生成, 特殊字符转义

### Docker 部署包 (`model_export_service.py`)

`/export/deploy` 生成完整 Docker 部署包:
- `serve.py` — FastAPI 推理服务 (/health + /predict)
- `Dockerfile` — HEALTHCHECK + 多框架支持
- `docker-compose.yml` — 单容器编排
- `requirements.txt` — 按框架生成 (PyTorch/TensorFlow/sklearn)
- `README.md` — 部署说明 + API 文档 + 示例
- `model.*` — 模型权重文件 (自动收集 .pkl/.pt/.keras/.h5/.onnx)

### 异步导出 (`export_task_tracker.py`)

- `ExportTaskTracker` — 单例, 内存存储, 线程安全
- `POST /export/async/<onnx|deploy>` → 返回 `task_id` → 前端 1.5s 轮询 `GET /export/status`
- 支持进度平滑推进 (10%→50%→100%) + 超时保护 (90s)

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

### 美化增强 v4.0

| 组件 | 文件 | 功能 |
|------|------|------|
| Animate.css 4.1.1 | CDN | 即用型 CSS 动画库 |
| `beautify.css` | `app/static/css/` | 28类CSS效果 (粒子/3D/涟漪/发光边框/骨架屏) |
| `beautify.js` | `app/static/js/` | 11个交互模块 (Canvas粒子网络 + 滚动揭示 + 3D倾斜 + 涟漪) |

核心特效:
- Canvas **粒子网络背景** — 科技感节点+连线+鼠标交互, 60fps, 移动端降密度50%
- **滚动揭示** — IntersectionObserver, 支持 fadeUp/Left/Right/Scale + 延迟序列
- **3D卡片倾斜** — `tilt-card` class, 随鼠标位置旋转
- **涟漪按钮** — `btn-ripple` class, 点击水波扩散
- **页面过渡** — 内容区淡入上移动画
- **无障碍**: `prefers-reduced-motion` 自动关闭动画

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
python -m pytest tests/ -v    # 162 个测试
python -m pytest tests/ -q    # 快速模式
```

## 工具模块

### TTL 缓存 (`app/utils/cache.py` v2.0)

```python
cache = TTLCache(default_ttl=60, max_size=500, cleanup_interval=60)

@cache.cached(key='my_key', ttl=30)
def heavy_query(): ...

cache.stats  # {'hits': 42, 'misses': 5, 'hit_rate': 0.893, 'size': 3, 'evictions': 0}
cache.invalidate('dashboard_')  # 前缀批量清除
```

全局单例: `dashboard_cache` (60s), `leaderboard_cache` (300s), `recommendation_cache` (600s)

### 表单参数解析 (`app/utils/helpers.py`)

```python
from app.utils.helpers import parse_form_params

params = parse_form_params(dict(request.form),
    int_fields={'batch_size', 'epochs', 'max_depth', 'n_estimators'},
    float_fields={'learning_rate', 'test_size', 'dropout'},
    str_fields={'algorithm', 'kernel', 'penalty'})
```

### 错误页面

- `errors/404.html` — 玻璃态卡片 + 返回首页/上页按钮
- `errors/500.html` — 玻璃态卡片 + 刷新/返回按钮
- `errors/error.html` — 通用回退 (保留 HTTPException 兜底)

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
12. **AutoML伪随机 (v9)**: 移除所有硬编码 `random_state=42`, 默认 `None` 真随机, 支持可复现种子
13. **前端美化 v4.0**: Canvas粒子网络 + CSS/JS增强 + Animate.css + 滚动揭示 + 3D倾斜
14. **缓存升级 v2.0**: 后台定期清理 + max_size驱逐 + 命中率统计
15. **CSS样式去重**: 合并两处 `.form-control:focus` 定义, 移除 `!important` 覆写
16. **404/500错误页**: 专用模板替代泛型 error.html
17. **serve端点类型安全 (v10)**: 非dict JSON body检测 `isinstance(data, dict)`, predict()异常包裹 try/except, 503/400分类补充`加载失败`关键词
18. **YAML/BibTeX注入防护 (v10)**: `generate_model_card()` 对 model名/owner名做YAML双引号转义 + BibTeX特殊字符转义 (`\`, `$`, `&`, `#`, `%`, `_`, `~`, `^`, `{`, `}`)
19. **HTTP连接泄漏 (v10)**: `/deploy/health` 使用 `with urlopen() as resp` 确保连接关闭
20. **JS fetch错误处理 (v10)**: detail.html deploy health badge 添加 `.catch()` 处理网络错误, 显示红色`检查失败`状态
21. **NLP TF-IDF管道 (v11)**: SklearnTrainer自动检测`dataset.category='nlp'`→TF-IDF(jieba分词 tokenizer, 500维); save_model保存vectorizer+class_labels→load_model提取→quick_predict端到端预测; `_jieba_cut()` 模块级函数确保可pickle
22. **set_metrics test_前缀修复 (v11)**: 分类指标按`test_`→裸键优先级查找(对齐已有回归逻辑), 修复所有sklearn模型accuracy列存储为None的问题
23. **DataFrame列名对齐 (v11)**: quick_predict/batch_predict中vectorizer路径使用metadata.feature_names对齐列名, 避免int列名vs str特征名不匹配导致predict失败
24. **NLP模型质量全面修复 (v12, 6批次A-F)**: 删除80个过拟合模型(clean_overfit_models.py); NLP训练管道升级(argparse/并行/CV/数据增强/文本增强); 推理健壮修复(特征对齐/Scaler验证/labelEncoder回退/confidence修复/输入校验); 新增38个NLP测试(推理16+管道19+模型3); 全量重训练70→18成功→删除17低质量→117个高质量模型(A级62/B级46/C级9)
25. **requirements.txt补全 (v12)**: 添加jieba/imbalanced-learn/scikit-image/torchvision 4个依赖
26. **config.py NLP配置 (v12)**: 添加NLP_DEFAULT_MAX_FEATURES/MIN_DF/MAX_DF/BALANCE_MODE/CV_FOLDS 5个配置项

## NLP 文本处理管道 (v11 新增)

### 训练阶段

`SklearnTrainer.load_data()` 自动检测 NLP 数据集:
1. 检查 `dataset.category == 'nlp'`
2. 扫描 `['text', 'review', 'comment', 'content', 'sentence']` 列
3. 若找到文本列 → `TfidfVectorizer(tokenizer=_jieba_cut, max_features=500)` (jieba中文分词, 词级切分优于char ngram) → 替换原始文本列; jieba不可用时回退到char ngram
4. 保存 `class_labels` (sorted unique string labels) 到 bundle
5. `save_model()` 输出 `{'model', 'scaler', 'label_encoders', 'feature_names', 'task_type', 'algorithm', 'vectorizer', 'class_labels'}`

### 预测阶段

`quick_predict` (POST /api/models/<uuid>/quick-predict) 两条路径:
- **路径1d-i**: vectorizer可用 → `vectorizer.transform(text)` → TF-IDF → DataFrame(columns=feature_names) → `ModelInferenceService.predict()` → 返回概率分布
- **路径1d-ii**: vectorizer缺失 → `FeatureExtractor.analyze_sentiment()` 关键词情感分析fallback

### NLP 重训练

```bash
python scripts/retrain_nlp.py --quick   # 1个模型快速验证
python scripts/retrain_nlp.py           # 全部3个数据集×3种算法
python scripts/retrain_nlp.py --algo all --cv 5 --augment 2 --balance undersample --workers 2  # 全量并行重训练
python scripts/verify_nlp_e2e.py        # 端到端验证 (直接预测+HTTP API)
```

### NLP 质量清理 (Batch A-F 全部完成)

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 总模型 | 290 | 239 |
| NLP sklearn模型 | 99 | 117 (重训练后) |
| 过拟合模型 (acc≥99.9%) | 80 | 0 |
| 测试通过 | 124 | 162 |

清理脚本:
```bash
python scripts/clean_overfit_models.py   # 删除准确率≥99.9%的过拟合模型 (THRESHOLD=0.999)
```

### NLP 配置项 (config.py)

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `NLP_DEFAULT_MAX_FEATURES` | 2000 | TF-IDF 最大特征数 |
| `NLP_DEFAULT_MIN_DF` | 2 | 最小文档频率 |
| `NLP_DEFAULT_MAX_DF` | 0.9 | 最大文档频率 |
| `NLP_DEFAULT_BALANCE_MODE` | smote | 类别平衡方式 (smote/undersample/none) |
| `NLP_DEFAULT_CV_FOLDS` | 5 | 交叉验证折数 |

### 已知局限
- jieba 分词 TF-IDF 4/4 测试正确 (替代了原本 2/4 的 char ngram)
- 若训练集同时含文本列+数值列, quick_predict仅生成TF-IDF特征→列数不匹配→回退到关键词fallback
- 旧NLP模型(Phase 3)无vectorizer, 需重新训练
- jieba 必须已安装 (`pip install jieba`); 不可用时自动回退 char ngram

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
