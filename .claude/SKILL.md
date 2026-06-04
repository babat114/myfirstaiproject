# myfirstaiproject 开发 Skill

## 项目概览

AI模型训练管理平台 — Flask + MySQL Web 应用，支持数据集管理、多框架模型训练(sklearn/PyTorch/TensorFlow/ONNX)、模型注册/测试/对比/导出。

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
│   │   ├── hyperparameter_tuning.py
│   │   ├── dataset_recommendation_service.py
│   │   ├── dataset_import_service.py
│   │   ├── model_export_service.py
│   │   └── parameter_guidance_service.py
│   ├── executor/            # 训练执行引擎
│   │   ├── engine.py        # 单例训练引擎
│   │   └── trainers/        # 训练器 (sklearn, pytorch, keras)
│   ├── templates/           # Jinja2 模板
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

## 训练器支持

- **sklearn**: RandomForest, GradientBoosting, SVM, KNN, LogisticRegression, LinearRegression, Ridge, DecisionTree
- **PyTorch**: MLP 分类/回归 (NN.Sequential, GPU支持, AdamW+CosineAnnealing)
- **TensorFlow/Keras**: MLP (Dense+ReLU+BatchNorm+Dropout, .keras保存)

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

## 开发注意事项

- **Windows GBK**: 避免在 Python 源码中使用 emoji, 用 ASCII 替代
- **SQLAlchemy 2.0**: 使用 `db.session.get(Model, id)` 替代 `Model.query.get(id)`
- **Flask 测试客户端**: 用 `follow_redirects=True` 跟随重定向, 注意 session 隔离
- **numpy 兼容**: 涉及 torch/numpy 转换时显式指定 dtype, 用 `.tolist()` 替代 `.numpy()`
- **CSRF**: 测试时使用 `app.test_client()` 绕过 CSRF; API 蓝图已全局豁免

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
