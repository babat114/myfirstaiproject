# 🤖 AI Model & Dataset Management Platform

一站式 AI 模型与数据集管理平台 —— 基于 Python + Flask + MySQL 构建的全栈 Web 应用。

## 📋 功能特性

### 🔐 用户认证
- 用户注册与登录 (支持用户名/邮箱登录)
- 角色权限管理 (管理员 / 研究员 / 查看者)
- API Key 认证支持
- 个人资料管理与密码修改

### 📊 数据集管理
- 数据集上传 (CSV, JSON, Parquet, 图片等格式)
- 元数据管理 (名称、描述、标签、版本)
- 分类组织 (图像、文本、表格、音频、视频等)
- 公开/私有权限控制
- 文件大小与统计追踪

### 🎓 AI 模型注册表
- 模型注册与版本管理
- 性能指标追踪 (准确率、精确率、召回率、F1)
- 超参数记录
- 模型文件上传 (PyTorch, TensorFlow, ONNX 等)
- 排行榜功能
- 部署状态追踪

### ▶️ 训练任务管理
- 训练任务创建与配置
- 实时进度监控
- 训练日志查看
- 资源配额管理 (GPU/CPU/内存)
- 任务生命周期管理 (启动/暂停/取消/完成)

### 🔮 模型预测与推理
- 交互式模型测试页 (文本/特征/文件/图像 4 种输入)
- 实时 NLP 情感预测 (500ms 防抖)
- 批量预测 + 结果导出 (CSV/JSON)
- Docker 一键部署包自动生成
- HuggingFace 风格模型卡片

### 🔍 质量治理
- 7 维模型质量自动诊断
- Tier 1/2/3 分级管理系统
- 自动看门狗 (once/daemon/quick 三模式)
- 超参数调优 (GridSearchCV / RandomSearchCV / AutoML)
- AI 智能诊断面板 (健康评分 + 参数建议)

### 🔌 RESTful API
- 60+ JSON API 端点 (7 Tag 分组)
- Swagger 3.0.3 在线文档
- Session / JWT Bearer / API Key 三模式认证
- 支持分页、筛选、搜索

### 📈 平台统计
- **已训练模型**: 114 个 (68 NLP + 46 通用)，5 种框架
- **数据集**: 100 个，15 个公开数据源
- **质量**: 0 过拟合、0 常数预测器
- **测试**: 356 个 pytest (覆盖率 73%)
- **文档**: [需求规格说明书](需求规格说明书.md) | [架构设计](ARCHITECTURE.md) | [PRD](PRD.md)

## 🏗️ 技术架构

```
myfirstaiproject/
├── app/                          # 应用主包
│   ├── __init__.py               # Flask 应用工厂
│   ├── models/                   # 数据库模型层
│   │   ├── user.py               # 用户模型
│   │   ├── dataset.py            # 数据集模型
│   │   ├── model_record.py       # 模型注册模型
│   │   └── training_job.py       # 训练任务模型
│   ├── services/                 # 业务逻辑层
│   │   ├── auth_service.py       # 认证服务
│   │   ├── dataset_service.py    # 数据集服务
│   │   ├── model_service.py      # 模型服务
│   │   └── training_service.py   # 训练服务
│   ├── routes/                   # 路由层
│   │   ├── auth.py               # 认证页面路由
│   │   ├── dashboard.py          # 仪表盘路由
│   │   ├── datasets.py           # 数据集页面路由
│   │   ├── models.py             # 模型页面路由
│   │   ├── training.py           # 训练页面路由
│   │   └── api/                  # RESTful API
│   │       ├── datasets.py       # 数据集 API
│   │       ├── models.py         # 模型 API
│   │       └── training.py       # 训练 API
│   ├── templates/                # Jinja2 模板
│   │   ├── base.html             # 基础布局
│   │   ├── dashboard.html        # 仪表盘
│   │   ├── datasets/             # 数据集页面
│   │   ├── models/               # 模型页面
│   │   ├── training/             # 训练页面
│   │   └── errors/               # 错误页面
│   ├── static/                   # 静态资源
│   │   ├── css/style.css         # 样式
│   │   └── js/main.js            # 脚本
│   └── utils/                    # 工具
│       ├── helpers.py            # 辅助函数
│       └── decorators.py         # 装饰器
├── scripts/                      # 脚本
│   ├── init_db.py                # 数据库初始化
│   └── seed_data.py              # 种子数据
├── tests/                        # 测试
│   ├── conftest.py               # 测试配置
│   ├── test_auth.py              # 认证测试
│   ├── test_datasets.py          # 数据集测试
│   └── test_models.py            # 模型测试
├── config.py                     # 配置管理
├── run.py                        # 启动入口
├── requirements.txt              # 依赖包
└── .env.example                  # 环境变量示例
```

## 🚀 快速开始

### 前置要求

- Python 3.11+
- MySQL 8.0+
- Git

### 1. 克隆项目

```bash
cd ~/Desktop/myfirstaiproject
```

### 2. 创建虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置数据库

```bash
# 复制环境变量文件
cp .env.example .env

# 编辑 .env 文件，修改 MySQL 连接信息
# MYSQL_USER=root
# MYSQL_PASSWORD=your_password
# MYSQL_DATABASE=ai_platform
```

**创建 MySQL 数据库：**

```sql
CREATE DATABASE ai_platform CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 5. 初始化数据库

```bash
python scripts/init_db.py
```

### 6. (可选) 填充示例数据

```bash
python scripts/seed_data.py
```

### 7. 启动应用

```bash
python run.py
```

访问 **http://127.0.0.1:5000**

### 默认账户

| 角色 | 用户名 | 密码 |
|------|--------|------|
| 管理员 | admin | Admin123456 |
| 研究员 | researcher1 | User123456 |
| 查看者 | viewer1 | User123456 |

## 📡 API 接口

### 认证方式

在请求头中添加 API Key：
```
X-API-Key: ak_your_api_key_here
```

### 数据集 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/datasets/` | 获取数据集列表 |
| GET | `/api/datasets/<uuid>` | 获取数据集详情 |
| POST | `/api/datasets/` | 上传新数据集 |
| PUT | `/api/datasets/<uuid>` | 更新数据集 |
| DELETE | `/api/datasets/<uuid>` | 删除数据集 |
| GET | `/api/datasets/stats` | 获取统计信息 |

### 模型 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/models/` | 获取模型列表 |
| GET | `/api/models/<uuid>` | 获取模型详情 |
| POST | `/api/models/` | 注册新模型 |
| PUT | `/api/models/<uuid>` | 更新模型 |
| PUT | `/api/models/<uuid>/metrics` | 更新性能指标 |
| DELETE | `/api/models/<uuid>` | 删除模型 |
| GET | `/api/models/leaderboard` | 获取排行榜 |

### 训练 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/training/` | 获取任务列表 |
| GET | `/api/training/<uuid>` | 获取任务详情 |
| POST | `/api/training/` | 创建训练任务 |
| POST | `/api/training/<uuid>/start` | 启动训练 |
| PUT | `/api/training/<uuid>/progress` | 更新进度 |
| POST | `/api/training/<uuid>/complete` | 完成训练 |
| POST | `/api/training/<uuid>/fail` | 标记失败 |
| DELETE | `/api/training/<uuid>` | 删除任务 |

## 🧪 运行测试

```bash
# 运行所有测试
pytest

# 运行测试并显示覆盖率
pytest --cov=app

# 运行特定测试文件
pytest tests/test_auth.py -v
```

## 🔧 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `FLASK_ENV` | 运行环境 (development/testing/production) | development |
| `MYSQL_USER` | MySQL 用户名 | root |
| `MYSQL_PASSWORD` | MySQL 密码 | - |
| `MYSQL_HOST` | MySQL 主机 | localhost |
| `MYSQL_PORT` | MySQL 端口 | 3306 |
| `MYSQL_DATABASE` | 数据库名 | ai_platform |
| `SECRET_KEY` | Flask 密钥 | - |
| `JWT_SECRET_KEY` | JWT 密钥 | - |

## 📄 许可证

MIT License

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
