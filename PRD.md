# AI模型训练管理平台 v2.0 — PRD 轻量增强版

> **⚠️ 历史文档**: 本文档为 v2.0 原始规划（已全部实现）。当前系统需求详见 **[需求规格说明书.md](需求规格说明书.md)** (SRS v3.0)，覆盖 114 模型/100 数据集/60+ API/5 框架/质量治理等完整功能。

## 项目目标

让当前平台从"训练任务追踪工具"升级为"**可在网页上传数据集、选择模型类型、一键执行真实训练**的 AI 工作台"。

**核心原则：** 零新增第三方依赖、本地直接运行、基于 Python 标准库。

---

## 用户核心流程

```
上传数据集 → 选择模型类型 → 配置超参数 → 点击开始训练
    → 实时看进度条和 Loss 曲线 → 训练完成 → 查看指标 → 下载模型
```

---

## 一、训练执行引擎 (ThreadPoolExecutor)

### 1.1 为什么不用 Celery/Redis
- 本地单机使用无需分布式
- `concurrent.futures.ThreadPoolExecutor` 是 Python 标准库
- 零配置，`python run.py` 启动即用

### 1.2 架构

```
用户在网页点"开始训练"
        │
        ▼
Flask 路由层  ──→  TrainingService.start_job()
        │
        ▼
TrainingExecutor (线程池)
        │
        ├── 线程1: sklearn 训练 (随机森林等)
        ├── 线程2: pytorch 训练 (简单神经网络)
        └── 主线程继续响应 Web 请求
        │
        ▼ (每个epoch回调)
更新数据库 (进度%、指标、日志)
        │
        ▼ (SSE推送)
前端页面实时更新进度条 + 折线图
```

### 1.3 新增文件

```
app/executor/
├── __init__.py
├── engine.py                 # 训练执行引擎
├── callbacks.py              # 训练回调 (记录指标/进度)
└── trainers/
    ├── __init__.py
    ├── base.py               # 抽象基类 BaseTrainer
    ├── sklearn_trainer.py    # scikit-learn 训练器 (优先实现)
    └── pytorch_trainer.py    # PyTorch 训练器
```

### 1.4 BaseTrainer 设计

```python
class BaseTrainer(ABC):
    """所有训练器的抽象基类"""
    
    def __init__(self, job, dataset, hyperparams):
        self.job = job              # TrainingJob 模型实例
        self.dataset = dataset       # Dataset 模型实例
        self.hyperparams = hyperparams
        self.pause_event = threading.Event()   # 暂停信号
        self.cancel_event = threading.Event()  # 取消信号
        self.pause_event.set()                  # 初始为"不暂停"
    
    @abstractmethod
    def load_data(self) -> tuple[X, y]: ...
    
    @abstractmethod
    def build_model(self): ...
    
    @abstractmethod
    def train_epoch(self, epoch: int) -> dict: ...
    
    @abstractmethod
    def evaluate(self) -> dict: ...
    
    @abstractmethod
    def save_model(self, path: str): ...
    
    # 模板方法 — 子类不需要重写
    def run(self):
        self.load_data()
        self.build_model()
        for epoch in range(self.total_epochs):
            if self.cancel_event.is_set(): break
            self.pause_event.wait()  # 暂停时阻塞
            metrics = self.train_epoch(epoch)
            self.update_progress_in_db(epoch, metrics)
        self.evaluate()
        self.save_model(final_path)
```

### 1.5 SklearnTrainer (优先实现)

支持用户在网页上选择以下模型类型：
- **分类 (Classification):** 随机森林、逻辑回归、SVM、KNN
- **回归 (Regression):** 线性回归、随机森林回归、SVR

用户在网页表单中选择：
```
模型类型:  [下拉框: 分类/回归]
算法:      [下拉框: 随机森林/逻辑回归/SVM/KNN/线性回归]
测试比例:  [滑块: 0.1 ~ 0.4]
```

### 1.6 暂停/取消机制

用 `threading.Event` 实现：
- 训练循环每个 epoch 间检查 `cancel_event`
- `pause_event.wait()` 在暂停时阻塞线程
- 取消时清理临时文件、更新 DB 状态为 `cancelled`

---

## 二、实时进度推送 (SSE)

### 2.1 方案
用 Flask 原生的 **Server-Sent Events**，不需要 WebSocket 或任何额外库。

### 2.2 服务端

```python
@app.route('/api/stream/training/<int:job_id>')
def training_stream(job_id):
    def generate():
        while True:
            job = TrainingJob.query.get(job_id)
            yield f"data: {json.dumps(job_status(job))}\n\n"
            if job.status in ('completed', 'failed', 'cancelled'):
                break
            time.sleep(2)
    return Response(generate(), mimetype='text/event-stream')
```

### 2.3 前端
```javascript
const source = new EventSource('/api/stream/training/123');
source.onmessage = (e) => {
    const data = JSON.parse(e.data);
    updateProgressBar(data.progress_percent);
    updateLossChart(data.metrics);
    appendLog(data.log_text);
};
```

### 2.4 训练详情页改造
- 进度条：实时更新 `progress_percent`
- Loss/Accuracy 折线图：使用 Chart.js 动态追加数据点
- 日志区：自动滚屏显示最新训练日志
- 按钮组：启动 / 暂停 / 恢复 / 取消

---

## 三、数据集上传与自动解析

### 3.1 现状
数据集上传后只存文件路径，没有自动分析。

### 3.2 新增功能
上传 CSV/Excel 后自动解析：
- 读取列名和类型
- 计算行列数
- 检测缺失值
- 自动推测 target 列（最后一列或命名为 label/target 的列）
- 生成数据预览（前 20 行 HTML 表格）

### 3.3 训练前数据预处理
- 自动处理缺失值（均值填充 / 删除）
- 自动编码分类变量（LabelEncoder / OneHotEncoder）
- 自动标准化（StandardScaler）
- 用户可选择 target 列

---

## 四、网页交互流程

### 4.1 创建训练任务页

```
┌──────────────────────────────────────────────┐
│  创建训练任务                                  │
│                                              │
│  任务名称: [________________]                 │
│                                              │
│  选择数据集: [下拉框: 我的数据集列表]            │
│    └─ 选中后自动预览数据、显示列信息             │
│                                              │
│  模型类型: [○分类  ○回归]                     │
│                                              │
│  算法: [下拉框: 随机森林/逻辑回归/SVM/...]      │
│                                              │
│  ▼ 高级参数 (可折叠)                           │
│    测试集比例: [====○====] 0.2                │
│    随机种子: [42]                             │
│    交叉验证折数: [5]                           │
│    (不同算法的专属参数...)                      │
│                                              │
│  [创建并开始训练]   [仅创建]                    │
└──────────────────────────────────────────────┘
```

### 4.2 训练详情页 (实时)

```
┌──────────────────────────────────────────────┐
│  CNN-MNIST 训练                     [暂停] [取消]
│                                              │
│  ████████████░░░░░░░░ 65.5%                   │
│  Epoch: 33/50  Step: 3300/5000               │
│                                              │
│  ┌─────────────────────┐ ┌─────────────────┐ │
│  │   Loss 曲线          │ │  Accuracy 曲线   │ │
│  │   📉 (Chart.js)     │ │  📈 (Chart.js)  │ │
│  └─────────────────────┘ └─────────────────┘ │
│                                              │
│  ┌─ 训练日志 ─────────────────────────────┐   │
│  │ [10:00] Epoch 1 - loss: 0.85 acc: 0.72 │   │
│  │ [10:02] Epoch 2 - loss: 0.62 acc: 0.81 │   │
│  │ [10:04] Epoch 3 - loss: 0.45 acc: 0.87 │   │
│  │ ← 自动滚屏                               │   │
│  └────────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

---

## 五、实验记录 (简易版)

### 5.1 存储方式
训练完成后自动保存到本地 JSON 文件（不需要 MLflow）：

```
experiments/{job_uuid}/
├── config.json          # 超参数快照
├── metrics.json         # epoch 级别指标数组
├── model/               # 模型文件
│   └── model.pkl
└── training.log         # 纯文本日志
```

### 5.2 多模型对比
模型列表页增加"对比"按钮，选中 2~4 个模型 → 并排对比表：

| 模型 | 准确率 | 精确率 | 召回率 | F1 | 训练时长 |
|------|--------|--------|--------|-----|----------|
| 随机森林 | 0.92 | 0.91 | 0.90 | 0.905 | 3分20秒 |
| SVM | 0.89 | 0.88 | 0.87 | 0.875 | 12分5秒 |

---

## 六、代码修复 (顺手修)

| 问题 | 修复 |
|------|------|
| `datetime.utcnow()` 已弃用 | 替换为 `datetime.now(datetime.UTC)` |
| `_get_current_user()` 重复3次 | 提取到 `utils/auth_helpers.py` |
| `db.create_all()` 每次启动执行 | 改用 Flask-Migrate |
| CSRF 保护缺失 | 启用 Flask-WTF 全局 CSRF |
| `import` 放在函数体内 | 移到文件顶部 |
| API Key 可从 URL 传 | 仅保留 Header 方式 |

---

## 七、实施计划（3轮）

### Round 1: 核心 — 训练执行
1. 创建 `app/executor/` 模块
2. 实现 `BaseTrainer` + `SklearnTrainer`
3. 实现 `TrainingExecutor` (线程池)
4. 改造 `TrainingService` 对接执行器
5. 改造训练创建页面 (增加模型选择)
6. 改造训练详情页面 (增加启停按钮)

### Round 2: 实时与可视化
1. 实现 SSE 流端点
2. 前端 EventSource + 进度条动画
3. Loss/Accuracy Chart.js 折线图
4. 训练日志实时滚屏
5. 实验记录 JSON 存储

### Round 3: 修复与完善
1. 代码修复 (utcnow、去重、CSRF等)
2. Flask-Migrate 替换 db.create_all()
3. 数据集自动解析 + 预览
4. 模型对比页
5. 补充测试

---

## 八、技术不变项

- 不引入任何新第三方依赖
- 数据库仍用 MySQL
- Flask + Jinja2 + Bootstrap 5 前端
- threading + SSE 做实时更新
- 本地 JSON 文件存实验记录

---

## 九、验证标准

每轮完成后，在浏览器中完成以下流程：

1. 注册/登录 → 上传 CSV 数据集 → 系统自动解析显示列信息
2. 创建训练任务 → 选数据集 → 选"分类-随机森林" → 点击开始
3. 页面实时显示进度条、Loss 曲线、日志滚动
4. 可暂停/恢复/取消训练
5. 训练完成 → 查看指标 → 模型列表中可查看详情
6. 对比两个模型的指标
