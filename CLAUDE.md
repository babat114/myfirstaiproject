# myfirstaiproject — AI模型训练管理平台

> 完整 Skill 见 `.claude/SKILL.md`

## 启动

```bash
cd ~/Desktop/myfirstaiproject && python run.py
```
→ http://127.0.0.1:5000 | admin/Admin123456 | researcher1/User123456

## 关键架构

- **认证**: Session(Web) → JWT Bearer(API) → API Key(兼容)
- **训练器**: sklearn / PyTorch / TensorFlow.Keras / Transformers (BERT NLP)
- **测试**: 375个pytest, `python -m pytest tests/ -q`
- **数据集**: 55个 (全真实数据集, 合成已清理), 支持多来源自动采集
- **模型**: 138个 (all trained, 0 draft), sklearn:133, pytorch:5
- **质量看门狗**: `scripts/quality_watchdog.py` — 定期自动扫描质量, `diagnose_model_quality.py` Tier 1/2/3分级

## GPU 环境

- **硬件**: NVIDIA RTX 4060 Laptop (8GB VRAM)
- **PyTorch**: 2.6.0+cu124, `torch.cuda.is_available() == True`
- **TensorFlow**: 2.20.0 (Windows CPU-only, GPU 需 WSL2)
- 训练器自动检测 GPU, 无需手动配置

## 超参数调优 (v9)

| 模式 | 方法 | 适用场景 |
|------|------|---------|
| **GridSearchCV** | 穷举 + CV | 参数组合少 |
| **RandomSearchCV** | 随机采样 + CV | 参数空间大 |
| **AutoML** | 遍历所有算法 → 每算法快速 RandomSearch | 不确定用哪个算法 |

- 聚类不用 CV: 改用手动 ParameterGrid 遍历 + 全量 fit + 子采样 silhouette_score
- SSE 实时进度: `TuningProgressTracker` 单例, 500ms 推送
- KMeans `algorithm` 参数冲突三重防护 (lloyd/elkan vs kmeans)
- **random_state**: 所有调优方法默认 `None` (真随机), 传入 `int` 种子可复现 (v9 修复)

## AI 智能优化面板

训练详情页三个 Tab:
- **AI 诊断**: 自动分析训练结果 (健康度评分/问题检测/参数建议)
- **GridSearchCV 单算法调优**: SSE 实时进度 + 一键应用最佳参数
- **AutoML 全算法对比**: 遍历所有算法排名, 可选用任一算法

## NLP 质量修复 (v12, Batch A-F 全部完成)

- 80个过拟合模型已清理 (clean_overfit_models.py), 117个NLP模型重训练完成
- 162测试通过, 质量分级: A级62 / B级46 / C级9
- 新增: argparse并行训练 / 交叉验证 / 数据增强 / 推理健壮性修复

## 重要约定

- Windows GBK 环境, 避免 emoji 用 ASCII
- 登录表单字段是 `login_id` 不是 `username`
- SQLAlchemy 2.0: `db.session.get(Model, id)` 不用 `Model.query.get(id)`
- numpy 兼容: `.tolist()` 替代 `.numpy()`, 显式 dtype
- **居中通知**: `showCenteredMessage(msg, type, dur)` 替代 ElMessage
- **确认弹窗**: `[data-confirm]` 属性 或 `showConfirmModal()` 替代原生 confirm
- **数据集类别感知**: auto-config API 会检查 dataset.category 并发出警告
- **聚类调优**: 不应对无监督学习使用 CV; 用 `_manual_clustering_search()`
- **best_params 合并**: 必须跳过 `algorithm`, `ml_task_type`, `task_type`, `framework` 等冲突键
- **TTL 缓存**: `app/utils/cache.py` v2.0 — 后台定期清理 + max_size LRU驱逐 + hit/miss统计
- **表单解析**: `parse_form_params()` 统一路由层参数类型转换，减少重复代码
- **前端美化 v4.0**: `beautify.css/js` — Canvas粒子网络 + 滚动揭示 + 3D卡片倾斜 + 涟漪按钮
- **TF-IDF 特征保护**: StandardScaler 不可应用于 `tfidf_*` 列 — TfidfVectorizer 已做 L2 归一化, 叠加会破坏稀疏性导致常数预测器
- **合并分词器**: `nlp_preprocessing.combined_tokenize()` — jieba词分割 + 中文字符unigram, 确保短文本 nnz>0
- **TF .keras 推理**: `inference_service.py` 支持 `.keras`/`.h5` 加载, config 从 `model_config.pkl` 读取; predict() 使用 `model.predict()` 路径 (softmax 已在末层)
- **模型质量治理**: `quality_watchdog.py` (扫描/守护/自动清理) + `diagnose_model_quality.py` (7维诊断) + `clean_overfit_models.py` (批量安全删除, `--dedup`去重) — Tier 1/2/3 分级

## 自动 Skill 调度

编辑代码后自动调用 `/code-review-expert` (SOLID/安全/性能审查)。
涉及以下场景时主动调用对应 Skill 后再回答:

## 自动 Git 提交约定

每次修改代码后自动执行 `git add -A && git commit -m "描述"`，确保每个变更都可回退。
异常关机/强制重启后按 `git log --oneline -5` 查看最近版本，用 `git reset --hard <hash>` 回滚。

| 场景 | 自动调用的 Skill |
|------|-----------------|
| 代码审查/PR | code-review-expert, pr-reviewer |
| Django/ORM/Python | django-expert, django-doctor, python-pro |
| 数据库/SQL/查询优化 | database-optimizer, sql-pro, postgres-pro |
| API 设计 | api-designer, fastapi-expert |
| 架构设计 | architecture-designer |
| 前端改动 | react-expert, vue-expert, typescript-pro |
| 测试编写 | test-master, playwright-expert |
| Bug 修复 | debugging-wizard, code-review-expert |
| ML/数据处理 | ml-pipeline, pandas-pro |
| 安全相关 | secure-code-guardian, security-reviewer |
| DevOps/部署 | devops-engineer, kubernetes-specialist |
