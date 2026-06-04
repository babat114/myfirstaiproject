# myfirstaiproject — AI模型训练管理平台

> 完整 Skill 见 `.claude/SKILL.md`

## 启动

```bash
cd ~/Desktop/myfirstaiproject && python run.py
```
→ http://127.0.0.1:5000 | admin/Admin123456 | researcher1/User123456

## 关键架构

- **认证**: Session(Web) → JWT Bearer(API) → API Key(兼容)
- **训练器**: sklearn / PyTorch / TensorFlow.Keras
- **测试**: 77个pytest, `python -m pytest tests/ -v`
- **数据集**: 100个, 支持15个公开数据集导入
- **模型**: 113个已训练, 5种框架

## 重要约定

- Windows GBK 环境, 避免 emoji 用 ASCII
- 登录表单字段是 `login_id` 不是 `username`
- SQLAlchemy 2.0: `db.session.get(Model, id)` 不用 `Model.query.get(id)`
- numpy 兼容: `.tolist()` 替代 `.numpy()`, 显式 dtype
- **居中通知**: `showCenteredMessage(msg, type, dur)` 替代 ElMessage
- **确认弹窗**: `[data-confirm]` 属性 或 `showConfirmModal()` 替代原生 confirm
- **数据集类别感知**: auto-config API 会检查 dataset.category 并发出警告
