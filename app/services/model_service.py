"""
============================================
AI模型服务
处理模型注册、版本管理、性能追踪
============================================
"""
import os
import uuid
from datetime import datetime
from typing import Optional, Tuple
from app._timezone import localnow
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from flask import current_app
from app import db, logger
from app.models.model_record import ModelRecord
from app.models.user import User
from app.utils.cache import dashboard_cache, leaderboard_cache
from app.utils.helpers import sanitize_service_error
from sqlalchemy.orm import joinedload


# ═══════════════════════════════════════════════════════════════
# 模型卡片生成 — 安全转义工具 (提取自 generate_model_card)
# ═══════════════════════════════════════════════════════════════

def _escape_yaml_value(s: str) -> str:
    """转义 YAML 值中的特殊字符, 防止 YAML 注入"""
    if not s:
        return s
    dangerous = ('{', '}', '[', ']', '&', '*', '!', '|', '>', "'", '"', '%', '@', '`')
    if '\n' in s or ': ' in s or (s and s[0] in dangerous):
        escaped = s.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'
    return s


def _escape_bibtex_value(s: str) -> str:
    """转义 BibTeX 值中的特殊字符, 防止 BibTeX 注入"""
    if not s:
        return s
    # 注意: \\ 必须最先转义, 否则会破坏后续 \{ \} 等转义序列
    for ch in [('\\', '\\textbackslash{}'), ('{', '\\{'), ('}', '\\}'),
                ('$', '\\$'), ('&', '\\&'), ('#', '\\#'),
                ('%', '\\%'), ('_', '\\_'), ('~', '\\~{}'), ('^', '\\^{}')]:
        s = s.replace(ch[0], ch[1])
    return s


class ModelService:
    """AI模型注册管理服务"""

    ALLOWED_EXTENSIONS = {'pt', 'pth', 'h5', 'pb', 'onnx', 'pkl', 'joblib', 'json', 'yaml'}

    @staticmethod
    def allowed_file(filename: str) -> bool:
        """检查模型文件扩展名"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ModelService.ALLOWED_EXTENSIONS

    @staticmethod
    def create_model(user: User, name: str, model_type: str = 'other',
                     framework: str = None, description: str = None,
                     version: str = '1.0.0', hyperparameters: dict = None,
                     is_public: bool = False) -> Tuple[Optional[ModelRecord], Optional[str]]:
        """
        注册新AI模型

        Returns:
            (ModelRecord, error_message)
        """
        try:
            model = ModelRecord(
                name=name,
                description=description,
                version=version,
                model_type=model_type,
                framework=framework,
                is_public=is_public,
                owner_id=user.id,
                status='draft',
            )

            if hyperparameters:
                model.set_hyperparameters(hyperparameters)

            db.session.add(model)
            db.session.commit()
            dashboard_cache.invalidate('model_stats:')
            dashboard_cache.invalidate('dashboard:')

            logger.info(f"模型注册成功: {name} v{version} by {user.username}")
            return model, None

        except Exception as e:
            db.session.rollback()
            return None, sanitize_service_error(e, '模型注册失败')

    @staticmethod
    def upload_model_file(model: ModelRecord, file: FileStorage,
                          upload_folder: str = None) -> Tuple[bool, Optional[str]]:
        """
        上传模型权重文件

        Returns:
            (success, error_message)
        """
        if not ModelService.allowed_file(file.filename):
            return False, f'不支持的模型文件格式。'

        if upload_folder is None:
            upload_folder = current_app.config['UPLOAD_FOLDER']

        model_dir = os.path.join(upload_folder, 'models')
        os.makedirs(model_dir, exist_ok=True)

        original_name = secure_filename(file.filename)
        unique_name = f"{model.uuid}_{original_name}"
        file_path = os.path.join(model_dir, unique_name)

        try:
            file.save(file_path)
            file_size = os.path.getsize(file_path)

            model.model_file_path = file_path
            model.file_size = file_size
            model.status = 'trained'
            model.updated_at = localnow()

            db.session.commit()

            logger.info(f"模型文件上传成功: {model.name} ({file_size} bytes)")
            return True, None

        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            db.session.rollback()
            return False, sanitize_service_error(e, '模型文件上传失败')

    @staticmethod
    def update_metrics(model: ModelRecord, metrics: dict) -> Tuple[bool, Optional[str]]:
        """
        更新模型性能指标

        Returns:
            (success, error_message)
        """
        try:
            model.set_metrics(metrics)
            model.updated_at = localnow()
            db.session.commit()
            return True, None
        except Exception as e:
            db.session.rollback()
            return False, sanitize_service_error(e, '更新模型指标失败')

    @staticmethod
    def get_model_by_id(model_id: int) -> Optional[ModelRecord]:
        """根据 ID 获取模型"""
        return db.session.get(ModelRecord, model_id)

    @staticmethod
    def get_model_by_uuid(model_uuid: str) -> Optional[ModelRecord]:
        """根据 UUID 获取模型"""
        return db.session.execute(
            db.select(ModelRecord).filter_by(uuid=model_uuid)
        ).scalar_one_or_none()

    @staticmethod
    def update_model(model: ModelRecord, data: dict) -> Tuple[bool, Optional[str]]:
        """更新模型信息"""
        allowed_fields = {
            'name', 'description', 'version', 'model_type',
            'framework', 'is_public', 'status'
        }
        try:
            for field, value in data.items():
                if field in allowed_fields and hasattr(model, field):
                    setattr(model, field, value)

            if 'hyperparameters' in data and isinstance(data['hyperparameters'], dict):
                model.set_hyperparameters(data['hyperparameters'])

            model.updated_at = localnow()
            db.session.commit()
            dashboard_cache.invalidate('model_stats:')
            dashboard_cache.invalidate('dashboard:')
            leaderboard_cache.clear()
            return True, None

        except Exception as e:
            db.session.rollback()
            return False, sanitize_service_error(e, '更新模型失败')

    @staticmethod
    def delete_model(model: ModelRecord) -> Tuple[bool, Optional[str]]:
        """删除模型及其文件 — 先解除关联训练任务的外键约束"""
        try:
            from app.models.training_job import TrainingJob

            # 解除关联的训练任务引用 (避免外键约束报错) — SA 2.0 bulk update
            db.session.execute(
                db.update(TrainingJob)
                .where(TrainingJob.model_id == model.id)
                .values(model_id=None)
            )

            # 删除关联文件
            for path_attr in ['model_file_path', 'weights_file_path', 'config_file_path']:
                path = getattr(model, path_attr)
                if path and os.path.exists(path):
                    os.remove(path)

            db.session.delete(model)
            db.session.commit()
            dashboard_cache.invalidate('model_stats:')
            dashboard_cache.invalidate('dashboard:')
            leaderboard_cache.clear()

            logger.info(f"模型已删除: {model.name}")
            return True, None

        except Exception as e:
            db.session.rollback()
            logger.error(f'删除模型失败: {e}')
            return False, sanitize_service_error(e, '删除模型失败')

    # 允许排序的列名白名单 (防SQL注入)
    _SORTABLE_COLUMNS = {
        'accuracy', 'precision', 'recall', 'f1_score', 'loss',
        'r2', 'mse', 'mae', 'created_at', 'updated_at', 'name',
    }

    @staticmethod
    def list_models(page: int = 1, per_page: int = 15,
                    model_type: str = None, framework: str = None,
                    owner_id: int = None, status: str = None,
                    search: str = None, is_public: bool = None,
                    include_public: bool = False,
                    sort_by: str = 'created_at', sort_order: str = 'desc') -> dict:
        """
        获取模型列表 (支持多条件筛选 + 排序)

        Args:
            is_public: 按公开/私有筛选 (None=全部, True=仅公开, False=仅私有)
            include_public: 当设置了 owner_id 时，同时包含其他用户的公开模型
            sort_by: 排序字段 (accuracy/precision/recall/f1_score/loss/r2/mse/mae/created_at/updated_at/name)
            sort_order: 排序方向 (asc/desc)

        Returns:
            分页结果字典
        """
        query = ModelRecord.query.options(joinedload(ModelRecord.owner))

        if model_type:
            query = query.filter_by(model_type=model_type)
        if framework:
            query = query.filter_by(framework=framework)
        if owner_id:
            if include_public:
                # 用户自己的模型 + 其他用户的公开模型
                query = query.filter(
                    db.or_(
                        ModelRecord.owner_id == owner_id,
                        ModelRecord.is_public == True  # noqa: E712
                    )
                )
            else:
                query = query.filter_by(owner_id=owner_id)
        if status:
            query = query.filter_by(status=status)
        if is_public is not None:
            query = query.filter_by(is_public=is_public)
        if search:
            term = f'%{search}%'
            query = query.filter(
                db.or_(
                    ModelRecord.name.ilike(term),
                    ModelRecord.description.ilike(term),
                )
            )

        # 排序 (白名单校验)
        sort_by = sort_by if sort_by in ModelService._SORTABLE_COLUMNS else 'created_at'
        sort_order = sort_order if sort_order in ('asc', 'desc') else 'desc'
        sort_column = getattr(ModelRecord, sort_by, ModelRecord.created_at)

        # NULL值排在最后 (升序时NULL最小，降序时NULL...都需要排最后)
        if sort_order == 'asc':
            query = query.order_by(sort_column.is_(None), sort_column.asc())
        else:
            query = query.order_by(sort_column.is_(None), sort_column.desc())

        pagination = query.paginate(
            page=page, per_page=per_page, error_out=False
        )

        return {
            'items': [m.to_dict() for m in pagination.items],
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev,
        }

    @staticmethod
    def get_top_models(limit: int = 5, metric: str = 'accuracy') -> list:
        """
        获取性能最佳的模型 (带缓存, TTL=300s)
        """
        cache_key = f'leaderboard:{limit}:{metric}'
        cached = leaderboard_cache.get(cache_key)
        if cached is not None:
            return cached

        sort_column = getattr(ModelRecord, metric, ModelRecord.accuracy)
        models = ModelRecord.query \
            .options(joinedload(ModelRecord.owner)) \
            .filter(ModelRecord.status.in_(['trained', 'deployed'])) \
            .filter(sort_column.isnot(None)) \
            .order_by(sort_column.desc()) \
            .limit(limit) \
            .all()

        result = [m.to_dict() for m in models]
        leaderboard_cache.set(cache_key, result)
        return result

    @staticmethod
    def get_model_statistics(user_id: int = None) -> dict:
        """获取模型统计数据 (带缓存, TTL=60s)

        使用 SQL GROUP BY 聚合, O(k) 内存 (k=类别数), 而非 O(n) 全量加载。
        """
        from sqlalchemy import func

        cache_key = f'model_stats:{user_id or "all"}'
        cached = dashboard_cache.get(cache_key)
        if cached is not None:
            return cached

        def _filtered_query(*cols):
            q = db.select(*cols)
            if user_id:
                q = q.filter_by(owner_id=user_id)
            return q

        # 按模型类型聚合
        type_rows = db.session.execute(
            _filtered_query(ModelRecord.model_type, func.count(ModelRecord.id))
            .group_by(ModelRecord.model_type)
        ).all()
        type_counts = {row[0]: row[1] for row in type_rows}

        # 按状态聚合
        status_rows = db.session.execute(
            _filtered_query(ModelRecord.status, func.count(ModelRecord.id))
            .group_by(ModelRecord.status)
        ).all()
        status_counts = {row[0]: row[1] for row in status_rows}

        # 按框架聚合 (过滤 NULL)
        framework_rows = db.session.execute(
            _filtered_query(ModelRecord.framework, func.count(ModelRecord.id))
            .filter(ModelRecord.framework.isnot(None))
            .group_by(ModelRecord.framework)
        ).all()
        framework_counts = {row[0]: row[1] for row in framework_rows}

        # 聚合指标: 总数/公开数/已部署数/平均准确率/最高准确率
        agg_row = db.session.execute(
            _filtered_query(
                func.count(ModelRecord.id),
                func.sum(db.case((ModelRecord.is_public == True, 1), else_=0)),
                func.sum(db.case((ModelRecord.status == 'deployed', 1), else_=0)),
                func.avg(ModelRecord.accuracy),
                func.max(ModelRecord.accuracy),
            )
        ).one()
        total_count, public_count, deployed_count, avg_acc, max_acc = agg_row

        result = {
            'total_count': total_count,
            'deployed_count': deployed_count or 0,
            'public_count': public_count or 0,
            'private_count': total_count - (public_count or 0),
            'types': type_counts,
            'statuses': status_counts,
            'frameworks': framework_counts,
            'avg_accuracy': round(float(avg_acc), 4) if avg_acc is not None else None,
            'max_accuracy': round(float(max_acc), 4) if max_acc is not None else None,
        }
        dashboard_cache.set(cache_key, result)
        return result

    # ═══════════════════════════════════════════════════════════════
    # HuggingFace 风格模型卡片生成 (v2: 上帝方法拆分为3个方法)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _prepare_card_context(model: ModelRecord) -> dict:
        """Pre-compute all data needed for the model card.

        Extracts hyperparameters, metrics, features, class labels,
        and builds all markdown tables. Returns a context dict
        that is passed to the rendering methods.

        This method was extracted from generate_model_card to
        separate data preparation from string assembly (M1 refactor).
        """
        hp = model.hyperparameters_dict
        mm = model.metrics_dict
        fw = model.framework or 'unknown'
        task = model.model_type or 'other'
        algo = hp.get('algorithm', '')
        now_str = localnow().strftime('%Y-%m-%d')

        # ── 任务类型中英文映射 ──
        task_labels = {
            'classification': 'Classification',
            'regression': 'Regression',
            'clustering': 'Clustering',
            'nlp': 'Natural Language Processing',
            'computer_vision': 'Computer Vision',
            'reinforcement': 'Reinforcement Learning',
            'generative': 'Generative',
            'other': 'Other',
        }
        task_label = task_labels.get(task, task)

        # ── 收集标签 ──
        tags = [task_label, fw]
        if algo:
            tags.append(algo)
        tags_str = '\n  - '.join(tags)

        # ── 提取特征名 (最多 30 个) ──
        feature_names = mm.get('feature_names', [])
        if not feature_names:
            try:
                from app.services.inference_service import ModelInferenceService
                _, metadata, _, _ = ModelInferenceService.load_model(model)
                if metadata and metadata.get('feature_names'):
                    feature_names = list(metadata['feature_names'])
            except Exception:
                pass

        # ── 类别标签 (分类模型) ──
        class_labels = []
        label_encoders = mm.get('label_encoders', {})
        target_le = label_encoders.get('__target__')
        if target_le and hasattr(target_le, 'classes_'):
            class_labels = list(target_le.classes_)

        # ── 构建指标表 ──
        metrics_rows = ModelService._build_metrics_rows(task, model, mm)

        if not metrics_rows:
            metrics_rows.append('| *No metrics recorded* | - |')

        metrics_table = '\n'.join(metrics_rows)

        # ── 构建特征表 (最多 20 个) ──
        feature_table = ModelService._build_feature_table(feature_names)

        # ── 构建类别表 (分类模型) ──
        class_table = ModelService._build_class_table(class_labels)

        # ── 构建超参数表 ──
        hp_table = ModelService._build_hyperparams_table(hp)

        # ── 数据集信息 ──
        dataset_info = ModelService._build_dataset_info(model)

        # ── 模型 UUID (URL-friendly) ──
        model_uuid = model.uuid

        # ── 预计算嵌入文本 (避免 f-string 中嵌套引号) ──
        algo_suffix = f" using the `{algo}` algorithm" if algo else ""
        nlp_hint = " or text" if task == 'nlp' else ""

        # ── 安全字符串 ──
        safe_name = _escape_yaml_value(model.name)
        safe_owner = _escape_yaml_value(model.owner.username if model.owner else 'unknown')
        safe_bibtex_name = _escape_bibtex_value(model.name)
        safe_bibtex_owner = _escape_bibtex_value(
            model.owner.username if model.owner else 'AI Platform'
        )
        safe_bibtex_fw = _escape_bibtex_value(fw)
        safe_bibtex_task = _escape_bibtex_value(task)

        return {
            'hp': hp, 'mm': mm, 'fw': fw, 'task': task, 'algo': algo,
            'now_str': now_str, 'task_label': task_label, 'tags_str': tags_str,
            'feature_names': feature_names, 'class_labels': class_labels,
            'metrics_table': metrics_table,
            'feature_table': feature_table, 'class_table': class_table,
            'hp_table': hp_table, 'dataset_info': dataset_info,
            'model_uuid': model_uuid, 'algo_suffix': algo_suffix,
            'nlp_hint': nlp_hint,
            'safe_name': safe_name, 'safe_owner': safe_owner,
            'safe_bibtex_name': safe_bibtex_name, 'safe_bibtex_owner': safe_bibtex_owner,
            'safe_bibtex_fw': safe_bibtex_fw, 'safe_bibtex_task': safe_bibtex_task,
            'model': model,
        }

    @staticmethod
    def _build_metrics_rows(task: str, model: ModelRecord,
                            mm: dict) -> list:
        """Build individual metric row strings for the evaluation table."""
        rows = []
        if task == 'classification':
            if model.accuracy is not None:
                rows.append(f'| Accuracy | {model.accuracy * 100:.2f}% |')
            if model.precision is not None:
                rows.append(f'| Precision (macro) | {model.precision:.4f} |')
            if model.recall is not None:
                rows.append(f'| Recall (macro) | {model.recall:.4f} |')
            if model.f1_score is not None:
                rows.append(f'| F1 Score (macro) | {model.f1_score:.4f} |')
        elif task == 'regression':
            if model.r2 is not None:
                rows.append(f'| R² | {model.r2:.4f} |')
            if model.mse is not None:
                rows.append(f'| MSE | {model.mse:.4f} |')
            if model.mae is not None:
                rows.append(f'| MAE | {model.mae:.4f} |')
        elif task == 'clustering':
            sc = mm.get('test_silhouette_score', mm.get('silhouette_score'))
            db_idx = mm.get('test_davies_bouldin_score', mm.get('davies_bouldin_score'))
            ch = mm.get('test_calinski_harabasz_score', mm.get('calinski_harabasz_score'))
            if sc is not None:
                rows.append(f'| Silhouette Score | {sc:.4f} |')
            if db_idx is not None:
                rows.append(f'| Davies-Bouldin Index | {db_idx:.4f} |')
            if ch is not None:
                rows.append(f'| Calinski-Harabasz Index | {ch:.2f} |')

        if model.loss is not None:
            rows.append(f'| Loss | {model.loss:.4f} |')
        return rows

    @staticmethod
    def _build_feature_table(feature_names: list) -> str:
        """Build the feature listing markdown table."""
        if not feature_names:
            return ''
        shown = feature_names[:20]
        table = '| # | Feature |\n|---|--------|\n'
        table += '\n'.join(
            f'| {i + 1} | `{name}` |'
            for i, name in enumerate(shown)
        )
        if len(feature_names) > 20:
            table += f'\n| ... | *{len(feature_names) - 20} more features* |'
        return table

    @staticmethod
    def _build_class_table(class_labels: list) -> str:
        """Build the class labels markdown table."""
        if not class_labels:
            return ''
        table = '\n| Index | Class |\n|-------|-------|\n'
        table += '\n'.join(
            f'| {i} | `{c}` |' for i, c in enumerate(class_labels)
        )
        return table

    @staticmethod
    def _build_hyperparams_table(hp: dict) -> str:
        """Build the hyperparameters markdown table."""
        if not hp:
            return ''
        table = '| Parameter | Value |\n|-----------|-------|\n'
        show_hp = {k: v for k, v in hp.items()
                   if not k.startswith('_') and k not in ('task_type', 'ml_task_type', 'tuned')}
        table += '\n'.join(
            f'| `{k}` | {v} |' for k, v in sorted(show_hp.items())
        )
        return table

    @staticmethod
    def _build_dataset_info(model: ModelRecord) -> str:
        """Build the dataset info string for the training data section."""
        if not model.training_dataset:
            return ''
        ds = model.training_dataset
        info = f'**{ds.name}**'
        if ds.description:
            info += f' — {ds.description[:200]}'
        return info

    @staticmethod
    def _render_card_frontmatter(ctx: dict) -> str:
        """Render the YAML frontmatter section of the model card."""
        model = ctx['model']
        return f'''---
language: zh
tags:
  - {ctx['tags_str']}
license: mit
model_name: {ctx['safe_name']}
version: {model.version}
framework: {ctx['fw']}
task_type: {ctx['task']}
created_by: {ctx['safe_owner']}
created_at: {model.created_at.strftime('%Y-%m-%d') if model.created_at else '-'}
updated_at: {model.updated_at.strftime('%Y-%m-%d') if model.updated_at else '-'}
pipeline_tag: {ctx['task'].replace('_', '-')}
---
'''

    @staticmethod
    def _render_card_body(ctx: dict) -> str:
        """Render the main body sections of the model card.

        Broken into logical sub-sections: description, intended use,
        training data/procedure, evaluation, architecture, usage,
        limitations, and citation.
        """
        model = ctx['model']
        fw = ctx['fw']
        task = ctx['task']
        task_label = ctx['task_label']
        algo = ctx['algo']
        feature_names = ctx['feature_names']
        model_uuid = ctx['model_uuid']
        now_str = ctx['now_str']

        parts = []

        # ── 模型描述 ──
        parts.append(f'''
# {model.name}

> **Version**: {model.version} | **Framework**: {fw} | **Task**: {task_label}
> **Created**: {model.created_at.strftime('%Y-%m-%d') if model.created_at else '-'} | **Status**: {model.status}

## Model Description

{model.description or '*No description provided.*'}

This is a **{task_label}** model built with **{fw}**{ctx['algo_suffix']}.
The model has been trained and evaluated on the AI Model Training Platform, and is ready
for inference via the platform API or as a standalone Docker container.
''')

        # ── 预期用途 ──
        parts.append(ModelService._render_intended_use_section(
            task_label, fw, algo, feature_names, ctx['nlp_hint']
        ))

        # ── 训练数据 + 过程 ──
        parts.append(ModelService._render_training_section(
            ctx['dataset_info'], ctx['hp_table'], model
        ))

        # ── 评估结果 ──
        parts.append(ModelService._render_evaluation_section(
            ctx['metrics_table'], ctx['mm']
        ))

        # ── 模型架构 + 特征 ──
        parts.append(ModelService._render_architecture_section(
            fw, algo, model, ctx['feature_table'], ctx['class_table']
        ))

        # ── 使用方法 ──
        parts.append(ModelService._render_usage_section(
            model_uuid, task, feature_names
        ))

        # ── 局限性 ──
        parts.append(ModelService._render_limitations_section(fw))

        # ── 引用 ──
        parts.append(ModelService._render_citation_section(
            model, now_str, ctx['safe_bibtex_name'], ctx['safe_bibtex_owner'],
            ctx['safe_bibtex_fw'], ctx['safe_bibtex_task']
        ))

        return ''.join(parts)

    @staticmethod
    def _render_intended_use_section(task_label: str, fw: str, algo: str,
                                     feature_names: list, nlp_hint: str) -> str:
        """Render the Intended Use section."""
        section = f'''
## Intended Use

- **Primary Task**: {task_label}
- **Framework**: {fw}'''
        if algo:
            section += f'\n- **Algorithm**: `{algo}`'
        if feature_names:
            section += f'\n- **Input Features**: {len(feature_names)}'

        section += f'''

### Use Cases

This model is suitable for {task_label.lower()} tasks. It can be integrated into
applications requiring automated predictions based on structured data{nlp_hint}.

### Out-of-Scope

- Production-critical systems without additional validation
- Domains significantly different from the training data distribution
- Real-time safety-critical applications
'''
        return section

    @staticmethod
    def _render_training_section(dataset_info: str, hp_table: str,
                                 model: ModelRecord) -> str:
        """Render the Training Data + Training Procedure section."""
        section = f'''
## Training Data

{dataset_info if dataset_info else '*Training dataset information is not available.*'}

## Training Procedure

### Hyperparameters'''

        if hp_table:
            section += f'''

{hp_table}'''
        else:
            section += '\n\n*No hyperparameter information available.*'

        section += f'''

### Training Duration

'''
        if model.training_duration_seconds is not None:
            secs = model.training_duration_seconds
            if secs < 60:
                section += f'{secs} seconds'
            elif secs < 3600:
                section += f'{secs // 60} min {secs % 60} sec'
            else:
                section += f'{secs // 3600} h {(secs % 3600) // 60} min'
        else:
            section += '*Not recorded*'

        return section

    @staticmethod
    def _render_evaluation_section(metrics_table: str, mm: dict) -> str:
        """Render the Evaluation Results section."""
        section = f'''
## Evaluation Results

| Metric | Value |
|--------|-------|
{metrics_table}'''

        if mm.get('test_accuracy') is not None:
            section += f'\n| Test Accuracy | {mm["test_accuracy"] * 100:.2f}% |'
        if mm.get('test_precision_macro') is not None:
            section += f'\n| Test Precision (macro) | {mm["test_precision_macro"]:.4f} |'
        if mm.get('test_recall_macro') is not None:
            section += f'\n| Test Recall (macro) | {mm["test_recall_macro"]:.4f} |'
        if mm.get('test_f1_macro') is not None:
            section += f'\n| Test F1 (macro) | {mm["test_f1_macro"]:.4f} |'

        return section

    @staticmethod
    def _render_architecture_section(fw: str, algo: str, model: ModelRecord,
                                     feature_table: str, class_table: str) -> str:
        """Render the Model Architecture + Input Features section."""
        section = f'''
## Model Architecture

- **Framework**: {fw}'''

        if algo:
            section += f'\n- **Algorithm**: `{algo}`'
        if model.file_size:
            section += f'\n- **File Size**: {model.file_size_mb} MB'

        section += f'''

## Input Features'''

        if feature_table:
            section += f'''

{feature_table}'''
        else:
            section += '\n\n*Feature information is not available.*'

        if class_table:
            section += f'''

## Class Labels

{class_table}'''

        return section

    @staticmethod
    def _render_usage_section(model_uuid: str, task: str,
                              feature_names: list) -> str:
        """Render the How to Use section (Python + curl + Docker)."""
        n_features = min(len(feature_names) if feature_names else 4, 4)
        sample_features = ', '.join(['1.0'] * n_features)

        return f'''

## How to Use

### Python (API)

```python
import requests

url = "http://localhost:5000/api/v1/models/{model_uuid}/serve"
data = {{"features": [[{sample_features}]]}}

headers = {{"Authorization": "Bearer YOUR_API_TOKEN"}}
resp = requests.post(url, json=data, headers=headers)
print(resp.json())
# {{"predictions": [...], "task_type": "{task}"}}
```

### cURL

```bash
curl -X POST http://localhost:5000/api/v1/models/{model_uuid}/serve \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_TOKEN" \\
  -d '{{"features": [[{sample_features}]]}}'
```

### Docker Deployment

```bash
# Build and run the Docker container
docker-compose up -d

# Check health
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"features": [[{sample_features}]]}}'
```
'''

    @staticmethod
    def _render_limitations_section(fw: str) -> str:
        """Render the Limitations section."""
        return f'''
## Limitations

1. **Training Data Bias**: The model may reflect biases present in the training data.
   Always evaluate on your specific use case before deployment.
2. **Domain Shift**: Performance may degrade when applied to data that differs
   significantly from the training distribution.
3. **No Guarantees**: This model is provided as-is without guarantees of fitness
   for any particular purpose.
4. **Feature Dependencies**: The model requires exactly the input features listed
   above. Missing or extra features may cause errors or degraded performance.
5. **Version Compatibility**: The model was trained with specific library versions;
   using different versions of {fw} may affect inference results.
'''

    @staticmethod
    def _render_citation_section(model: ModelRecord, now_str: str,
                                 safe_bibtex_name: str, safe_bibtex_owner: str,
                                 safe_bibtex_fw: str, safe_bibtex_task: str) -> str:
        """Render the Citation + Footer section."""
        return f'''
## Citation

If you use this model in your research or application, please cite:

```bibtex
@misc{{{model.name_slug}-{model.version}}},
  title = {{{safe_bibtex_name}}},
  author = {{{safe_bibtex_owner}}},
  year = {{{model.created_at.year if model.created_at else localnow().year}}},
  publisher = {{AI Model Training Platform}},
  version = {{{model.version}}},
  framework = {{{safe_bibtex_fw}}},
  task = {{{safe_bibtex_task}}},
}}
```

---
*Model card generated on {now_str} by AI Model Training Platform*
*Format inspired by HuggingFace model cards*
'''

    @staticmethod
    def generate_model_card(model: ModelRecord) -> str:
        """生成 HuggingFace 风格的模型卡片 (Markdown)

        包含:
            - YAML 元数据头
            - 模型描述
            - 预期用途与局限性
            - 训练过程 (超参数/框架)
            - 评估结果 (指标)
            - 使用方法 (Python + curl 示例)
            - 局限性声明

        Args:
            model: ModelRecord 实例

        Returns:
            Markdown 格式的模型卡片字符串
        """
        ctx = ModelService._prepare_card_context(model)
        frontmatter = ModelService._render_card_frontmatter(ctx)
        body = ModelService._render_card_body(ctx)
        return frontmatter + body
