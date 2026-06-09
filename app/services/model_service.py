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
            logger.error(f"模型注册失败: {e}")
            return None, f'注册失败: {str(e)}'

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
            logger.error(f"模型文件上传失败: {e}")
            return False, f'上传失败: {str(e)}'

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
            logger.error(f"更新模型指标失败: {e}")
            return False, str(e)

    @staticmethod
    def get_model_by_id(model_id: int) -> Optional[ModelRecord]:
        """根据 ID 获取模型"""
        return db.session.get(ModelRecord, model_id)

    @staticmethod
    def get_model_by_uuid(model_uuid: str) -> Optional[ModelRecord]:
        """根据 UUID 获取模型"""
        return ModelRecord.query.filter_by(uuid=model_uuid).first()

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
            logger.error(f"更新模型失败: {e}")
            return False, str(e)

    @staticmethod
    def delete_model(model: ModelRecord) -> Tuple[bool, Optional[str]]:
        """删除模型及其文件 — 先解除关联训练任务的外键约束"""
        try:
            from app.models.training_job import TrainingJob

            # 解除关联的训练任务引用 (避免外键约束报错)
            TrainingJob.query.filter_by(model_id=model.id).update(
                {TrainingJob.model_id: None}
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
            logger.error(f"删除模型失败: {e}")
            return False, str(e)

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
        query = ModelRecord.query

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
        """获取模型统计数据 (带缓存, TTL=60s)"""
        cache_key = f'model_stats:{user_id or "all"}'
        cached = dashboard_cache.get(cache_key)
        if cached is not None:
            return cached

        query = ModelRecord.query
        if user_id:
            query = query.filter_by(owner_id=user_id)

        models = query.all()

        type_counts = {}
        status_counts = {}
        framework_counts = {}

        for m in models:
            type_counts[m.model_type] = type_counts.get(m.model_type, 0) + 1
            status_counts[m.status] = status_counts.get(m.status, 0) + 1
            if m.framework:
                framework_counts[m.framework] = framework_counts.get(m.framework, 0) + 1

        # 公开/私有计数
        public_count = sum(1 for m in models if m.is_public)

        # 计算平均准确率
        accuracies = [m.accuracy for m in models if m.accuracy is not None]

        result = {
            'total_count': len(models),
            'deployed_count': status_counts.get('deployed', 0),
            'public_count': public_count,
            'private_count': len(models) - public_count,
            'types': type_counts,
            'statuses': status_counts,
            'frameworks': framework_counts,
            'avg_accuracy': round(sum(accuracies) / len(accuracies), 4) if accuracies else None,
            'max_accuracy': round(max(accuracies), 4) if accuracies else None,
        }
        dashboard_cache.set(cache_key, result)
        return result
