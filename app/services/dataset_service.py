"""
============================================
数据集服务
处理数据集的创建、查询、统计等业务逻辑
============================================
"""
import os
import uuid
from datetime import datetime
from typing import Optional, Tuple
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from app import db, logger
from app.models.dataset import Dataset
from app.models.user import User


class DatasetService:
    """数据集管理服务"""

    ALLOWED_EXTENSIONS = {'csv', 'json', 'txt', 'xlsx', 'parquet', 'jpg', 'png', 'npy'}

    @staticmethod
    def allowed_file(filename: str) -> bool:
        """检查文件扩展名是否允许"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in DatasetService.ALLOWED_EXTENSIONS

    @staticmethod
    def create_dataset(user: User, name: str, file: FileStorage,
                       description: str = None, category: str = 'other',
                       tags: list = None, is_public: bool = False,
                       upload_folder: str = None) -> Tuple[Optional[Dataset], Optional[str]]:
        """
        创建新数据集并上传文件

        Returns:
            (Dataset, error_message)
        """
        if not file or not file.filename:
            return None, '请选择要上传的文件。'

        if not DatasetService.allowed_file(file.filename):
            return None, f'不支持的文件格式。允许的格式: {", ".join(DatasetService.ALLOWED_EXTENSIONS)}'

        # 生成安全文件名
        original_filename = secure_filename(file.filename)
        file_ext = original_filename.rsplit('.', 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{file_ext}"

        # 确定保存路径
        if upload_folder is None:
            from flask import current_app
            upload_folder = current_app.config['UPLOAD_FOLDER']

        dataset_dir = os.path.join(upload_folder, 'datasets')
        os.makedirs(dataset_dir, exist_ok=True)

        file_path = os.path.join(dataset_dir, unique_name)

        try:
            # 保存文件
            file.save(file_path)
            file_size = os.path.getsize(file_path)

            # 创建数据库记录
            dataset = Dataset(
                name=name,
                description=description,
                file_path=file_path,
                file_size=file_size,
                file_format=file_ext,
                category=category,
                is_public=is_public,
                owner_id=user.id,
                status='ready',
            )

            if tags:
                dataset.tags_list = tags

            db.session.add(dataset)
            db.session.commit()

            logger.info(f"数据集创建成功: {name} ({file_size} bytes) by {user.username}")
            return dataset, None

        except Exception as e:
            # 清理已保存的文件
            if os.path.exists(file_path):
                os.remove(file_path)
            db.session.rollback()
            logger.error(f"数据集创建失败: {e}")
            return None, f'创建失败: {str(e)}'

    @staticmethod
    def get_dataset_by_id(dataset_id: int) -> Optional[Dataset]:
        """根据 ID 获取数据集"""
        return Dataset.query.get(dataset_id)

    @staticmethod
    def get_dataset_by_uuid(dataset_uuid: str) -> Optional[Dataset]:
        """根据 UUID 获取数据集"""
        return Dataset.query.filter_by(uuid=dataset_uuid).first()

    @staticmethod
    def update_dataset(dataset: Dataset, data: dict) -> Tuple[bool, Optional[str]]:
        """
        更新数据集元数据

        Returns:
            (success, error_message)
        """
        allowed_fields = {
            'name', 'description', 'version', 'category',
            'is_public', 'tags'
        }

        try:
            for field, value in data.items():
                if field in allowed_fields and hasattr(dataset, field):
                    if field == 'tags' and isinstance(value, list):
                        dataset.tags_list = value
                    else:
                        setattr(dataset, field, value)

            dataset.updated_at = datetime.utcnow()
            db.session.commit()
            return True, None

        except Exception as e:
            db.session.rollback()
            logger.error(f"更新数据集失败: {e}")
            return False, f'更新失败: {str(e)}'

    @staticmethod
    def delete_dataset(dataset: Dataset) -> Tuple[bool, Optional[str]]:
        """
        删除数据集及其文件

        Returns:
            (success, error_message)
        """
        try:
            # 删除物理文件
            if dataset.file_path and os.path.exists(dataset.file_path):
                os.remove(dataset.file_path)

            db.session.delete(dataset)
            db.session.commit()

            logger.info(f"数据集已删除: {dataset.name}")
            return True, None

        except Exception as e:
            db.session.rollback()
            logger.error(f"删除数据集失败: {e}")
            return False, f'删除失败: {str(e)}'

    @staticmethod
    def list_datasets(page: int = 1, per_page: int = 15,
                      category: str = None, owner_id: int = None,
                      public_only: bool = False, search: str = None) -> dict:
        """
        获取数据集列表 (支持筛选和搜索)

        Returns:
            分页结果字典
        """
        query = Dataset.query

        # 筛选条件
        if public_only:
            query = query.filter_by(is_public=True)
        if category:
            query = query.filter_by(category=category)
        if owner_id:
            query = query.filter_by(owner_id=owner_id)
        if search:
            search_term = f'%{search}%'
            query = query.filter(
                db.or_(
                    Dataset.name.ilike(search_term),
                    Dataset.description.ilike(search_term),
                    Dataset.tags.ilike(search_term),
                )
            )

        query = query.order_by(Dataset.created_at.desc())

        pagination = query.paginate(
            page=page, per_page=per_page, error_out=False
        )

        return {
            'items': [ds.to_dict() for ds in pagination.items],
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev,
        }

    @staticmethod
    def get_dataset_statistics(user_id: int = None) -> dict:
        """
        获取数据集统计信息

        Returns:
            统计数据字典
        """
        query = Dataset.query
        if user_id:
            query = query.filter_by(owner_id=user_id)

        datasets = query.all()

        total_size = sum(d.file_size for d in datasets)
        category_counts = {}
        format_counts = {}

        for d in datasets:
            category_counts[d.category] = category_counts.get(d.category, 0) + 1
            format_counts[d.file_format] = format_counts.get(d.file_format, 0) + 1

        return {
            'total_count': len(datasets),
            'total_size_bytes': total_size,
            'total_size_gb': round(total_size / (1024 ** 3), 2),
            'categories': category_counts,
            'formats': format_counts,
            'public_count': sum(1 for d in datasets if d.is_public),
        }
