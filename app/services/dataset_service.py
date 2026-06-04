"""
============================================
数据集服务
处理数据集的创建、查询、统计等业务逻辑
============================================
"""
import os
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple
from werkzeug.utils import secure_filename
from flask import current_app
from werkzeug.datastructures import FileStorage
from app import db, logger
from app.models.dataset import Dataset
from app.models.user import User


# ========== 智能分类推断 ==========

# 关键词 → 分类映射 (按优先级排序)
_CATEGORY_KEYWORDS = [
    # (关键词列表, 分类)
    (['nlp', 'text', 'language', 'corpus', 'sentiment', 'word', 'document',
      '自然语言', '文本', '语料', '情感', '词向量'], 'nlp'),
    (['vision', 'image', 'mnist', 'fashion', 'cifar', 'cifar10', 'cifar100',
      'photo', 'picture', 'pixel', '图像', '图片', '视觉', '手写'], 'vision'),
    (['time_series', 'timeseries', 'temporal', 'stock', 'weather', 'temperature',
      '时序', '时间序列', '股票', '天气', '传感器', 'sensor'], 'time_series'),
    (['regression', 'reg', 'price', 'housing', 'california', 'boston',
      'predict', 'forecast', '回归', '房价', '预测'], 'regression'),
    (['cluster', 'clustering', 'blob', 'segment', 'group',
      '聚类', '分群', '分段'], 'clustering'),
    (['biology', 'bio', 'gene', 'cancer', 'breast', 'diabetes', 'disease',
      'medical', 'health', 'patient', 'cell', 'protein', 'dna',
      '医疗', '生物', '基因', '癌症', '疾病', '糖尿病'], 'biology'),
    (['finance', 'fin', 'credit', 'loan', 'bank', 'income', 'census', 'adult',
      'economic', 'financial', 'payment', 'salary', 'revenue',
      '金融', '经济', '信贷', '银行', '收入', '贷款'], 'finance'),
    (['synthetic', 'syn', 'generate', 'make_class', 'make_reg', 'artificial',
      'fake', 'simulated', '合成', '生成', '人工', '模拟'], 'synthetic'),
    (['classification', 'class', 'classify', 'binary', 'multiclass',
      'label', 'target', 'category',
      '分类', '二分类', '多分类', '标签'], 'classification'),
]


def _infer_category(name: str, df, file_ext: str) -> str:
    """
    智能推断数据集分类

    推断优先级:
    1. 文件名关键词匹配
    2. 列名关键词匹配
    3. 数据特征 (列数/类型/目标列)
    4. 兜底: tabular (表格) 或 other
    """
    name_lower = name.lower().replace('_', ' ').replace('-', ' ')

    # 1. 文件名关键词匹配
    for keywords, category in _CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in name_lower:
                return category

    # 2. 列名关键词匹配
    if df is not None:
        cols_lower = ' '.join(str(c).lower() for c in df.columns)
        for keywords, category in _CATEGORY_KEYWORDS:
            for kw in keywords:
                if kw in cols_lower:
                    return category

        # 3. 数据特征推断
        num_cols = len(df.select_dtypes(include=['number']).columns)
        total_cols = len(df.columns)

        # 图像数据: 大量列+像素列名
        pixel_cols = sum(1 for c in df.columns
                        if str(c).lower().startswith(('pixel', 'px')))
        if pixel_cols > 10:
            return 'vision'

    # 4. 图像文件格式
    if file_ext in ('jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff'):
        return 'vision'

    # 5. 文本文件
    if file_ext in ('txt', 'json', 'jsonl'):
        return 'nlp'

    # 6. 兜底: 通用表格
    if file_ext in ('csv', 'xlsx', 'xls', 'parquet'):
        return 'tabular'

    return 'other'


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
            db.session.flush()  # 获取 dataset.id

            # 自动解析文件统计信息
            try:
                _analyze_dataset_file(dataset, file_path, file_ext)
            except Exception as parse_err:
                logger.warning(f"数据集自动解析失败 (非致命): {parse_err}")

            db.session.commit()

            logger.info(f"数据集创建成功: {name} ({file_size} bytes, "
                        f"{dataset.row_count}行x{dataset.column_count}列) by {user.username}")
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
        return db.session.get(Dataset, dataset_id)

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

            dataset.updated_at = datetime.now(timezone.utc)
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


# ============ 数据集文件自动解析 ============

def _analyze_dataset_file(dataset, file_path: str, file_ext: str):
    """自动解析数据集文件，提取行列数、列名等统计信息"""
    import pandas as pd

    if file_ext == 'csv':
        df = pd.read_csv(file_path)
    elif file_ext in ('xlsx', 'xls'):
        df = pd.read_excel(file_path)
    elif file_ext == 'json':
        df = pd.read_json(file_path)
    elif file_ext == 'parquet':
        df = pd.read_parquet(file_path)
    elif file_ext == 'txt':
        df = pd.read_csv(file_path, sep='\t', nrows=1000)
    else:
        return

    dataset.row_count = len(df)
    dataset.column_count = len(df.columns)

    # 构建摘要：列名 + 类型 + 缺失值
    import json
    summary = {
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0},
        'sample_rows': 5,
    }
    dataset.summary_json = json.dumps(summary, ensure_ascii=False)

    # 智能推断分类 — 基于数据集名称+列名+内容
    dataset.category = _infer_category(dataset.name, df, file_ext)

    dataset.status = 'ready'
