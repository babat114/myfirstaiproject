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
from app._timezone import localnow
from werkzeug.utils import secure_filename
from flask import current_app
from werkzeug.datastructures import FileStorage
from app import db, logger
from app.models.dataset import Dataset
from app.models.user import User
from app.utils.cache import dashboard_cache
from app.utils.helpers import paginate_query, sanitize_service_error
from sqlalchemy.orm import joinedload


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

    # MIME 类型白名单 (与允许的扩展名对应)
    ALLOWED_MIMETYPES = {
        'text/csv', 'application/csv', 'text/plain',
        'application/json',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
        'application/octet-stream',  # parquet
        'image/jpeg', 'image/png',
        'application/x-npy',
    }

    @staticmethod
    def allowed_file(filename: str, file_storage=None) -> bool:
        """检查文件扩展名和 MIME 类型是否允许

        Args:
            filename: 文件名
            file_storage: (可选) Flask FileStorage 对象, 传入时同时校验 MIME 类型

        Returns:
            是否允许上传
        """
        # 1. 扩展名检查
        ext_ok = '.' in filename and \
                 filename.rsplit('.', 1)[1].lower() in DatasetService.ALLOWED_EXTENSIONS
        if not ext_ok:
            return False

        # 2. MIME 类型检查 (如提供 file_storage)
        if file_storage is not None:
            mime = (getattr(file_storage, 'content_type', '') or '').lower()
            if not mime:
                # Content-Type 为空 — 无法验证 MIME, 但扩展名已通过检查
                # 在调试模式下记录警告, 生产模式下允许通过 (保守策略)
                pass
            elif mime not in DatasetService.ALLOWED_MIMETYPES:
                # 部分 CSV 文件可能被识别为 application/csv
                if not mime.startswith('text/') and mime != 'application/json':
                    return False

        return True

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

        if not DatasetService.allowed_file(file.filename, file_storage=file):
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

            # 清除统计缓存
            dashboard_cache.invalidate('dataset_stats:')
            dashboard_cache.invalidate('dashboard:')

            logger.info(f"数据集创建成功: {name} ({file_size} bytes, "
                        f"{dataset.row_count}行x{dataset.column_count}列) by {user.username}")
            return dataset, None

        except Exception as e:
            # 清理已保存的文件
            if os.path.exists(file_path):
                os.remove(file_path)
            db.session.rollback()
            return None, sanitize_service_error(e, '数据集创建失败')

    @staticmethod
    def get_dataset_by_id(dataset_id: int) -> Optional[Dataset]:
        """根据 ID 获取数据集"""
        return db.session.get(Dataset, dataset_id)

    @staticmethod
    def get_dataset_by_uuid(dataset_uuid: str) -> Optional[Dataset]:
        """根据 UUID 获取数据集"""
        return db.session.execute(
            db.select(Dataset).filter_by(uuid=dataset_uuid)
        ).scalar_one_or_none()

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

            dataset.updated_at = localnow()
            db.session.commit()
            dashboard_cache.invalidate('dataset_stats:')
            dashboard_cache.invalidate('dashboard:')
            return True, None

        except Exception as e:
            db.session.rollback()
            return False, sanitize_service_error(e, '更新数据集失败')

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
            dashboard_cache.invalidate('dataset_stats:')
            dashboard_cache.invalidate('dashboard:')

            logger.info(f"数据集已删除: {dataset.name}")
            return True, None

        except Exception as e:
            db.session.rollback()
            return False, sanitize_service_error(e, '删除数据集失败')

    @staticmethod
    def copy_dataset_to_user(dataset: Dataset, user: User) -> Tuple[Optional[Dataset], Optional[str]]:
        """
        将公开数据集复制到目标用户的名下

        复制物理文件 + 创建新 Dataset 记录 (owner=user, is_public=False)
        如果用户已拥有同名数据集，自动追加后缀避免冲突

        Returns:
            (new_dataset, error_message)
        """
        import shutil

        # 检查是否已经复制过 (同名 + 同 owner)
        existing = db.session.execute(
            db.select(Dataset).filter_by(name=dataset.name, owner_id=user.id)
        ).scalar_one_or_none()
        if existing:
            return existing, None  # 已存在，直接返回

        # 生成新文件名 (避免冲突)
        file_ext = dataset.file_format
        new_uuid = uuid.uuid4().hex
        unique_filename = f"{new_uuid}.{file_ext}"

        # 确定保存路径
        try:
            from flask import current_app
            upload_folder = current_app.config['UPLOAD_FOLDER']
        except RuntimeError:
            upload_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'uploads')

        dataset_dir = os.path.join(upload_folder, 'datasets')
        os.makedirs(dataset_dir, exist_ok=True)
        new_file_path = os.path.join(dataset_dir, unique_filename)

        try:
            # 复制物理文件 (始终独立副本, 避免硬链接数据完整性风险)
            if os.path.exists(dataset.file_path):
                shutil.copy2(dataset.file_path, new_file_path)

            # 创建新数据集记录
            new_dataset = Dataset(
                name=dataset.name,
                description=dataset.description,
                file_path=new_file_path,
                file_size=dataset.file_size,
                file_format=dataset.file_format,
                category=dataset.category,
                is_public=False,  # 复制品默认私有
                owner_id=user.id,
                status='ready',
                row_count=dataset.row_count,
                column_count=dataset.column_count,
                summary_json=dataset.summary_json,
                tags=dataset.tags,
            )

            db.session.add(new_dataset)
            db.session.commit()

            dashboard_cache.invalidate('dataset_stats:')
            dashboard_cache.invalidate('dashboard:')

            logger.info(f"数据集已复制: {dataset.name} -> {user.username} (id={new_dataset.id})")
            return new_dataset, None

        except Exception as e:
            # 清理已复制的文件
            if os.path.exists(new_file_path):
                os.remove(new_file_path)
            db.session.rollback()
            return None, sanitize_service_error(e, '复制数据集失败')

    @staticmethod
    def list_datasets(page: int = 1, per_page: int = 15,
                      category: str = None, owner_id: int = None,
                      public_only: bool = False, include_public: bool = False,
                      search: str = None) -> dict:
        """
        获取数据集列表 (支持筛选和搜索)

        Args:
            include_public: 当设置了 owner_id 时，同时包含其他用户的公开数据集
                           用于训练创建页，允许用户选择公开数据集进行训练

        Returns:
            分页结果字典
        """
        query = Dataset.query.options(joinedload(Dataset.owner))

        # 筛选条件
        if public_only:
            query = query.filter_by(is_public=True)
        if category:
            query = query.filter_by(category=category)
        if owner_id:
            if include_public:
                # 用户自己的数据集 + 其他用户的公开数据集
                from sqlalchemy import or_
                query = query.filter(
                    or_(Dataset.owner_id == owner_id, Dataset.is_public == True)  # noqa: E712
                )
            else:
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

        return paginate_query(query, page, per_page, transform_fn=lambda x: x.to_dict())

    @staticmethod
    def get_dataset_statistics(user_id: int = None) -> dict:
        """
        获取数据集统计信息 (带缓存, TTL=60s)

        使用 SQL GROUP BY 聚合, O(k) 内存 (k=类别数), 而非 O(n) 全量加载。
        """
        from sqlalchemy import func, or_

        cache_key = f'dataset_stats:{user_id or "all"}'
        cached = dashboard_cache.get(cache_key)
        if cached is not None:
            return cached

        def _filtered_query(*cols):
            q = db.select(*cols)
            if user_id:
                q = q.filter_by(owner_id=user_id)
            return q

        # 按类别聚合
        category_rows = db.session.execute(
            _filtered_query(Dataset.category, func.count(Dataset.id))
            .group_by(Dataset.category)
        ).all()
        category_counts = {row[0]: row[1] for row in category_rows}

        # 按格式聚合
        format_rows = db.session.execute(
            _filtered_query(Dataset.file_format, func.count(Dataset.id))
            .group_by(Dataset.file_format)
        ).all()
        format_counts = {row[0]: row[1] for row in format_rows}

        # 总计/总大小/公开数 — 单次查询
        agg_row = db.session.execute(
            _filtered_query(
                func.count(Dataset.id),
                func.coalesce(func.sum(Dataset.file_size), 0),
                func.sum(db.case((Dataset.is_public == True, 1), else_=0)),
            )
        ).one()
        total_count, total_size, public_count = agg_row[0], agg_row[1], agg_row[2]

        result = {
            'total_count': total_count,
            'total_size_bytes': total_size,
            'total_size_gb': round(total_size / (1024 ** 3), 2),
            'categories': category_counts,
            'formats': format_counts,
            'public_count': public_count or 0,
        }
        dashboard_cache.set(cache_key, result)
        return result


# ============ 数据集文件自动解析 ============

def _analyze_dataset_file(dataset, file_path: str, file_ext: str):
    """自动解析数据集文件，提取行列数、列名等统计信息"""
    from app.utils.data_io import load_dataframe

    df = load_dataframe(file_path, file_ext)
    if df is None:
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
