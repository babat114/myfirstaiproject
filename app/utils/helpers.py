"""
============================================
辅助工具函数
通用的格式转换、验证和工具方法
============================================
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

# 北京时区 (UTC+8) — 保留供外部引用
BEIJING_TZ = timezone(timedelta(hours=8))

# 统一使用 _timezone 模块的 localnow，避免重复实现
import contextlib

from app._timezone import localnow  # noqa: E402, F401 — re-export for backward compatibility


def format_file_size(size_bytes: int) -> str:
    """将字节数转换为人类可读的文件大小"""
    if size_bytes == 0:
        return '0 B'

    units = ['B', 'KB', 'MB', 'GB', 'TB']
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1

    return f'{size:.2f} {units[i]}'


def format_datetime(dt: datetime, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
    """格式化日期时间"""
    if dt is None:
        return '-'
    return dt.strftime(fmt)


def format_duration(seconds: float) -> str:
    """将秒数格式化为人类可读的时长"""
    if seconds < 60:
        return f'{int(seconds)}秒'
    if seconds < 3600:
        mins = int(seconds / 60)
        secs = int(seconds % 60)
        return f'{mins}分{secs}秒'

    hours = int(seconds / 3600)
    mins = int((seconds % 3600) / 60)
    return f'{hours}小时{mins}分钟'


def generate_file_hash(filepath: str, algorithm: str = 'sha256') -> str | None:
    """计算文件的哈希值"""
    try:
        h = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def safe_json_loads(json_str: str | None, default: Any = None) -> Any:
    """安全地解析 JSON 字符串"""
    if not json_str:
        return default
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """安全地将对象序列化为 JSON 字符串"""
    defaults = {'ensure_ascii': False, 'default': str}
    defaults.update(kwargs)
    return json.dumps(obj, **defaults)


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """截断文本到指定长度"""
    if not text:
        return ''
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_status_color(status: str) -> str:
    """根据状态返回对应的 Bootstrap 颜色类"""
    colors = {
        'ready': 'success',
        'uploading': 'warning',
        'processing': 'info',
        'error': 'danger',
        'draft': 'secondary',
        'trained': 'primary',
        'deployed': 'success',
        'archived': 'dark',
        'failed': 'danger',
        'queued': 'info',
        'preparing': 'warning',
        'running': 'primary',
        'paused': 'warning',
        'completed': 'success',
        'cancelled': 'secondary',
    }
    return colors.get(status, 'secondary')


def get_status_icon(status: str) -> str:
    """根据状态返回对应的 Bootstrap Icons 类名"""
    icons = {
        'ready': 'bi-check-circle-fill',
        'uploading': 'bi-cloud-upload-fill',
        'processing': 'bi-gear-fill',
        'error': 'bi-x-circle-fill',
        'draft': 'bi-pencil-fill',
        'trained': 'bi-mortarboard-fill',
        'deployed': 'bi-rocket-takeoff-fill',
        'archived': 'bi-archive-fill',
        'failed': 'bi-exclamation-triangle-fill',
        'queued': 'bi-hourglass-split',
        'preparing': 'bi-tools',
        'running': 'bi-play-circle-fill',
        'paused': 'bi-pause-circle-fill',
        'completed': 'bi-check-circle-fill',
        'cancelled': 'bi-x-octagon-fill',
    }
    return icons.get(status, 'bi-question-circle-fill')


def chart_colors() -> list:
    """返回图表颜色调色板"""
    return [
        '#4e73df', '#1cc88a', '#36b9cc', '#f6c23e',
        '#e74a3b', '#858796', '#5a5c69', '#2e59d9',
        '#17a673', '#2c9faf',
    ]


def parse_form_params(form_data: dict, int_fields: set = None, float_fields: set = None,
                     str_fields: set = None) -> dict:
    """解析表单参数为强类型 dict — 减少路由层重复的类型转换代码

    Args:
        form_data: request.form 字典 (ImmutableMultiDict.to_dict())
        int_fields: 整数类型字段名集合
        float_fields: 浮点类型字段名集合
        str_fields: 字符串类型字段名集合 (显式列出的不会被自动推断)

    Returns:
        {field_name: typed_value} — 未识别的字段会尝试智能推断, 空值会被跳过
    """
    int_fields = int_fields or set()
    float_fields = float_fields or set()
    str_fields = str_fields or set()

    result = {}
    for key, val in form_data.items():
        if val is None or val == '':
            continue
        key_lower = key.lower()
        if key_lower in str_fields or key_lower == 'hidden_layers_str':
            result[key] = val
        elif key_lower in float_fields:
            try:
                result[key] = float(val)
            except (ValueError, TypeError):
                pass  # 跳过无效数值
        elif key_lower in int_fields:
            with contextlib.suppress(ValueError, TypeError):
                result[key] = int(val)
        else:
            # 保持字符串 — 避免误转 "00123" → 123、ID字段丢失前导零
            result[key] = val
    return result


def to_python_type(val, recurse: bool = False):
    """将 numpy/pandas 标量转换为 Python 原生类型 (安全 JSON 序列化)

    Args:
        val: 任意值 (可能包含 numpy scalar / pandas Timestamp 等)
        recurse: 是否递归转换 list/dict 中的值

    Returns:
        Python 原生类型值 (int/float/bool/str/list/dict/None)
    """
    if val is None:
        return None
    # numpy scalar (np.int64, np.float32, etc.) — 有 .item() 方法
    try:
        if hasattr(val, 'item'):
            val = val.item()
    except Exception:
        pass
    # pandas Timestamp
    if hasattr(val, 'isoformat') and not isinstance(val, str):
        return val.isoformat()
    # 递归转换
    if recurse:
        if isinstance(val, list):
            return [to_python_type(v, recurse=True) for v in val]
        if isinstance(val, dict):
            return {str(k): to_python_type(v, recurse=True) for k, v in val.items()}
    # 基础类型
    if isinstance(val, bool):
        return bool(val)
    if isinstance(val, (int, float)):
        return val
    return val


def sanitize_error(error: Exception | str, fallback: str = '操作失败，请稍后重试。') -> str:
    """安全错误消息: 生产环境返回通用消息, 开发环境返回详细信息

    防止将数据库错误/文件路径/栈追踪泄露给客户端。

    Args:
        error: Exception 或 字符串错误消息
        fallback: 生产环境返回的通用消息

    Returns:
        安全的错误消息字符串
    """
    try:
        from flask import current_app
        if current_app.config.get('DEBUG'):
            return str(error)
    except RuntimeError:
        pass
    return fallback


def sanitize_service_error(error: Exception, log_message: str = None) -> str:
    """Service 层专用: 记录详细错误到日志, 返回脱敏消息给客户端

    用法:
        except Exception as e:
            return False, sanitize_service_error(e, '删除数据集失败')

    Args:
        error: 捕获的异常
        log_message: 日志前缀 (可选)

    Returns:
        脱敏后的客户端消息
    """
    from app import logger as _logger
    detail = str(error)
    if log_message:
        _logger.error(f'{log_message}: {detail}')
    else:
        _logger.error(detail)
    return sanitize_error(error)


def paginate_query(query, page: int = 1, per_page: int = 20,
                   item_key: str = 'items',
                   transform_fn: callable = None) -> dict:
    """通用分页辅助 — 消除 5 个 Service 中重复的分页返回构造逻辑

    支持两种查询对象:
      - flask_sqlalchemy.BaseQuery (Model.query 风格): query.paginate(...)
      - sqlalchemy.Select (stmt 风格, 由 db.paginate(...) 处理)

    Args:
        query: Flask-SQLAlchemy Query 或 SQLAlchemy Select 对象
        page: 当前页码
        per_page: 每页条数
        item_key: 返回字典中 items 列表的键名 (如 'comments', 'users')
        transform_fn: 对每个 item 应用的转换函数 (如 Model.to_dict), None 则直接返回

    Returns:
        {'items': [...], 'total': int, 'pages': int, 'current_page': int, 'has_next': bool, 'has_prev': bool}
    """
    from sqlalchemy import Select

    from app import db as _app_db

    if isinstance(query, Select):
        pagination = _app_db.paginate(query, page=page, per_page=per_page, error_out=False)
    else:
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    items = pagination.items
    items = [transform_fn(i) for i in items] if transform_fn else list(items)

    return {
        item_key: items,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'has_next': pagination.has_next,
        'has_prev': pagination.has_prev,
    }
