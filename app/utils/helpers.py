"""
============================================
辅助工具函数
通用的格式转换、验证和工具方法
============================================
"""
import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any

# 北京时区 (UTC+8) — 保留供外部引用
BEIJING_TZ = timezone(timedelta(hours=8))

# 统一使用 _timezone 模块的 localnow，避免重复实现
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
    except (IOError, OSError):
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
