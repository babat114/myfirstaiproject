"""
============================================
内存 TTL 缓存工具
基于 dict + 过期时间的轻量级缓存
============================================
"""
import time
import threading
from functools import wraps
from typing import Any, Callable, Optional


class TTLCache:
    """简单的 TTL 内存缓存

    使用方式:
        cache = TTLCache(default_ttl=60)

        @cache.cached(key='dashboard_stats', ttl=30)
        def get_stats():
            return heavy_query()

        # 或手动使用:
        stats = cache.get('dashboard_stats')
        if stats is None:
            stats = heavy_query()
            cache.set('dashboard_stats', stats, ttl=30)
    """

    def __init__(self, default_ttl: int = 60):
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值，过期返回 None"""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """设置缓存值"""
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.monotonic() + ttl
        with self._lock:
            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        """删除缓存"""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._store.clear()

    def invalidate(self, prefix: str) -> int:
        """按前缀批量清除"""
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]
            return len(keys)

    def cached(self, key: str = None, ttl: int = None):
        """装饰器: 缓存函数返回值

        Args:
            key: 缓存键 (默认使用 func.__name__ + args/kwargs)
            ttl: TTL 秒数 (默认使用实例 default_ttl)
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = key or f'{func.__module__}.{func.__qualname__}:{args}:{kwargs}'
                result = self.get(cache_key)
                if result is not None:
                    return result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl)
                return result
            # 附加手动清除方法
            wrapper.cache_invalidate = lambda: self.delete(key) if key else None
            return wrapper
        return decorator


# ============ 全局单例 ============

# 仪表盘统计缓存 (TTL=60s, 高频访问 + 低频变更)
dashboard_cache = TTLCache(default_ttl=60)

# 排行榜缓存 (TTL=300s, 低频变更)
leaderboard_cache = TTLCache(default_ttl=300)
