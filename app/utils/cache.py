"""
============================================
TTL 内存缓存工具 v2.0
基于 dict + 自动过期清理 + 容量限制 + 统计
============================================
"""
import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import Any


class TTLCache:
    """TTL 内存缓存 — 多 worker 安全的轻量级缓存

    特性:
        - TTL 自动过期 (get 时惰性清理 + 后台定期清理)
        - max_size 容量限制 (LRU-like 驱逐)
        - 线程安全 (可重入锁)
        - 命中/未命中/驱逐统计
        - 前缀批量清除

    使用方式:
        cache = TTLCache(default_ttl=60, max_size=500)

        @cache.cached(key='dashboard_stats', ttl=30)
        def get_stats():
            return heavy_query()

        # 查看统计:
        stats = cache.stats  # {'hits': 42, 'misses': 5, 'size': 3, 'evictions': 0}
    """

    def __init__(self, default_ttl: int = 60, max_size: int = 0,
                 cleanup_interval: int = 60):
        """初始化缓存

        Args:
            default_ttl: 默认过期秒数
            max_size: 最大条目数 (0 = 无限制)
            cleanup_interval: 后台清理间隔秒数 (0 = 禁用后台清理)
        """
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self.default_ttl = default_ttl
        self.max_size = max_size

        # 统计
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # 后台清理线程
        self._cleanup_thread: threading.Thread | None = None
        self._cleanup_stop = threading.Event()
        if cleanup_interval > 0:
            self._start_cleanup_thread(cleanup_interval)

    # ── 核心 API ──────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """获取缓存值，过期返回 None"""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """设置缓存值 (超过 max_size 时驱逐最旧的条目)"""
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.monotonic() + ttl
        with self._lock:
            # 容量检查: 驱逐最早过期的条目
            if self.max_size > 0 and len(self._store) >= self.max_size and key not in self._store:
                self._evict_oldest()
            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> bool:
        """删除缓存, 返回是否成功"""
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._store.clear()

    def invalidate(self, prefix: str) -> int:
        """按前缀批量清除, 返回清除数"""
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]
            return len(keys)

    def has(self, key: str) -> bool:
        """检查键是否存在且未过期"""
        return self.get(key) is not None

    # ── 装饰器 ───────────────────────────────────────

    def cached(self, key: str = None, ttl: int = None):
        """装饰器: 缓存函数返回值

        Args:
            key: 缓存键 (默认 func.__qualname__)
            ttl: TTL 秒数
        """
        def decorator(func: Callable):
            cache_key = key or f'{func.__module__}.{func.__qualname__}'

            @wraps(func)
            def wrapper(*args, **kwargs):
                result = self.get(cache_key)
                if result is not None:
                    return result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl)
                return result

            wrapper.cache_invalidate = lambda: self.delete(cache_key)
            wrapper.cache_refresh = lambda: (
                self.delete(cache_key),
                self.set(cache_key, func(), ttl=ttl)
            )
            return wrapper
        return decorator

    # ── 统计 ─────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """返回缓存统计 (线程安全)"""
        with self._lock:
            total = self._hits + self._misses
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': round(self._hits / max(total, 1), 4),
                'size': len(self._store),
                'max_size': self.max_size,
                'evictions': self._evictions,
            }

    def reset_stats(self) -> None:
        """重置统计计数器"""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    # ── 内部方法 ─────────────────────────────────────

    def _evict_oldest(self) -> None:
        """驱逐最早过期的条目 (LRU-like)"""
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k][1])
        del self._store[oldest_key]
        self._evictions += 1

    def _cleanup_expired(self) -> None:
        """清理所有过期条目"""
        now = time.monotonic()
        with self._lock:
            expired = [k for k, (_, exp) in self._store.items() if now > exp]
            for k in expired:
                del self._store[k]

    def _start_cleanup_thread(self, interval: int) -> None:
        """启动后台定期清理线程 (daemon, 不阻止进程退出)"""
        def _cleanup_loop():
            while not self._cleanup_stop.wait(interval):
                try:
                    self._cleanup_expired()
                except Exception:
                    pass  # 清理失败不应导致线程崩溃

        self._cleanup_thread = threading.Thread(
            target=_cleanup_loop, daemon=True, name='ttl-cache-cleanup'
        )
        self._cleanup_thread.start()

    def __del__(self):
        """析构: 停止清理线程"""
        self._cleanup_stop.set()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: str) -> bool:
        return self.has(key)


# ============ 全局单例 ============

# 仪表盘统计缓存 (TTL=60s, 高频访问 + 低频变更, 上限 200)
dashboard_cache = TTLCache(default_ttl=60, max_size=200)

# 排行榜缓存 (TTL=300s, 低频变更, 上限 50)
leaderboard_cache = TTLCache(default_ttl=300, max_size=50)

# 数据集推荐缓存 (TTL=600s, 极低频变更, 上限 100)
recommendation_cache = TTLCache(default_ttl=600, max_size=100)
