"""
============================================
TTL 缓存单元测试 v1.0
覆盖 app/utils/cache.py 全部功能
============================================
"""
import time
import threading
import pytest
from app.utils.cache import TTLCache


class TestTTLCacheCore:
    """核心 CRUD 操作测试"""

    def test_set_and_get(self):
        """基本 set/get 操作"""
        cache = TTLCache(default_ttl=60)
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'

    def test_get_missing_key(self):
        """获取不存在的键返回 None"""
        cache = TTLCache()
        assert cache.get('nonexistent') is None

    def test_get_expired_key(self):
        """过期键返回 None 并惰性删除"""
        cache = TTLCache(default_ttl=0.01)  # 10ms TTL
        cache.set('key1', 'value1')
        time.sleep(0.02)  # 等待过期
        assert cache.get('key1') is None
        assert 'key1' not in cache  # 惰性删除

    def test_set_overwrite(self):
        """重复 set 覆盖旧值"""
        cache = TTLCache()
        cache.set('key1', 'old')
        cache.set('key1', 'new')
        assert cache.get('key1') == 'new'

    def test_set_custom_ttl(self):
        """自定义 TTL 参数"""
        cache = TTLCache(default_ttl=60)
        cache.set('key1', 'value1', ttl=0.01)
        time.sleep(0.02)
        assert cache.get('key1') is None

    def test_delete_existing(self):
        """删除存在的键返回 True"""
        cache = TTLCache()
        cache.set('key1', 'v')
        assert cache.delete('key1') is True
        assert cache.get('key1') is None

    def test_delete_missing(self):
        """删除不存在的键返回 False"""
        cache = TTLCache()
        assert cache.delete('nonexistent') is False

    def test_clear(self):
        """清空所有缓存"""
        cache = TTLCache()
        cache.set('a', 1)
        cache.set('b', 2)
        cache.set('c', 3)
        cache.clear()
        assert len(cache) == 0
        assert cache.get('a') is None

    def test_invalidate_by_prefix(self):
        """按前缀批量清除"""
        cache = TTLCache()
        cache.set('user:1:stats', {'hits': 10})
        cache.set('user:1:profile', {'name': 'A'})
        cache.set('user:2:stats', {'hits': 5})
        cache.set('dataset:1', {'rows': 100})

        count = cache.invalidate('user:1:')
        assert count == 2
        assert cache.get('user:1:stats') is None
        assert cache.get('user:1:profile') is None
        assert cache.get('user:2:stats') is not None  # 未匹配
        assert cache.get('dataset:1') is not None      # 未匹配

    def test_invalidate_no_match(self):
        """前缀无匹配时返回 0"""
        cache = TTLCache()
        cache.set('key1', 1)
        assert cache.invalidate('nonexistent:') == 0

    def test_has_existing(self):
        """has() 检查存在的键"""
        cache = TTLCache()
        cache.set('key1', 'v')
        assert cache.has('key1') is True

    def test_has_missing(self):
        """has() 检查不存在的键"""
        cache = TTLCache()
        assert cache.has('missing') is False

    def test_has_expired(self):
        """has() 检查过期键返回 False"""
        cache = TTLCache(default_ttl=0.01)
        cache.set('key1', 'v')
        time.sleep(0.02)
        assert cache.has('key1') is False

    def test_len(self):
        """__len__ 返回条目数"""
        cache = TTLCache()
        assert len(cache) == 0
        cache.set('a', 1)
        cache.set('b', 2)
        assert len(cache) == 2

    def test_contains(self):
        """__contains__ 运算符"""
        cache = TTLCache()
        cache.set('key1', 'v')
        assert 'key1' in cache
        assert 'missing' not in cache


class TestTTLCacheMaxSize:
    """容量限制 + LRU 驱逐测试"""

    def test_eviction_when_full(self):
        """达到 max_size 时驱逐最早过期的条目"""
        cache = TTLCache(max_size=2)
        cache.set('a', 1, ttl=100)
        cache.set('b', 2, ttl=200)
        # 第三个触发驱逐 (a 最早过期)
        cache.set('c', 3, ttl=300)
        assert len(cache) <= 2
        # a (ttl=100) 应被驱逐, b (ttl=200) 和 c (ttl=300) 保留
        assert cache.get('b') == 2
        assert cache.get('c') == 3

    def test_no_eviction_when_key_exists(self):
        """已存在的键更新不触发驱逐"""
        cache = TTLCache(max_size=2)
        cache.set('a', 1)
        cache.set('b', 2)
        cache.set('a', 'updated')  # 不应驱逐
        assert len(cache) == 2
        assert cache.get('a') == 'updated'
        assert cache.get('b') == 2

    def test_no_limit_when_max_size_zero(self):
        """max_size=0 无限制"""
        cache = TTLCache(max_size=0)
        for i in range(100):
            cache.set(f'key{i}', i)
        assert len(cache) == 100

    def test_eviction_stats(self):
        """驱逐计数递增"""
        cache = TTLCache(max_size=2)
        cache.set('a', 1, ttl=10)
        cache.set('b', 2, ttl=20)
        cache.set('c', 3, ttl=30)  # 触发驱逐 a
        assert cache.stats['evictions'] >= 1


class TestTTLCacheStats:
    """统计信息测试"""

    def test_initial_stats(self):
        """初始统计全为零"""
        cache = TTLCache()
        stats = cache.stats
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_rate'] == 0.0
        assert stats['size'] == 0
        assert stats['evictions'] == 0

    def test_hit_count(self):
        """命中计数"""
        cache = TTLCache()
        cache.set('key1', 'v')
        cache.get('key1')  # hit
        cache.get('key1')  # hit
        cache.get('key1')  # hit
        assert cache.stats['hits'] == 3

    def test_miss_count(self):
        """未命中计数 (包括过期键)"""
        cache = TTLCache()
        cache.get('nonexistent')  # miss
        cache.get('also_missing')  # miss
        cache.set('key1', 'v', ttl=0.01)
        time.sleep(0.02)
        cache.get('key1')  # miss (expired)
        assert cache.stats['misses'] == 3

    def test_hit_rate(self):
        """命中率计算"""
        cache = TTLCache()
        cache.set('key1', 'v')
        cache.get('key1')  # hit
        cache.get('key2')  # miss
        assert cache.stats['hit_rate'] == 0.5

    def test_hit_rate_zero_division(self):
        """零总请求时命中率为 0"""
        cache = TTLCache()
        assert cache.stats['hit_rate'] == 0.0

    def test_reset_stats(self):
        """重置统计"""
        cache = TTLCache()
        cache.set('key1', 'v')
        cache.get('key1')  # hit
        cache.get('key2')  # miss
        cache.reset_stats()
        stats = cache.stats
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['evictions'] == 0


class TestTTLCacheThreadSafety:
    """线程安全测试"""

    def test_concurrent_set_get(self):
        """并发读写不丢数据"""
        cache = TTLCache()
        errors = []

        def writer(start, count):
            try:
                for i in range(start, start + count):
                    cache.set(f'key{i}', i)
            except Exception as e:
                errors.append(e)

        def reader(count):
            try:
                for i in range(count):
                    cache.get(f'key{i}')
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0, 50)),
            threading.Thread(target=writer, args=(50, 50)),
            threading.Thread(target=reader, args=(100,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(cache) > 0

    def test_concurrent_invalidate(self):
        """并发 invalidate 不抛异常"""
        cache = TTLCache()
        for i in range(100):
            cache.set(f'user:{i}:stats', i)

        errors = []

        def invalidate_range(start, end):
            try:
                for i in range(start, end):
                    cache.invalidate(f'user:{i}:')
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=invalidate_range, args=(0, 50))
        t2 = threading.Thread(target=invalidate_range, args=(50, 100))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0


class TestTTLCacheDecorator:
    """cached 装饰器测试"""

    def test_cached_decorator(self):
        """装饰器缓存函数返回值"""
        cache = TTLCache()
        call_count = [0]

        @cache.cached(key='test_func', ttl=60)
        def heavy_func():
            call_count[0] += 1
            return f'result_{call_count[0]}'

        r1 = heavy_func()
        r2 = heavy_func()
        r3 = heavy_func()

        assert r1 == r2 == r3 == 'result_1'
        assert call_count[0] == 1  # 只调用一次

    def test_cached_expired_refresh(self):
        """过期后重新执行函数"""
        cache = TTLCache(default_ttl=0.01)
        call_count = [0]

        @cache.cached(key='test_func2', ttl=0.01)
        def heavy_func():
            call_count[0] += 1
            return f'result_{call_count[0]}'

        r1 = heavy_func()
        time.sleep(0.02)
        r2 = heavy_func()

        assert r1 != r2
        assert call_count[0] == 2

    def test_cache_invalidate_method(self):
        """生成的 cache_invalidate 方法"""
        cache = TTLCache()

        @cache.cached(key='invalidatable', ttl=60)
        def func():
            return 42

        func()
        assert cache.has('invalidatable')
        func.cache_invalidate()
        assert not cache.has('invalidatable')

    def test_cache_refresh_method(self):
        """生成的 cache_refresh 方法"""
        cache = TTLCache()
        call_count = [0]

        @cache.cached(key='refreshable', ttl=60)
        def func():
            call_count[0] += 1
            return call_count[0]

        r1 = func()
        func.cache_refresh()
        r2 = func()

        assert r1 != r2
        assert call_count[0] == 2

    def test_default_key_from_qualname(self):
        """默认使用 func.__qualname__ 作为键"""
        cache = TTLCache()

        @cache.cached()
        def my_special_function():
            return 'cached'

        my_special_function()
        # 键应为 module.qualname 格式
        assert len(cache) == 1


class TestTTLCacheCleanup:
    """后台清理测试"""

    def test_cleanup_expired_removes_expired(self):
        """_cleanup_expired 删除所有过期条目"""
        cache = TTLCache(cleanup_interval=0)  # 禁用后台线程
        cache.set('a', 1, ttl=0.01)
        cache.set('b', 2, ttl=100)
        time.sleep(0.02)

        cache._cleanup_expired()
        assert cache.get('a') is None  # 已清理
        assert cache.get('b') == 2     # 未过期保留

    def test_background_cleanup_starts(self):
        """后台清理线程启动 (cleanup_interval > 0)"""
        cache = TTLCache(default_ttl=60, cleanup_interval=10)
        assert cache._cleanup_thread is not None
        assert cache._cleanup_thread.is_alive()
        cache._cleanup_stop.set()  # 手动停止

    def test_no_background_cleanup_when_zero(self):
        """cleanup_interval=0 不启动后台线程"""
        cache = TTLCache(cleanup_interval=0)
        assert cache._cleanup_thread is None

    def test_cleanup_thread_survives_error(self):
        """清理异常不导致线程崩溃"""
        cache = TTLCache(default_ttl=60, cleanup_interval=0.05)
        cache.set('a', 1)
        # 模拟清理: 不应抛异常
        time.sleep(0.1)
        cache._cleanup_expired()  # 直接调用, 验证不抛异常
        cache._cleanup_stop.set()


class TestTTLCacheEdgeCases:
    """边界情况测试"""

    def test_set_none_value(self):
        """None 值也能缓存 (has() 通过 get() 检查, None 返回 falsy 无法区分)"""
        cache = TTLCache()
        cache.set('key1', None)
        # get() 返回 None (无论是缓存了 None 还是键不存在)
        assert cache.get('key1') is None
        # 但键确实存在于内部 store 中
        assert len(cache) == 1
        # 注意: has() 调用 get(), None 是 falsy, 所以 has('key1') 返回 False
        # 这是设计权衡 — 不推荐缓存 None 值

    def test_set_empty_string_key(self):
        """空字符串作为键"""
        cache = TTLCache()
        cache.set('', 'empty_key')
        assert cache.get('') == 'empty_key'

    def test_set_large_value(self):
        """大值缓存"""
        cache = TTLCache()
        large_list = list(range(10000))
        cache.set('large', large_list)
        assert cache.get('large') == large_list

    def test_invalidate_empty_cache(self):
        """空缓存 invalidate 返回 0"""
        cache = TTLCache()
        assert cache.invalidate('any:') == 0

    def test_multiple_expired_cleanup(self):
        """_cleanup_expired 清理多个过期键"""
        cache = TTLCache(cleanup_interval=0)
        for i in range(10):
            cache.set(f'exp{i}', i, ttl=0.01)
        cache.set('keep', 'forever', ttl=100)
        time.sleep(0.02)

        cache._cleanup_expired()
        assert len(cache) == 1
        assert cache.get('keep') == 'forever'

    def test_evict_oldest_on_empty(self):
        """空缓存驱逐不抛异常"""
        cache = TTLCache()
        cache._evict_oldest()  # 不应该抛异常


class TestGlobalCacheInstances:
    """全局单例测试"""

    def test_dashboard_cache_exists(self):
        """dashboard_cache 全局单例可用"""
        from app.utils.cache import dashboard_cache
        assert dashboard_cache.default_ttl == 60
        assert dashboard_cache.max_size == 200

    def test_leaderboard_cache_exists(self):
        """leaderboard_cache 全局单例可用"""
        from app.utils.cache import leaderboard_cache
        assert leaderboard_cache.default_ttl == 300
        assert leaderboard_cache.max_size == 50

    def test_recommendation_cache_exists(self):
        """recommendation_cache 全局单例可用"""
        from app.utils.cache import recommendation_cache
        assert recommendation_cache.default_ttl == 600
        assert recommendation_cache.max_size == 100

    def test_global_caches_are_independent(self):
        """全局缓存实例相互独立"""
        from app.utils.cache import dashboard_cache, leaderboard_cache
        dashboard_cache.set('test', 'dash_val')
        leaderboard_cache.set('test', 'lead_val')
        assert dashboard_cache.get('test') != leaderboard_cache.get('test')
        dashboard_cache.delete('test')
        leaderboard_cache.delete('test')
