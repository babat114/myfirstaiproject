"""
训练事件总线 — 线程安全的事件发布/订阅
训练线程发布事件 → SSE 客户端订阅消费
替代旧的 50 FPS DB轮询，实现真正的事件驱动推送
"""
import queue
import threading
import json
from typing import Optional
from app import logger


class TrainingEventBus:
    """训练事件总线 (单例)

    使用方式:
        bus = get_event_bus()
        bus.subscribe(job_id)         # SSE 客户端订阅
        bus.publish(job_id, {...})    # 训练线程发布
        bus.unsubscribe(job_id, q)    # SSE 断开时清理
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        # {job_id: set[queue.Queue]} — 一个job可能有多个SSE客户端
        self._subscribers: dict[int, set[queue.Queue]] = {}
        self._sub_lock = threading.Lock()
        logger.info('TrainingEventBus 初始化完成')

    def publish(self, job_id: int, event_type: str, data: dict):
        """发布事件到所有订阅者
        Args:
            job_id: 训练任务ID
            event_type: 'progress' | 'log' | 'metrics' | 'status_change' | 'complete' | 'error'
            data: 事件负载
        """
        message = json.dumps({
            'type': event_type,
            'job_id': job_id,
            'data': data,
        }, ensure_ascii=False)

        with self._sub_lock:
            subscribers = self._subscribers.get(job_id, set())
            dead = set()
            for q in subscribers:
                try:
                    q.put_nowait(message)
                except queue.Full:
                    dead.add(q)

            # 清理已满的队列 (客户端断开)
            if dead:
                subscribers -= dead

    def subscribe(self, job_id: int) -> queue.Queue:
        """订阅训练事件 — 返回一个带缓冲的消息队列"""
        q = queue.Queue(maxsize=500)  # 最多缓冲 500 条消息
        with self._sub_lock:
            if job_id not in self._subscribers:
                self._subscribers[job_id] = set()
            self._subscribers[job_id].add(q)
        logger.debug(f'SSE 客户端订阅 job {job_id}, 当前订阅者: {len(self._subscribers[job_id])}')
        return q

    def unsubscribe(self, job_id: int, q: queue.Queue):
        """取消订阅"""
        with self._sub_lock:
            subs = self._subscribers.get(job_id, set())
            subs.discard(q)
            if not subs:
                del self._subscribers[job_id]
        logger.debug(f'SSE 客户端取消订阅 job {job_id}')


# 全局单例
_event_bus: Optional[TrainingEventBus] = None


def get_event_bus() -> TrainingEventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = TrainingEventBus()
    return _event_bus
