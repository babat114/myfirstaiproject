"""
============================================
异步导出任务追踪器
支持 ONNX 导出和 Docker 部署包生成的后台进度跟踪
============================================
"""
import uuid
import threading
import time
from typing import Optional, Callable

from app._timezone import localnow


class ExportTask:
    """单个导出任务"""

    def __init__(self, task_id: str, model_uuid: str, task_type: str):
        self.task_id = task_id
        self.model_uuid = model_uuid
        self.task_type = task_type  # 'onnx' | 'deploy'
        self.status = 'pending'     # pending | running | completed | failed
        self.progress = 0           # 0-100
        self.message = ''
        self.result = None          # 成功时的结果数据
        self.error = None
        self.created_at = localnow()
        self.completed_at = None

    def to_dict(self) -> dict:
        return {
            'task_id': self.task_id,
            'model_uuid': self.model_uuid,
            'task_type': self.task_type,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }


class ExportTaskTracker:
    """导出任务追踪器 (单例, 内存存储)

    使用:
        tracker = ExportTaskTracker()
        task_id = tracker.create_task(model_uuid, 'onnx')
        tracker.run_async(task_id, export_fn)
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._tasks: dict[str, ExportTask] = {}
        self._lock = threading.Lock()
        self._initialized = True

    def create_task(self, model_uuid: str, task_type: str) -> str:
        """创建新任务，返回 task_id"""
        task_id = str(uuid.uuid4())[:12]
        task = ExportTask(task_id, model_uuid, task_type)
        with self._lock:
            self._tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> Optional[dict]:
        """获取任务状态"""
        with self._lock:
            task = self._tasks.get(task_id)
        return task.to_dict() if task else None

    def update_task(self, task_id: str, **kwargs):
        """更新任务状态"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                for k, v in kwargs.items():
                    if hasattr(task, k):
                        setattr(task, k, v)

    def run_async(self, task_id: str, fn: Callable, *args, **kwargs):
        """在后台线程中执行导出任务

        Args:
            task_id: 任务 ID
            fn: 导出函数 (应返回 (success: bool, message: str, result: any))
        """
        def _run():
            self.update_task(task_id, status='running', progress=10,
                           message='任务已启动...')
            try:
                # 模拟进度 (实际导出通常很快, 但大模型可能需要时间)
                self._simulate_progress(task_id, 10, 50)

                result_tuple = fn(*args, **kwargs)
                success, message = result_tuple[0], result_tuple[1]
                result_data = None
                if len(result_tuple) > 2:
                    result_data = {'path': result_tuple[2]}
                if len(result_tuple) > 3:
                    result_data['zip_file'] = result_tuple[3]

                if success:
                    self.update_task(
                        task_id, status='completed', progress=100,
                        message=message, result=result_data,
                        completed_at=localnow()
                    )
                else:
                    self.update_task(
                        task_id, status='failed', progress=0,
                        error=message or '导出失败',
                        completed_at=localnow()
                    )
            except Exception as e:
                self.update_task(
                    task_id, status='failed', progress=0,
                    error=str(e),
                    completed_at=localnow()
                )
            finally:
                from app import db
                db.session.remove()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _simulate_progress(self, task_id: str, start: int, end: int):
        """渐进式更新进度 (非精确, 仅用于 UX)"""
        steps = 5
        step_size = (end - start) // steps
        for i in range(1, steps + 1):
            time.sleep(0.3)
            pct = start + step_size * i
            self.update_task(task_id, progress=min(pct, end),
                           message=f'处理中 ({min(pct, end)}%)...')

    def cleanup_old_tasks(self, max_age_seconds: int = 3600):
        """清理超过 max_age_seconds 的已完成/失败任务"""
        now = localnow()
        with self._lock:
            stale = []
            for tid, task in self._tasks.items():
                if task.status in ('completed', 'failed') and task.completed_at:
                    if (now - task.completed_at).total_seconds() > max_age_seconds:
                        stale.append(tid)
            for tid in stale:
                del self._tasks[tid]
