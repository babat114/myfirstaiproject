"""
训练 SSE 流端点 — 事件驱动架构
TrainingCallback → EventBus → SSE 客户端
零数据库轮询，仅在训练事件发生时推送数据
"""
import json
import queue
import threading

from flask import Blueprint, request, Response, jsonify, current_app
from flask_login import login_required, current_user
from app import db, logger
from app.models.training_job import TrainingJob
from app.services.training_service import TrainingService

stream_bp = Blueprint('stream', __name__)

# SSE 连接限制 — 防止连接耗尽
_MAX_SSE_CONNECTIONS = 50
_sse_connections = 0
_sse_lock = threading.Lock()


def _get_max_sse_connections() -> int:
    """从配置读取 SSE 最大连接数, 默认 50"""
    try:
        return int(current_app.config.get('SSE_MAX_CONNECTIONS', _MAX_SSE_CONNECTIONS))
    except RuntimeError:
        return _MAX_SSE_CONNECTIONS


def _acquire_sse_slot() -> bool:
    """尝试获取 SSE 连接槽位, 成功返回 True"""
    global _sse_connections
    with _sse_lock:
        if _sse_connections >= _get_max_sse_connections():
            return False
        _sse_connections += 1
        return True


def _release_sse_slot():
    """释放 SSE 连接槽位"""
    global _sse_connections
    with _sse_lock:
        _sse_connections = max(0, _sse_connections - 1)


@stream_bp.route('/tuning/<tuning_id>/stream')
@login_required
def tuning_stream(tuning_id):
    """SSE — 超参数调优实时进度
    ---
    tags:
      - Stream
    summary: 超参数调优 SSE 流
    description: Server-Sent Events — GridSearchCV/RandomSearch/AutoML 实时进度推送 (500ms 间隔)。仅支持 Session Cookie 认证。
    parameters:
      - in: path
        name: tuning_id
        required: true
        schema:
          type: string
        description: 调优会话ID
    responses:
      200:
        description: text/event-stream SSE 流
      503:
        description: SSE 连接数已达上限
    """
    # 先获取 tracker (可能抛异常), 成功后再占用 slot (避免 slot 泄漏)
    from app.services.hyperparameter_tuning import get_tuning_tracker
    tracker = get_tuning_tracker()

    if not _acquire_sse_slot():
        return Response(
            f"data: {json.dumps({'error': 'SSE 连接数已达上限，请稍后重试'}, ensure_ascii=False)}\n\n",
            mimetype='text/event-stream',
            status=503,
        )

    def generate():
        last_step = -1
        init_retries = 0
        try:
            while True:
                session = tracker.get(tuning_id)
                if session is None:
                    # 后台线程可能尚未初始化 tracker — 等待最多 5s
                    if init_retries < 10:
                        init_retries += 1
                        import time as _time
                        _time.sleep(0.5)
                        continue
                    yield f"data: {json.dumps({'error': '会话不存在或已过期'}, ensure_ascii=False)}\n\n"
                    return

                # 只在进度变化时推送 (减少网络传输)
                current_step = session.get('current_step', 0)
                if current_step != last_step or session['status'] != 'running':
                    last_step = current_step
                    yield f"data: {json.dumps(session, ensure_ascii=False)}\n\n"

                if session['status'] in ('completed', 'failed'):
                    return

                import time as _time
                _time.sleep(0.5)  # 500ms 间隔
        except GeneratorExit:
            pass
        finally:
            _release_sse_slot()

    return Response(generate(), mimetype='text/event-stream')


@stream_bp.route('/training/<int:job_id>/stream')
@login_required
def training_stream(job_id):
    """SSE — 训练进度实时推送
    ---
    tags:
      - Stream
    summary: 训练任务 SSE 流
    description: 事件驱动 SSE — TrainingCallback 推送训练进度/日志/指标变化，无数据库轮询。首次连接发送完整状态快照。15s 心跳保活。
    parameters:
      - in: path
        name: job_id
        required: true
        schema:
          type: integer
        description: 训练任务ID
    responses:
      200:
        description: text/event-stream SSE 流
      404:
        description: 任务不存在
      503:
        description: SSE 连接数已达上限
    """
    if not _acquire_sse_slot():
        return Response(
            f"data: {json.dumps({'error': 'SSE 连接数已达上限，请稍后重试'}, ensure_ascii=False)}\n\n",
            mimetype='text/event-stream',
            status=503,
        )

    job = TrainingService.get_job_by_id(job_id)
    if not job:
        _release_sse_slot()
        return Response(
            f"data: {json.dumps({'error': '任务不存在'}, ensure_ascii=False)}\n\n",
            mimetype='text/event-stream')

    if not job.is_viewable_by(current_user):
        _release_sse_slot()
        return Response(
            f"data: {json.dumps({'error': '权限不足'}, ensure_ascii=False)}\n\n",
            mimetype='text/event-stream')

    from app.utils.event_bus import get_event_bus
    event_bus = get_event_bus()
    event_queue = event_bus.subscribe(job_id)

    def generate():
        try:
            # 首次连接 — 发送完整状态快照
            db.session.commit()
            db.session.expire_all()
            status = TrainingService.get_job_status(job_id)
            if status:
                status['_init'] = True
                yield f"data: {json.dumps(status, ensure_ascii=False)}\n\n"

            # 如果任务已完成，发送快照后立即结束
            if job.is_finished:
                yield f"data: {json.dumps({'is_finished': True, 'message': '任务已结束'}, ensure_ascii=False)}\n\n"
                return

            # 事件循环 — 阻塞等待训练事件
            while True:
                try:
                    msg = event_queue.get(timeout=15)  # 15s 心跳超时
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    # 心跳 — 保持连接，检测断线
                    yield f": heartbeat\n\n"

                    # 心跳时检查任务是否已结束
                    db.session.commit()
                    full_job = db.session.get(TrainingJob, job_id)
                    if full_job and full_job.is_finished:
                        yield f"data: {json.dumps({'is_finished': True, 'message': '训练已完成'}, ensure_ascii=False)}\n\n"
                        return
        except GeneratorExit:
            pass  # 客户端断开连接
        finally:
            event_bus.unsubscribe(job_id, event_queue)
            _release_sse_slot()

    return Response(generate(), mimetype='text/event-stream')


@stream_bp.route('/training/<int:job_id>/status')
@login_required
def training_status(job_id):
    """AJAX 轮询训练状态
    ---
    tags:
      - Stream
    summary: 训练状态 JSON 查询
    description: SSE 的 AJAX 备选方案 — 用于页面初始加载和训练结束后查看最终状态。
    parameters:
      - in: path
        name: job_id
        required: true
        schema:
          type: integer
    responses:
      200:
        description: 训练任务完整状态JSON
      404:
        description: 任务不存在
    """
    job = TrainingService.get_job_by_id(job_id)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在'}), 404

    db.session.commit()
    db.session.expire_all()
    status = TrainingService.get_job_status(job_id)
    return jsonify({'success': True, 'data': status or job.to_dict()})
