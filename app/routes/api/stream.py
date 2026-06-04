"""
训练 SSE 流端点
提供 Server-Sent Events 实时推送训练进度
"""
import json
import time

from flask import Blueprint, request, Response, jsonify
from flask_login import login_required, current_user
from app.services.training_service import TrainingService

stream_bp = Blueprint('stream', __name__)


@stream_bp.route('/training/<int:job_id>/stream')
@login_required
def training_stream(job_id):
    """
    SSE 端点 — 实时推送训练进度和日志

    前端使用:
        const source = new EventSource('/api/stream/training/123/stream');
        source.onmessage = (e) => updateUI(JSON.parse(e.data));
    """
    job = TrainingService.get_job_by_id(job_id)
    if not job:
        return Response('data: {"error": "任务不存在"}\n\n',
                        mimetype='text/event-stream')

    # 权限检查
    if job.owner_id != current_user.id and not current_user.is_admin:
        return Response('data: {"error": "权限不足"}\n\n',
                        mimetype='text/event-stream')

    def generate():
        last_log_len = 0
        last_keyframe_count = 0
        while True:
            status = TrainingService.get_job_status(job_id)
            if not status:
                status = job.to_dict()
                status['is_finished'] = True

            # 只推送新增日志行
            log_text = status.get('log_tail', '')
            if log_text and len(log_text) > last_log_len:
                status['new_logs'] = log_text[last_log_len:]
                last_log_len = len(log_text)
            else:
                status['new_logs'] = ''

            # 推送进度关键帧 (新增 epoch → 发送关键帧增量)
            metrics_history = status.get('metrics_history', [])
            current_keyframe_count = len(metrics_history)
            if current_keyframe_count > last_keyframe_count:
                status['new_keyframes'] = metrics_history[last_keyframe_count:]
                last_keyframe_count = current_keyframe_count
            else:
                status['new_keyframes'] = []

            # 进度条数据 (确保前端有这些字段)
            status['progress_percent'] = status.get('progress_percent', 0)
            status['current_epoch'] = status.get('current_epoch', 0)
            status['total_epochs'] = status.get('total_epochs', 0)
            status['current_step'] = status.get('current_step', 0)
            status['total_steps'] = status.get('total_steps', 0)

            yield f"data: {json.dumps(status, ensure_ascii=False)}\n\n"

            if status.get('is_finished'):
                break

            time.sleep(1.5)

    return Response(generate(), mimetype='text/event-stream')


@stream_bp.route('/training/<int:job_id>/status')
@login_required
def training_status(job_id):
    """AJAX 轮询备选 — 返回 JSON 状态"""
    job = TrainingService.get_job_by_id(job_id)
    if not job:
        return jsonify({'success': False, 'message': '任务不存在'}), 404

    status = TrainingService.get_job_status(job_id)
    return jsonify({'success': True, 'data': status or job.to_dict()})
