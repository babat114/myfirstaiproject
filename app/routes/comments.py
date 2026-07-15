"""
============================================
评论 Web 路由
用户发表评论、删除评论 (需登录)
============================================
"""

from flask import Blueprint, flash, jsonify, redirect, request, url_for
from flask_login import current_user, login_required

from app.services.comment_service import CommentService

comments_bp = Blueprint('comments', __name__)


@comments_bp.route('/add', methods=['POST'])
@login_required
def add_comment():
    """添加评论 — POST /comments/add"""
    model_id = request.form.get('model_id', type=int)
    content = request.form.get('content', '').strip()
    parent_id = request.form.get('parent_id', type=int)

    if not model_id:
        flash('缺少模型ID。', 'danger')
        return redirect(request.referrer or url_for('models.public_models'))

    comment, error = CommentService.add_comment(
        user=current_user,
        model_id=model_id,
        content=content,
        parent_id=parent_id,
    )

    if error and not comment:
        flash(error, 'danger')
    elif error and comment:
        # 评论被自动屏蔽
        flash(error, 'warning')
    else:
        flash('评论发表成功！', 'success')

    return redirect(request.referrer or url_for('models.model_detail', model_id=model_id))


@comments_bp.route('/<int:comment_id>/delete', methods=['POST'])
@login_required
def delete_comment(comment_id):
    """删除评论 — POST /comments/<id>/delete"""
    # 检查是否是 AJAX 请求
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    success, error = CommentService.delete_comment(
        comment_id=comment_id,
        user=current_user,
        permanent=current_user.is_admin,
    )

    if is_ajax:
        if success:
            return jsonify({'success': True, 'message': '评论已删除。'})
        return jsonify({'success': False, 'message': error}), 403

    if success:
        flash('评论已删除。', 'success')
    else:
        flash(error, 'danger')

    return redirect(request.referrer or url_for('models.public_models'))


@comments_bp.route('/<int:comment_id>/restore', methods=['POST'])
@login_required
def restore_comment(comment_id):
    """恢复评论 — POST /comments/<id>/restore (仅管理员)"""
    success, error = CommentService.restore_comment(
        comment_id=comment_id,
        user=current_user,
    )

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    if is_ajax:
        if success:
            return jsonify({'success': True, 'message': '评论已恢复。'})
        return jsonify({'success': False, 'message': error}), 403

    if success:
        flash('评论已恢复。', 'success')
    else:
        flash(error, 'danger')

    return redirect(request.referrer or url_for('models.public_models'))
