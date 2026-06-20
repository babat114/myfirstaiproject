"""
============================================
评论 RESTful API
JSON 接口 — 支持 JWT/Session/API Key 认证
============================================
"""
from flask import Blueprint, request, jsonify
from app.services.comment_service import CommentService
from app.utils.decorators import api_login_required
from app.utils.auth_helpers import get_current_user

comments_api_bp = Blueprint('comments_api', __name__)


@comments_api_bp.route('/models/<int:model_id>/comments', methods=['GET'])
@api_login_required
def list_comments(model_id):
    """GET /api/models/<model_id>/comments — 获取模型的评论列表"""
    user = get_current_user()
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    include_hidden = request.args.get('include_hidden', '').lower() == 'true'

    result = CommentService.get_comments_for_model(
        model_id=model_id,
        user=user,
        page=page,
        per_page=per_page,
        include_hidden=include_hidden and user.is_admin if user else False,
    )

    return jsonify({'success': True, 'data': result})


@comments_api_bp.route('/models/<int:model_id>/comments', methods=['POST'])
@api_login_required
def add_comment(model_id):
    """POST /api/models/<model_id>/comments — 发表评论"""
    user = get_current_user()
    data = request.get_json(silent=True) or {}

    content = data.get('content', '').strip()
    parent_id = data.get('parent_id')

    comment, error = CommentService.add_comment(
        user=user,
        model_id=model_id,
        content=content,
        parent_id=parent_id,
    )

    if error and not comment:
        return jsonify({'success': False, 'message': error}), 400

    if error and comment:
        # 评论被自动屏蔽
        return jsonify({
            'success': True,
            'message': error,
            'data': comment.to_dict(),
            'flagged': True,
        }), 201

    return jsonify({
        'success': True,
        'message': '评论发表成功。',
        'data': comment.to_dict(),
    }), 201


@comments_api_bp.route('/comments/<int:comment_id>', methods=['DELETE'])
@api_login_required
def delete_comment(comment_id):
    """DELETE /api/comments/<comment_id> — 删除评论"""
    user = get_current_user()
    permanent = request.args.get('permanent', '').lower() == 'true'

    success, error = CommentService.delete_comment(
        comment_id=comment_id,
        user=user,
        permanent=permanent and user.is_admin if user else False,
    )

    if not success:
        return jsonify({'success': False, 'message': error}), 403

    return jsonify({'success': True, 'message': '评论已删除。'})


@comments_api_bp.route('/comments/<int:comment_id>/restore', methods=['POST'])
@api_login_required
def restore_comment(comment_id):
    """POST /api/comments/<comment_id>/restore — 恢复评论 (管理员)"""
    user = get_current_user()

    success, error = CommentService.restore_comment(
        comment_id=comment_id,
        user=user,
    )

    if not success:
        return jsonify({'success': False, 'message': error}), 403

    return jsonify({'success': True, 'message': '评论已恢复。'})


@comments_api_bp.route('/comments/<int:comment_id>/replies', methods=['GET'])
@api_login_required
def get_replies(comment_id):
    """GET /api/comments/<comment_id>/replies — 获取评论的回复"""
    user = get_current_user()

    replies = CommentService.get_replies_for_comment(
        parent_id=comment_id,
        user=user,
    )

    return jsonify({'success': True, 'data': replies})
