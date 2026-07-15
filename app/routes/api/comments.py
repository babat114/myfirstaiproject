"""
============================================
评论 RESTful API
JSON 接口 — 支持 JWT/Session/API Key 认证
============================================
"""
from flask import Blueprint, jsonify, request

from app.services.comment_service import CommentService
from app.utils.auth_helpers import get_current_user
from app.utils.decorators import api_login_required

comments_api_bp = Blueprint('comments_api', __name__)


@comments_api_bp.route('/models/<int:model_id>/comments', methods=['GET'])
@api_login_required
def list_comments(model_id):
    """获取模型评论列表
    ---
    tags:
      - Comments
    summary: 获取评论
    parameters:
      - in: path
        name: model_id
        required: true
        schema:
          type: integer
      - in: query
        name: page
        schema:
          type: integer
          default: 1
      - in: query
        name: per_page
        schema:
          type: integer
          default: 20
      - in: query
        name: include_hidden
        schema:
          type: string
        description: 管理员可查看被屏蔽评论
    responses:
      200:
        description: 评论列表 (按最新优先)
    """
    user = get_current_user()
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
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
    """发表评论
    ---
    tags:
      - Comments
    summary: 发表评论
    parameters:
      - in: path
        name: model_id
        required: true
        schema:
          type: integer
    requestBody:
      content:
        application/json:
          schema:
            type: object
            required: [content]
            properties:
              content: {type: string, description: 评论内容}
              parent_id:
                type: integer
                description: 父评论ID (回复时使用)
    responses:
      201:
        description: 发表成功 (可能被自动屏蔽返回 flagged=true)
      400:
        description: 内容为空 / 模型不存在
    """
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
    """删除评论
    ---
    tags:
      - Comments
    summary: 删除评论
    parameters:
      - in: path
        name: comment_id
        required: true
        schema:
          type: integer
      - in: query
        name: permanent
        schema:
          type: string
        description: 管理员传 true 可物理删除
    responses:
      200:
        description: 软删除成功
      403:
        description: 权限不足 (非作者或非管理员)
    """
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
    """恢复评论 (管理员)
    ---
    tags:
      - Comments
    summary: 恢复评论
    parameters:
      - in: path
        name: comment_id
        required: true
        schema:
          type: integer
    responses:
      200:
        description: 恢复成功
      403:
        description: 权限不足
    """
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
    """获取评论回复
    ---
    tags:
      - Comments
    summary: 获取回复列表
    parameters:
      - in: path
        name: comment_id
        required: true
        schema:
          type: integer
    responses:
      200:
        description: 回复列表
    """
    user = get_current_user()

    replies = CommentService.get_replies_for_comment(
        parent_id=comment_id,
        user=user,
    )

    return jsonify({'success': True, 'data': replies})
