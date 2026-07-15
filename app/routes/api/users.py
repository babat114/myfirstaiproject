"""
============================================
用户管理 API
RESTful 用户管理接口 (仅管理员)
============================================
"""
from flask import Blueprint, jsonify, request

from app import db, logger
from app._timezone import localnow
from app.models.user import User
from app.services.auth_service import AuthService
from app.utils.auth_helpers import get_current_user
from app.utils.decorators import api_admin_required, api_login_required

users_api_bp = Blueprint('users_api', __name__)


@users_api_bp.route('', methods=['GET'])
@api_admin_required
def list_users():
    """获取用户列表 (管理员)
    ---
    tags:
      - Users
    summary: 获取用户列表
    description: 管理员获取所有用户列表，支持分页、角色筛选和搜索。
    parameters:
      - in: query
        name: page
        schema:
          type: integer
          default: 1
        description: 页码
      - in: query
        name: per_page
        schema:
          type: integer
          default: 20
        description: 每页数量
      - in: query
        name: role
        schema:
          type: string
          enum: [admin, researcher, viewer]
        description: 按角色筛选
      - in: query
        name: search
        schema:
          type: string
        description: 搜索用户名或邮箱
    responses:
      200:
        description: 用户列表
      401:
        description: 未认证
      403:
        description: 非管理员
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    """
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    role = request.args.get('role')
    search = request.args.get('search', '').strip() or None

    result = AuthService.list_users(
        page=page, per_page=per_page, role=role, search=search
    )

    return jsonify({
        'success': True,
        'data': result,
    })


@users_api_bp.route('/<int:user_id>', methods=['GET'])
@api_admin_required
def get_user(user_id):
    """获取单个用户详情 (管理员)
    ---
    tags:
      - Users
    summary: 获取用户详情
    parameters:
      - in: path
        name: user_id
        required: true
        schema:
          type: integer
        description: 用户ID
    responses:
      200:
        description: 用户详情
      404:
        description: 用户不存在
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    """
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'success': False, 'message': '用户不存在。'}), 404

    return jsonify({
        'success': True,
        'data': user.to_dict(include_private=True),
    })


@users_api_bp.route('/<int:user_id>', methods=['PUT'])
@api_admin_required
def update_user(user_id):
    """更新用户信息 (管理员)
    ---
    tags:
      - Users
    summary: 更新用户信息
    parameters:
      - in: path
        name: user_id
        required: true
        schema:
          type: integer
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              full_name:
                type: string
              bio:
                type: string
              organization:
                type: string
              role:
                type: string
                enum: [admin, researcher, viewer]
              is_active:
                type: boolean
              is_verified:
                type: boolean
    responses:
      200:
        description: 更新成功
      404:
        description: 用户不存在
      500:
        description: 更新失败
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    """
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'success': False, 'message': '用户不存在。'}), 404

    data = request.get_json(silent=True) or {}

    # 可更新的字段
    updatable = {'full_name', 'bio', 'organization', 'role', 'is_active', 'is_verified'}
    # 角色值白名单校验 (防止写入无效角色破坏权限检查)
    VALID_ROLES = {'admin', 'researcher', 'viewer'}
    try:
        for field, value in data.items():
            if field in updatable and hasattr(user, field):
                if field == 'role' and value not in VALID_ROLES:
                    return jsonify({
                        'success': False,
                        'message': f'无效的角色值: {value}。有效值: {", ".join(sorted(VALID_ROLES))}',
                    }), 400
                setattr(user, field, value)

        user.updated_at = localnow()
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '用户信息已更新。',
            'data': user.to_dict(),
        })
    except Exception as e:
        db.session.rollback()
        logger.error(f'更新用户 {user_id} 失败: {e}', exc_info=True)
        return jsonify({'success': False, 'message': '更新失败，请稍后重试。'}), 500


@users_api_bp.route('/<int:user_id>/role', methods=['PUT'])
@api_admin_required
def update_user_role(user_id):
    """更新用户角色 (管理员)
    ---
    tags:
      - Users
    summary: 更新用户角色
    parameters:
      - in: path
        name: user_id
        required: true
        schema:
          type: integer
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [role]
            properties:
              role:
                type: string
                enum: [admin, researcher, viewer]
    responses:
      200:
        description: 角色已更新
      400:
        description: 无效角色
      404:
        description: 用户不存在
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    """
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'success': False, 'message': '用户不存在。'}), 404

    new_role = request.get_json(silent=True).get('role')
    if new_role not in ('admin', 'researcher', 'viewer'):
        return jsonify({'success': False, 'message': '无效的角色。可选: admin, researcher, viewer'}), 400

    try:
        user.role = new_role
        user.updated_at = localnow()
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'用户 {user.username} 的角色已更新为 {new_role}。',
            'data': user.to_dict(),
        })
    except Exception as e:
        db.session.rollback()
        logger.error(f'更新用户 {user_id} 失败: {e}', exc_info=True)
        return jsonify({'success': False, 'message': '更新失败，请稍后重试。'}), 500


@users_api_bp.route('/<int:user_id>', methods=['DELETE'])
@api_admin_required
def delete_user(user_id):
    """删除用户 (管理员)
    ---
    tags:
      - Users
    summary: 删除用户
    parameters:
      - in: path
        name: user_id
        required: true
        schema:
          type: integer
        description: 要删除的用户ID
    responses:
      200:
        description: 删除成功
      400:
        description: 不能删除自己
      404:
        description: 用户不存在
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    """
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'success': False, 'message': '用户不存在。'}), 404

    current_user_obj = get_current_user()
    if current_user_obj and current_user_obj.id == user.id:
        return jsonify({'success': False, 'message': '不能删除自己的账户。'}), 400

    success = AuthService.delete_user(user)
    if success:
        return jsonify({'success': True, 'message': f'用户 {user.username} 已删除。'})
    else:
        return jsonify({'success': False, 'message': '删除失败，请稍后重试。'}), 500


@users_api_bp.route('/<int:user_id>/api-key', methods=['POST'])
@api_admin_required
def reset_user_api_key(user_id):
    """重置用户 API Key (管理员)
    ---
    tags:
      - Users
    summary: 重置用户 API Key
    parameters:
      - in: path
        name: user_id
        required: true
        schema:
          type: integer
    responses:
      200:
        description: API Key 已重置, 返回新 key
      404:
        description: 用户不存在
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    """
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'success': False, 'message': '用户不存在。'}), 404

    new_key = AuthService.regenerate_api_key(user)
    return jsonify({
        'success': True,
        'message': 'API Key 已重置。',
        'data': {'api_key': new_key},
    })


@users_api_bp.route('/me', methods=['GET'])
@api_login_required
def get_my_profile():
    """获取当前用户自己的资料
    ---
    tags:
      - Users
    summary: 获取当前用户资料
    responses:
      200:
        description: 当前用户完整信息
      401:
        description: 未认证
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    """
    user = get_current_user()
    if not user:
        return jsonify({'success': False, 'message': '未认证。'}), 401

    return jsonify({
        'success': True,
        'data': user.to_dict(include_private=True),
    })


@users_api_bp.route('/me', methods=['PUT'])
@api_login_required
def update_my_profile():
    """当前用户更新自己的资料
    ---
    tags:
      - Users
    summary: 更新当前用户资料
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              full_name:
                type: string
              email:
                type: string
              bio:
                type: string
              organization:
                type: string
    responses:
      200:
        description: 资料已更新
      400:
        description: 更新失败 (如邮箱格式错误)
      401:
        description: 未认证
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    """
    user = get_current_user()
    if not user:
        return jsonify({'success': False, 'message': '未认证。'}), 401

    data = request.get_json(silent=True) or {}
    success, error = AuthService.update_profile(user, data)

    if success:
        return jsonify({
            'success': True,
            'message': '资料已更新。',
            'data': user.to_dict(),
        })
    else:
        return jsonify({'success': False, 'message': error}), 400
