"""
============================================
用户管理 API
RESTful 用户管理接口 (仅管理员)
============================================
"""
from flask import Blueprint, request, jsonify
from app import db
from app.services.auth_service import AuthService
from app.utils.decorators import api_login_required, api_admin_required
from app.utils.auth_helpers import get_current_user
from app.models.user import User

users_api_bp = Blueprint('users_api', __name__)


@users_api_bp.route('', methods=['GET'])
@api_admin_required
def list_users():
    """获取用户列表 (管理员)"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    role = request.args.get('role')
    search = request.args.get('search', '').strip() or None

    result = AuthService.list_users(page=page, per_page=per_page)

    # 额外筛选
    users = result['users']
    if role:
        users = [u for u in users if u.get('role') == role]
    if search:
        term = search.lower()
        users = [u for u in users if
                 term in (u.get('username') or '').lower() or
                 term in (u.get('email') or '').lower() or
                 term in (u.get('full_name') or '').lower() or
                 term in (u.get('organization') or '').lower()]

    return jsonify({
        'success': True,
        'data': {
            'users': users,
            'total': len(users),
            'page': page,
            'per_page': per_page,
        }
    })


@users_api_bp.route('/<int:user_id>', methods=['GET'])
@api_admin_required
def get_user(user_id):
    """获取单个用户详情 (管理员)"""
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
    """更新用户信息 (管理员)"""
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'success': False, 'message': '用户不存在。'}), 404

    data = request.get_json(silent=True) or {}

    # 可更新的字段
    updatable = {'full_name', 'bio', 'organization', 'role', 'is_active', 'is_verified'}
    try:
        from datetime import datetime, timezone

        for field, value in data.items():
            if field in updatable and hasattr(user, field):
                setattr(user, field, value)

        user.updated_at = datetime.now(timezone.utc)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '用户信息已更新。',
            'data': user.to_dict(),
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'更新失败: {str(e)}'}), 500


@users_api_bp.route('/<int:user_id>/role', methods=['PUT'])
@api_admin_required
def update_user_role(user_id):
    """更新用户角色"""
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'success': False, 'message': '用户不存在。'}), 404

    new_role = request.get_json(silent=True).get('role')
    if new_role not in ('admin', 'researcher', 'viewer'):
        return jsonify({'success': False, 'message': '无效的角色。可选: admin, researcher, viewer'}), 400

    try:
        from datetime import datetime, timezone

        user.role = new_role
        user.updated_at = datetime.now(timezone.utc)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'用户 {user.username} 的角色已更新为 {new_role}。',
            'data': user.to_dict(),
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'更新失败: {str(e)}'}), 500


@users_api_bp.route('/<int:user_id>', methods=['DELETE'])
@api_admin_required
def delete_user(user_id):
    """删除用户 (管理员)"""
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
        return jsonify({'success': False, 'message': '删除失败。'}), 500


@users_api_bp.route('/<int:user_id>/api-key', methods=['POST'])
@api_admin_required
def reset_user_api_key(user_id):
    """重置用户 API Key (管理员)"""
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
    """获取当前用户自己的资料"""
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
    """当前用户更新自己的资料"""
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
