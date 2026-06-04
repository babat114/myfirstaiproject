"""
============================================
JWT 认证 API
RESTful 认证接口 — 登录获取 Token、刷新 Token
============================================
"""
from flask import Blueprint, request, jsonify
from app.services.auth_service import AuthService
from app.utils.decorators import api_login_required
from app.utils.auth_helpers import get_current_user

auth_api_bp = Blueprint('auth_api', __name__)


@auth_api_bp.route('/login', methods=['POST'])
def login():
    """
    用户登录 — 返回 JWT Token 对

    Request JSON:
        {
            "login_id": "用户名或邮箱",
            "password": "密码"
        }

    Response 200:
        {
            "success": true,
            "data": {
                "access_token": "eyJ...",
                "refresh_token": "eyJ...",
                "token_type": "Bearer",
                "expires_in": 7200
            }
        }
    """
    data = request.get_json(silent=True) or {}
    login_id = data.get('login_id', '').strip()
    password = data.get('password', '')

    if not login_id or not password:
        return jsonify({
            'success': False,
            'message': '请提供 login_id 和 password。',
        }), 400

    tokens, error, status = AuthService.login_jwt(login_id, password)
    if error:
        return jsonify({'success': False, 'message': error}), status

    return jsonify({
        'success': True,
        'message': '登录成功。',
        'data': tokens,
    })


@auth_api_bp.route('/refresh', methods=['POST'])
def refresh():
    """
    刷新 Access Token

    Request JSON:
        {
            "refresh_token": "eyJ..."
        }

    Response 200:
        {
            "success": true,
            "data": {
                "access_token": "eyJ...",
                "refresh_token": "eyJ...",
                "token_type": "Bearer",
                "expires_in": 7200
            }
        }
    """
    data = request.get_json(silent=True) or {}
    refresh_token = data.get('refresh_token', '').strip()

    if not refresh_token:
        return jsonify({
            'success': False,
            'message': '请提供 refresh_token。',
        }), 400

    tokens, error, status = AuthService.refresh_jwt(refresh_token)
    if error:
        return jsonify({'success': False, 'message': error}), status

    return jsonify({
        'success': True,
        'message': 'Token 已刷新。',
        'data': tokens,
    })


@auth_api_bp.route('/me', methods=['GET'])
@api_login_required
def me():
    """
    获取当前认证用户信息

    Headers:
        Authorization: Bearer <access_token>
        或
        X-API-Key: <api_key>

    Response 200:
        {
            "success": true,
            "data": {
                "id": 1,
                "username": "admin",
                "email": "...",
                "role": "admin",
                ...
            }
        }
    """
    user = get_current_user()
    if not user:
        return jsonify({'success': False, 'message': '未认证。'}), 401

    return jsonify({
        'success': True,
        'data': user.to_dict(include_private=True),
    })
