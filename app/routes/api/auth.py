"""
============================================
JWT 认证 API
RESTful 认证接口 — 登录获取 Token、刷新 Token
============================================
"""

from flask import Blueprint, jsonify, request

from app.services.auth_service import AuthService
from app.utils.auth_helpers import get_current_user
from app.utils.decorators import api_login_required, rate_limit

auth_api_bp = Blueprint('auth_api', __name__)


@auth_api_bp.route('/login', methods=['POST'])
@rate_limit(max_calls=10, period=60)  # 每IP每分钟最多10次登录尝试
def login():
    """用户登录 — 返回 JWT Token 对
    ---
    tags:
      - Auth
    summary: 用户登录
    description: 使用用户名/邮箱+密码登录, 返回 JWT access token 和 refresh token。
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [login_id, password]
            properties:
              login_id:
                type: string
                description: 用户名或邮箱
                example: admin
              password:
                type: string
                format: password
                description: 密码
                example: Admin123456
    responses:
      200:
        description: 登录成功
        content:
          application/json:
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  example: true
                data:
                  type: object
                  properties:
                    access_token:
                      type: string
                    refresh_token:
                      type: string
                    token_type:
                      type: string
                      example: Bearer
                    expires_in:
                      type: integer
                      example: 7200
      400:
        description: 缺少 login_id 或 password
      401:
        description: 登录失败 (用户名/密码错误或账户未激活)
      429:
        description: 登录频率超限 (每IP 10次/分钟)
    security: []
    """
    data = request.get_json(silent=True) or {}
    login_id = data.get('login_id', '').strip()
    password = data.get('password', '')

    if not login_id or not password:
        return jsonify(
            {
                'success': False,
                'message': '请提供 login_id 和 password。',
            }
        ), 400

    tokens, error, status = AuthService.login_jwt(login_id, password)
    if error:
        return jsonify({'success': False, 'message': error}), status

    return jsonify(
        {
            'success': True,
            'message': '登录成功。',
            'data': tokens,
        }
    )


@auth_api_bp.route('/refresh', methods=['POST'])
@rate_limit(max_calls=30, period=60)  # 每IP每分钟最多30次刷新
def refresh():
    """刷新 Access Token
    ---
    tags:
      - Auth
    summary: 刷新 JWT Token
    description: 使用 refresh token 获取新的 access token 对。旧 refresh token 同时轮换。
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [refresh_token]
            properties:
              refresh_token:
                type: string
                description: 之前登录获取的 refresh_token
    responses:
      200:
        description: Token 刷新成功
      400:
        description: 缺少 refresh_token
      401:
        description: refresh_token 无效或已过期
      429:
        description: 刷新频率超限 (每IP 30次/分钟)
    security: []
    """
    data = request.get_json(silent=True) or {}
    refresh_token = data.get('refresh_token', '').strip()

    if not refresh_token:
        return jsonify(
            {
                'success': False,
                'message': '请提供 refresh_token。',
            }
        ), 400

    tokens, error, status = AuthService.refresh_jwt(refresh_token)
    if error:
        return jsonify({'success': False, 'message': error}), status

    return jsonify(
        {
            'success': True,
            'message': 'Token 已刷新。',
            'data': tokens,
        }
    )


@auth_api_bp.route('/me', methods=['GET'])
@api_login_required
def me():
    """获取当前认证用户信息
    ---
    tags:
      - Auth
    summary: 获取当前用户信息
    description: 返回当前认证用户的完整档案 (需 JWT Bearer Token 或 API Key)。
    responses:
      200:
        description: 用户信息
      401:
        description: 未认证或 Token 无效
    security:
      - BearerAuth: []
      - ApiKeyAuth: []
    """
    user = get_current_user()
    if not user:
        return jsonify({'success': False, 'message': '未认证。'}), 401

    return jsonify(
        {
            'success': True,
            'data': user.to_dict(include_private=True),
        }
    )
