"""
============================================
JWT 认证测试
测试 Token 生成、API 登录、刷新、认证装饰器
============================================
"""
import json
import pytest
from app import db
from app.models.user import User
from app.utils.jwt_helpers import (
    generate_access_token,
    generate_refresh_token,
    generate_token_pair,
    decode_access_token,
    decode_refresh_token,
    extract_token_from_header,
)


class TestJWTHelpers:
    """JWT 工具函数单元测试"""

    def test_generate_access_token(self, app):
        """生成 Access Token"""
        with app.app_context():
            token = generate_access_token(1, 'testuser', 'researcher')
            assert isinstance(token, str)
            assert token.count('.') == 2  # JWT 格式: header.payload.signature

    def test_generate_refresh_token(self, app):
        """生成 Refresh Token"""
        with app.app_context():
            token = generate_refresh_token(1, 'testuser')
            assert isinstance(token, str)
            assert token.count('.') == 2

    def test_generate_token_pair(self, app):
        """生成 Token 对"""
        with app.app_context():
            pair = generate_token_pair(1, 'testuser', 'researcher')
            assert 'access_token' in pair
            assert 'refresh_token' in pair
            assert pair['token_type'] == 'Bearer'
            assert pair['expires_in'] > 0

    def test_decode_access_token(self, app):
        """解码 Access Token"""
        with app.app_context():
            token = generate_access_token(1, 'testuser', 'researcher')
            payload, error = decode_access_token(token)
            assert error is None
            assert payload['sub'] == '1'       # PyJWT 2.10+ sub 为字符串
            assert payload['username'] == 'testuser'
            assert payload['role'] == 'researcher'
            assert payload['type'] == 'access'

    def test_decode_refresh_token(self, app):
        """解码 Refresh Token"""
        with app.app_context():
            token = generate_refresh_token(1, 'testuser')
            payload, error = decode_refresh_token(token)
            assert error is None
            assert payload['sub'] == '1'
            assert payload['type'] == 'refresh'

    def test_wrong_token_type(self, app):
        """用 decode_access_token 解码 refresh token 应失败"""
        with app.app_context():
            token = generate_refresh_token(1, 'testuser')
            payload, error = decode_access_token(token)
            assert error is not None
            assert '类型不匹配' in error

    def test_invalid_token(self, app):
        """解码无效 Token"""
        with app.app_context():
            payload, error = decode_access_token('not.a.real.token')
            assert error is not None

    def test_extract_token_from_header(self, app):
        """从请求头提取 Bearer token"""
        with app.test_request_context(headers={
            'Authorization': 'Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOjF9.xxx'
        }):
            token = extract_token_from_header()
            assert token == 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOjF9.xxx'

    def test_extract_token_no_header(self, app):
        """无 Authorization header 时返回 None"""
        with app.test_request_context():
            token = extract_token_from_header()
            assert token is None


class TestJWTAuthAPI:
    """JWT API 认证接口测试"""

    def test_login_success(self, client, test_user):
        """JWT 登录成功"""
        response = client.post('/api/auth/login', json={
            'login_id': 'testuser',
            'password': 'Test123456',
        })
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'access_token' in data['data']
        assert 'refresh_token' in data['data']

    def test_login_wrong_password(self, client, test_user):
        """JWT 登录 — 密码错误"""
        response = client.post('/api/auth/login', json={
            'login_id': 'testuser',
            'password': 'WrongPassword123',
        })
        assert response.status_code == 401
        data = json.loads(response.data)
        assert data['success'] is False

    def test_login_missing_fields(self, client):
        """JWT 登录 — 缺少字段"""
        response = client.post('/api/auth/login', json={})
        assert response.status_code == 400

    def test_refresh_token_success(self, client, test_user, app):
        """刷新 Access Token"""
        with app.app_context():
            refresh_token = generate_refresh_token(test_user.id, test_user.username)

        response = client.post('/api/auth/refresh', json={
            'refresh_token': refresh_token,
        })
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'access_token' in data['data']

    def test_refresh_token_invalid(self, client):
        """刷新 — 无效 token"""
        response = client.post('/api/auth/refresh', json={
            'refresh_token': 'invalid.refresh.token',
        })
        assert response.status_code == 401

    def test_refresh_token_missing(self, client):
        """刷新 — 缺少 token"""
        response = client.post('/api/auth/refresh', json={})
        assert response.status_code == 400

    def test_me_with_bearer_token(self, client, test_user, app):
        """Bearer Token 访问 /api/auth/me"""
        with app.app_context():
            access_token = generate_access_token(
                test_user.id, test_user.username, test_user.role
            )

        response = client.get('/api/auth/me', headers={
            'Authorization': f'Bearer {access_token}',
        })
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['data']['username'] == 'testuser'

    def test_me_without_auth(self, client):
        """无认证访问 /api/auth/me 应失败"""
        response = client.get('/api/auth/me')
        assert response.status_code == 401

    def test_me_with_api_key(self, client, test_user):
        """API Key 访问 /api/auth/me (兼容)"""
        response = client.get('/api/auth/me', headers={
            'X-API-Key': test_user.api_key,
        })
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

    def test_api_endpoint_with_bearer_token(self, client, test_user, app):
        """Bearer Token 访问其他 API 端点 (/api/users/me)"""
        with app.app_context():
            access_token = generate_access_token(
                test_user.id, test_user.username, test_user.role
            )

        response = client.get('/api/users/me', headers={
            'Authorization': f'Bearer {access_token}',
        })
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True


class TestJWTDisabledUser:
    """JWT 认证 — 禁用用户测试"""

    def test_disabled_user_login(self, client, app):
        """禁用用户无法通过 JWT 登录"""
        with app.app_context():
            user = User(
                username='disabled_user',
                email='disabled@test.com',
                role='viewer',
                is_active=False,
            )
            user.set_password('Disabled1')
            db.session.add(user)
            db.session.commit()

        response = client.post('/api/auth/login', json={
            'login_id': 'disabled_user',
            'password': 'Disabled1',
        })
        assert response.status_code == 401
