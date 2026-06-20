"""
============================================
JWT 认证测试 (参数化优化 v1.0)
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

    @pytest.mark.parametrize("gen_func,args,is_pair,check_jwt_format", [
        (generate_access_token,  (1, "testuser", "researcher"), False, True),
        (generate_refresh_token, (1, "testuser"),               False, True),
        (generate_token_pair,    (1, "testuser", "researcher"), True,  False),
    ])
    def test_generate_token(self, app, gen_func, args, is_pair, check_jwt_format):
        """参数化: access / refresh / token pair"""
        with app.app_context():
            result = gen_func(*args)
            if is_pair:
                assert isinstance(result, dict)
                assert 'access_token' in result
                assert 'refresh_token' in result
                assert result['token_type'] == 'Bearer'
                assert result['expires_in'] > 0
            else:
                assert isinstance(result, str)
                if check_jwt_format:
                    assert result.count('.') == 2

    @pytest.mark.parametrize("generate_type,decode_func,expect_error,expected_vals", [
        ("access",  decode_access_token,  False, {"sub": "1", "username": "testuser", "role": "researcher", "type": "access"}),
        ("refresh", decode_refresh_token, False, {"sub": "1", "type": "refresh"}),
        ("refresh", decode_access_token,  True,  "类型不匹配"),
        ("invalid", decode_access_token,  True,  None),
    ])
    def test_decode_token(self, app, generate_type, decode_func, expect_error, expected_vals):
        """参数化: 解码 access / refresh / 类型不匹配 / 无效token"""
        with app.app_context():
            if generate_type == "access":
                token = generate_access_token(1, "testuser", "researcher")
            elif generate_type == "refresh":
                token = generate_refresh_token(1, "testuser")
            else:
                token = "not.a.real.token"

            payload, error = decode_func(token)
            if expect_error:
                assert error is not None
                if isinstance(expected_vals, str):
                    assert expected_vals in error
            else:
                assert error is None
                for k, v in expected_vals.items():
                    assert payload[k] == v

    @pytest.mark.parametrize("headers,expected", [
        ({"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.test.test"}, "eyJhbGciOiJIUzI1NiJ9.test.test"),
        ({}, None),
    ])
    def test_extract_token(self, app, headers, expected):
        """参数化: 有 / 无 Authorization header"""
        with app.test_request_context(headers=headers):
            token = extract_token_from_header()
            assert token == expected


class TestJWTAuthAPI:
    """JWT API 认证接口测试"""

    @pytest.mark.parametrize("json_data,expected_status,expect_success", [
        ({"login_id": "testuser", "password": "Test123456"},      200, True),
        ({"login_id": "testuser", "password": "WrongPassword123"}, 401, False),
        ({},                                                      400, None),
    ])
    def test_login(self, client, test_user, json_data, expected_status, expect_success):
        """参数化: 成功 / 密码错误 / 缺少字段"""
        response = client.post('/api/auth/login', json=json_data)
        assert response.status_code == expected_status
        data = json.loads(response.data)
        if expect_success is not None:
            assert data['success'] is expect_success
        if expect_success:
            assert 'access_token' in data['data']
            assert 'refresh_token' in data['data']

    @pytest.mark.parametrize("token_type,expected_status", [
        ("valid",   200),
        ("invalid", 401),
        ("missing", 400),
    ])
    def test_refresh(self, client, test_user, app, token_type, expected_status):
        """参数化: 有效 / 无效 / 缺少 refresh token"""
        if token_type == "valid":
            with app.app_context():
                rt = generate_refresh_token(test_user.id, test_user.username)
            payload = {"refresh_token": rt}
        elif token_type == "invalid":
            payload = {"refresh_token": "invalid.refresh.token"}
        else:
            payload = {}

        response = client.post('/api/auth/refresh', json=payload)
        assert response.status_code == expected_status
        if expected_status == 200:
            data = json.loads(response.data)
            assert data['success'] is True
            assert 'access_token' in data['data']

    @pytest.mark.parametrize("auth_method,expected_status", [
        ("bearer", 200),
        ("none",   401),
        ("apikey", 200),
    ])
    def test_me_auth(self, client, test_user, app, auth_method, expected_status):
        """参数化: Bearer / 无认证 / API Key"""
        headers = {}
        if auth_method == "bearer":
            with app.app_context():
                token = generate_access_token(
                    test_user.id, test_user.username, test_user.role
                )
            headers["Authorization"] = f"Bearer {token}"
        elif auth_method == "apikey":
            headers["X-API-Key"] = test_user.api_key

        response = client.get('/api/auth/me', headers=headers)
        assert response.status_code == expected_status
        if expected_status == 200:
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
