"""
============================================
用户管理 API 测试 (参数化优化 v1.0)
============================================
"""
import json
import pytest
from app.services.auth_service import AuthService
from app.models.user import User
from app import db


class TestUserAPI:
    """用户管理 API 测试"""

    @pytest.mark.parametrize("client_fixture,expected_status_or_range", [
        ("admin_client",     200),
        ("client",           (401, 302, 403)),
        ("logged_in_client", (403, 302)),
    ])
    def test_list_users_access(self, request, client_fixture,
                                expected_status_or_range):
        """参数化: 管理员 / 未登录 / 非管理员"""
        client = request.getfixturevalue(client_fixture)
        response = client.get('/api/users?per_page=20')
        if isinstance(expected_status_or_range, tuple):
            assert response.status_code in expected_status_or_range
        else:
            assert response.status_code == expected_status_or_range
            data = json.loads(response.data)
            assert data['success'] is True

    def test_get_user_detail_as_admin(self, admin_client, test_user):
        """管理员获取用户详情"""
        response = admin_client.get(f'/api/users/{test_user.id}')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

    def test_update_user_as_admin(self, admin_client, test_user):
        """管理员更新用户"""
        response = admin_client.put(
            f'/api/users/{test_user.id}',
            data=json.dumps({
                'full_name': 'Updated Name',
                'organization': 'Test Org',
                'role': 'admin',
            }),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

    def test_update_user_role_invalid(self, admin_client, test_user):
        """测试设置无效角色"""
        response = admin_client.put(
            f'/api/users/{test_user.id}/role',
            data=json.dumps({'role': 'superadmin'}),
            content_type='application/json',
        )
        assert response.status_code == 400

    def test_delete_user_as_admin(self, admin_client):
        """管理员删除用户"""
        victim = User(
            username='victim', email='victim@test.com', role='viewer'
        )
        victim.set_password('Victim999')
        db.session.add(victim)
        db.session.commit()
        vid = victim.id

        response = admin_client.delete(f'/api/users/{vid}')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

    def test_cannot_delete_self(self, admin_client, test_admin):
        """管理员不能删除自己"""
        response = admin_client.delete(f'/api/users/{test_admin.id}')
        assert response.status_code in (400, 500)

    @pytest.mark.parametrize("method,data,expected_status", [
        ("get",  None,                         200),
        ("put",  {"full_name": "My New Name"}, 200),
    ])
    def test_my_profile(self, logged_in_client, method, data, expected_status):
        """参数化: GET / PUT 自己的资料"""
        if method == "get":
            response = logged_in_client.get('/api/users/me')
        else:
            response = logged_in_client.put(
                '/api/users/me',
                data=json.dumps(data),
                content_type='application/json',
            )
        assert response.status_code == expected_status
        data_resp = json.loads(response.data)
        assert data_resp['success'] is True


class TestAdminPages:
    """管理页面测试"""

    @pytest.mark.parametrize("client_fixture,endpoint,expect_status,follow", [
        ("client",           "/dashboard/admin",       200, True),
        ("admin_client",     "/dashboard/admin",       200, False),
        ("logged_in_client", "/dashboard/admin",       200, True),
        ("admin_client",     "/dashboard/admin/users", 200, False),
    ])
    def test_admin_page_access(self, request, client_fixture, endpoint,
                                expect_status, follow):
        """参数化: 不同角色 / 不同页面"""
        client = request.getfixturevalue(client_fixture)
        response = client.get(endpoint, follow_redirects=follow)
        assert response.status_code == expect_status


class TestAuthServiceUserManagement:
    """AuthService 用户管理测试"""

    def test_list_users(self, test_user, test_admin):
        """测试列出用户"""
        result = AuthService.list_users()
        assert result['total'] >= 2

    def test_regenerate_api_key(self, test_user):
        """测试重新生成 API Key"""
        old_key = test_user.api_key
        new_key = AuthService.regenerate_api_key(test_user)
        assert new_key != old_key
        assert new_key.startswith('ak_')

    def test_delete_user(self, test_user):
        """测试删除用户"""
        u = User(username='temp', email='temp@t.com', role='viewer')
        u.set_password('Temp12345')
        db.session.add(u)
        db.session.commit()
        uid = u.id
        success = AuthService.delete_user(u)
        assert success is True
        assert db.session.get(User, uid) is None
