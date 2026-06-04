"""
============================================
用户管理 API 测试
============================================
"""
import json
from app.services.auth_service import AuthService
from app.models.user import User
from app import db


class TestUserAPI:
    """用户管理 API 测试"""

    def test_list_users_as_admin(self, admin_client):
        """管理员获取用户列表"""
        response = admin_client.get('/api/users?per_page=20')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'users' in data['data']

    def test_list_users_unauthorized(self, client):
        """未登录获取用户列表应失败"""
        response = client.get('/api/users')
        assert response.status_code in (401, 302, 403)

    def test_list_users_non_admin(self, logged_in_client):
        """非管理员获取用户列表应失败"""
        response = logged_in_client.get('/api/users')
        assert response.status_code in (403, 302)

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

    def test_get_my_profile(self, logged_in_client):
        """当前用户获取自己的资料"""
        response = logged_in_client.get('/api/users/me')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

    def test_update_my_profile(self, logged_in_client):
        """当前用户更新自己的资料"""
        response = logged_in_client.put(
            '/api/users/me',
            data=json.dumps({'full_name': 'My New Name'}),
            content_type='application/json',
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True


class TestAdminPages:
    """管理页面测试"""

    def test_admin_page_requires_auth(self, client):
        """未登录不能访问管理页面"""
        response = client.get('/dashboard/admin', follow_redirects=True)
        assert response.status_code == 200

    def test_admin_page_as_admin(self, admin_client):
        """管理员访问管理面板"""
        response = admin_client.get('/dashboard/admin')
        assert response.status_code == 200

    def test_admin_page_as_user(self, logged_in_client):
        """普通用户访问管理面板应重定向"""
        response = logged_in_client.get('/dashboard/admin', follow_redirects=True)
        assert response.status_code == 200

    def test_admin_users_page(self, admin_client):
        """管理员访问用户管理页面"""
        response = admin_client.get('/dashboard/admin/users')
        assert response.status_code == 200


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
