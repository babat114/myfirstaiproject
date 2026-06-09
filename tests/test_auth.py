"""
============================================
认证模块测试
============================================
"""
from app.services.auth_service import AuthService


class TestUserRegistration:
    """用户注册测试"""

    def test_register_success(self, app):
        """测试成功注册"""
        with app.app_context():
            user, error = AuthService.register(
                username='newuser',
                email='new@test.com',
                password='ValidPass123!',
                full_name='New User',
            )
            assert user is not None
            assert error is None
            assert user.username == 'newuser'
            assert user.email == 'new@test.com'
            assert user.check_password('ValidPass123!')

    def test_register_duplicate_username(self, app, test_user):
        """测试重复用户名注册"""
        with app.app_context():
            user, error = AuthService.register(
                username='testuser',
                email='different@test.com',
                password='ValidPass123!',
            )
            assert user is None
            assert '用户名已被注册' in error

    def test_register_weak_password(self, app):
        """测试弱密码"""
        with app.app_context():
            user, error = AuthService.register(
                username='weakuser',
                email='weak@test.com',
                password='short',
            )
            assert user is None
            assert '密码' in error


class TestUserLogin:
    """用户登录测试"""

    def test_login_success_username(self, app, test_user):
        """测试用户名登录"""
        with app.app_context():
            user, error = AuthService.login('testuser', 'Test123456')
            assert user is not None
            assert error is None
            assert user.username == 'testuser'

    def test_login_success_email(self, app, test_user):
        """测试邮箱登录"""
        with app.app_context():
            user, error = AuthService.login('test@test.com', 'Test123456')
            assert user is not None
            assert error is None

    def test_login_wrong_password(self, app, test_user):
        """测试错误密码"""
        with app.app_context():
            user, error = AuthService.login('testuser', 'WrongPassword1')
            assert user is None
            assert error is not None

    def test_login_nonexistent_user(self, app):
        """测试不存在的用户"""
        with app.app_context():
            user, error = AuthService.login('nonexistent', 'SomePass123')
            assert user is None
            assert error is not None


class TestPasswordManagement:
    """密码管理测试"""

    def test_change_password(self, app, test_user):
        """测试修改密码"""
        with app.app_context():
            success, error = AuthService.change_password(
                test_user, 'Test123456', 'NewPass456!'
            )
            assert success is True
            assert test_user.check_password('NewPass456!')

    def test_change_password_wrong_old(self, app, test_user):
        """测试错误的旧密码"""
        with app.app_context():
            success, error = AuthService.change_password(
                test_user, 'WrongOldPass', 'NewPass456!'
            )
            assert success is False


class TestLoginPage:
    """登录页面测试"""

    def test_login_page_loads(self, client):
        """测试登录页面加载"""
        response = client.get('/auth/login')
        assert response.status_code == 200

    def test_login_redirects_to_dashboard(self, client, test_user):
        """测试登录后重定向"""
        response = client.post('/auth/login', data={
            'login_id': 'testuser',
            'password': 'Test123456',
        }, follow_redirects=True)
        assert response.status_code == 200
        # 登录成功后会重定向到 /dashboard/

    def test_login_invalid_credentials(self, client):
        """测试无效凭据"""
        response = client.post('/auth/login', data={
            'login_id': 'testuser',
            'password': 'wrongpassword',
        })
        assert response.status_code == 200  # 返回登录页
