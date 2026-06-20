"""
============================================
认证模块测试 (参数化优化 v1.0)
============================================
"""
import pytest
from app.services.auth_service import AuthService


class TestUserRegistration:
    """用户注册测试"""

    @pytest.mark.parametrize("username,email,password,full_name,"
                             "expect_user_not_none,expect_error_keyword,create_conflict_user", [
        # Happy path
        ("newuser",   "new@test.com",    "ValidPass123!", "New User", True,  None,             False),
        # Duplicate username
        ("testuser",  "different@test.com", "ValidPass123!", None,    False, "用户名已被注册", True),
        # Weak password
        ("weakuser",  "weak@test.com",   "short",         None,       False, "密码",           False),
    ])
    def test_register(self, app, test_user, username, email, password, full_name,
                      expect_user_not_none, expect_error_keyword, create_conflict_user):
        """参数化: 成功注册 / 重复用户名 / 弱密码"""
        with app.app_context():
            # 重复用户名: test_user fixture 已创建 testuser, 无需额外操作
            user, error = AuthService.register(
                username=username,
                email=email,
                password=password,
                full_name=full_name,
            )
            assert (user is not None) == expect_user_not_none
            if expect_error_keyword:
                assert expect_error_keyword in error
            else:
                assert error is None
                assert user.username == username
                assert user.email == email
                assert user.check_password(password)


class TestUserLogin:
    """用户登录测试"""

    @pytest.mark.parametrize("login_id,password,expect_user_not_none,expect_error_not_none", [
        # 用户名登录
        ("testuser",      "Test123456",    True,  False),
        # 邮箱登录
        ("test@test.com", "Test123456",    True,  False),
        # 错误密码
        ("testuser",      "WrongPassword1", False, True),
        # 不存在的用户
        ("nonexistent",   "SomePass123",   False, True),
    ])
    def test_login(self, app, test_user, login_id, password,
                   expect_user_not_none, expect_error_not_none):
        """参数化: 用户名/邮箱登录 / 错误密码 / 不存在用户"""
        with app.app_context():
            user, error = AuthService.login(login_id, password)
            assert (user is not None) == expect_user_not_none
            assert (error is not None) == expect_error_not_none
            if user is not None:
                if '@' in login_id:
                    assert user.email == login_id
                else:
                    assert user.username == login_id


class TestPasswordManagement:
    """密码管理测试"""

    @pytest.mark.parametrize("old_password,new_password,expect_success", [
        ("Test123456",  "NewPass456!", True),
        ("WrongOldPass", "NewPass456!", False),
    ])
    def test_change_password(self, app, test_user, old_password, new_password,
                             expect_success):
        """参数化: 正确旧密码 / 错误旧密码"""
        with app.app_context():
            success, error = AuthService.change_password(
                test_user, old_password, new_password
            )
            assert success == expect_success
            if expect_success:
                assert test_user.check_password(new_password)


class TestLoginPage:
    """登录页面测试"""

    @pytest.mark.parametrize("method,data,expect_status,expect_redirect", [
        # GET 登录页
        ("get",  None,                                             200, False),
        # POST 正确凭据
        ("post", {"login_id": "testuser", "password": "Test123456"}, 200, True),
        # POST 错误凭据
        ("post", {"login_id": "testuser", "password": "wrongpassword"}, 200, False),
    ])
    def test_login_endpoint(self, client, test_user, method, data, expect_status,
                            expect_redirect):
        """参数化: 登录页加载 / 成功登录重定向 / 无效凭据"""
        if method == "get":
            response = client.get('/auth/login')
        else:
            response = client.post('/auth/login', data=data,
                                   follow_redirects=expect_redirect)
        assert response.status_code == expect_status
