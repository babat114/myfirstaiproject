"""
============================================
测试配置
pytest fixtures 和共享配置
============================================
"""
import pytest
from app import create_app, db
from app.models.user import User
from app.services.auth_service import AuthService


@pytest.fixture
def app():
    """创建测试应用"""
    app = create_app('testing')
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'WTF_CSRF_ENABLED': False,
        'SECRET_KEY': 'test-secret-key',
    })

    with app.app_context():
        # SQLite 默认不强制外键, 显式开启以匹配生产环境 MySQL 行为
        from sqlalchemy import text
        db.session.execute(text('PRAGMA foreign_keys = ON'))
        db.create_all()
        yield app
        db.session.remove()
        # drop_all 前关闭 FK 约束, 否则无法删除被引用的表
        db.session.execute(text('PRAGMA foreign_keys = OFF'))
        db.drop_all()


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return app.test_client()


# 测试用 API 密钥 (原始值, 用于请求头; 数据库中存储其 SHA256 哈希)
_TEST_USER_API_KEY = 'ak_testuserkey1234567890abcdef'
_TEST_ADMIN_API_KEY = 'ak_testadminkey1234567890abcdef'


@pytest.fixture
def test_user(app):
    """创建测试用户"""
    # 使用 app fixture 的上下文 (不要创建新的 context block)
    user = User(
        username='testuser',
        email='test@test.com',
        full_name='Test User',
        role='researcher',
        is_active=True,
        is_verified=True,
    )
    user.set_password('Test123456')
    # API Key 哈希存储 (与 AuthService.register 行为一致)
    user.api_key = AuthService._hash_api_key(_TEST_USER_API_KEY)
    db.session.add(user)
    db.session.commit()
    return user


@pytest.fixture
def test_admin(app):
    """创建测试管理员"""
    user = User(
        username='testadmin',
        email='admin@test.com',
        full_name='Test Admin',
        role='admin',
        is_active=True,
        is_verified=True,
    )
    user.set_password('Admin123456')
    # API Key 哈希存储 (与 AuthService.register 行为一致)
    user.api_key = AuthService._hash_api_key(_TEST_ADMIN_API_KEY)
    db.session.add(user)
    db.session.commit()
    return user


@pytest.fixture
def logged_in_client(client, test_user):
    """已登录的测试客户端"""
    with client:
        client.post('/auth/login', data={
            'login_id': 'testuser',
            'password': 'Test123456',
        }, follow_redirects=True)
        yield client


@pytest.fixture
def admin_client(client, test_admin):
    """管理员登录的测试客户端"""
    with client:
        client.post('/auth/login', data={
            'login_id': 'testadmin',
            'password': 'Admin123456',
        }, follow_redirects=True)
        yield client
