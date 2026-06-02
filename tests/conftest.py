"""
============================================
测试配置
pytest fixtures 和共享配置
============================================
"""
import pytest
from app import create_app, db
from app.models.user import User


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
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return app.test_client()


@pytest.fixture
def runner(app):
    """创建 CLI 运行器"""
    return app.test_cli_runner()


@pytest.fixture
def test_user(app):
    """创建测试用户"""
    with app.app_context():
        user = User(
            username='testuser',
            email='test@test.com',
            full_name='Test User',
            role='researcher',
            is_active=True,
            is_verified=True,
        )
        user.set_password('Test123456')
        db.session.add(user)
        db.session.commit()
        return user


@pytest.fixture
def test_admin(app):
    """创建测试管理员"""
    with app.app_context():
        admin = User(
            username='testadmin',
            email='admin@test.com',
            full_name='Test Admin',
            role='admin',
            is_active=True,
            is_verified=True,
        )
        admin.set_password('Admin123456')
        db.session.add(admin)
        db.session.commit()
        return admin


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
