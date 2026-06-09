"""
============================================
应用配置文件
管理开发、测试、生产环境的配置
============================================
"""
import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()


class Config:
    """基础配置类"""
    # Flask 密钥 — 生产环境必须设置环境变量，开发环境自动随机生成
    _raw_secret = os.environ.get('SECRET_KEY')
    if _raw_secret:
        SECRET_KEY = _raw_secret
    elif os.environ.get('FLASK_ENV', 'development') == 'development':
        import secrets as _secrets
        SECRET_KEY = _secrets.token_hex(32)
    else:
        raise RuntimeError(
            '生产环境必须设置 SECRET_KEY 环境变量。'
            '请在 .env 中设置 SECRET_KEY=<your-secure-key>'
        )

    # 数据库配置
    MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', 'your_password')
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_PORT = os.environ.get('MYSQL_PORT', '3306')
    MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE', 'ai_platform')

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
        "?charset=utf8mb4"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'echo': False,
    }

    # 会话配置 — 安全优先: 默认开启 Secure (仅开发环境关闭)
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)
    SESSION_COOKIE_SECURE = True  # 默认开启 HTTPS-only
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # 文件上传配置
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    ALLOWED_EXTENSIONS = {'csv', 'json', 'txt', 'xlsx', 'parquet', 'jpg', 'png', 'npy'}

    # 分页配置
    ITEMS_PER_PAGE = 15

    # CSRF 保护
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600  # 1小时

    # JWT 配置 — 生产环境必须设置环境变量，开发环境自动随机生成
    _raw_jwt_secret = os.environ.get('JWT_SECRET_KEY')
    if _raw_jwt_secret:
        JWT_SECRET_KEY = _raw_jwt_secret
    elif os.environ.get('FLASK_ENV', 'development') == 'development':
        import secrets as _jwt_secrets
        JWT_SECRET_KEY = _jwt_secrets.token_hex(32)
    else:
        raise RuntimeError(
            '生产环境必须设置 JWT_SECRET_KEY 环境变量。'
            '请在 .env 中设置 JWT_SECRET_KEY=<your-secure-key>'
        )
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=2)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)

    # 日志配置
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

    # 训练执行引擎配置
    TRAINING_MAX_WORKERS = int(os.environ.get('TRAINING_MAX_WORKERS', '2'))


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    SESSION_COOKIE_SECURE = False  # 本地开发无 HTTPS
    SQLALCHEMY_ENGINE_OPTIONS = {
        **Config.SQLALCHEMY_ENGINE_OPTIONS,
        'echo': True,  # 开发环境打印 SQL
    }


class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SQLALCHEMY_ENGINE_OPTIONS = {}  # SQLite 不接受 pool_size 等参数
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    SESSION_COOKIE_SECURE = True


# 配置映射
config_map = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}


def get_config(env: str = None):
    """获取当前环境配置"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    return config_map.get(env, config_map['default'])
