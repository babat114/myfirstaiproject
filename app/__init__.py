"""
============================================
AI Model & Dataset Management Platform
应用工厂 - 创建和配置 Flask 应用
============================================
"""
import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_cors import CORS
from config import get_config

# 初始化扩展 (不绑定到应用)
db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()
cors = CORS()

# 全局日志记录器
logger = logging.getLogger(__name__)


def create_app(config_name=None):
    """
    应用工厂函数
    根据配置名称创建并配置 Flask 应用实例
    """
    app = Flask(__name__)

    # 加载配置
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')

    config_class = get_config()
    app.config.from_object(config_class)

    # 初始化扩展
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(app, resources={r"/api/*": {"origins": "*"}})

    # 登录管理器配置
    login_manager.login_view = 'auth.login'
    login_manager.login_message = '请先登录以访问此页面。'
    login_manager.login_message_category = 'warning'
    login_manager.session_protection = 'strong'

    # 确保上传目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'datasets'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'models'), exist_ok=True)

    # 配置日志
    configure_logging(app)

    # 注册蓝图
    register_blueprints(app)

    # 注册错误处理器
    register_error_handlers(app)

    # 注册上下文处理器
    register_context_processors(app)

    # 创建数据库表 (开发环境)
    with app.app_context():
        from app.models import user, dataset, model_record, training_job
        db.create_all()

    logger.info(f"应用启动成功 - 环境: {config_name}")
    return app


def configure_logging(app):
    """配置应用日志"""
    import colorlog

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))

    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))
    app.logger.setLevel(log_level)
    app.logger.addHandler(handler)


def register_blueprints(app):
    """注册所有蓝图"""
    from app.routes.auth import auth_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.datasets import datasets_bp
    from app.routes.models import models_bp
    from app.routes.training import training_bp

    # Web 页面路由
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(datasets_bp, url_prefix='/datasets')
    app.register_blueprint(models_bp, url_prefix='/models')
    app.register_blueprint(training_bp, url_prefix='/training')

    # RESTful API 路由
    from app.routes.api.datasets import datasets_api_bp
    from app.routes.api.models import models_api_bp
    from app.routes.api.training import training_api_bp

    app.register_blueprint(datasets_api_bp, url_prefix='/api/datasets')
    app.register_blueprint(models_api_bp, url_prefix='/api/models')
    app.register_blueprint(training_api_bp, url_prefix='/api/training')


def register_error_handlers(app):
    """注册错误处理"""
    from flask import render_template, jsonify
    from werkzeug.exceptions import HTTPException

    @app.errorhandler(404)
    def not_found(e):
        if app.config.get('DEBUG'):
            return jsonify({'error': 'Not Found', 'message': str(e)}), 404
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_error(e):
        db.session.rollback()
        app.logger.error(f"Internal Server Error: {e}")
        return render_template('errors/500.html'), 500

    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        return render_template('errors/error.html', error=e), e.code


def register_context_processors(app):
    """注册模板上下文处理器"""
    from datetime import datetime

    @app.context_processor
    def inject_now():
        return {'now': datetime.now()}

    @app.context_processor
    def inject_config():
        return {'app_name': 'AI Platform', 'version': '1.0.0'}
