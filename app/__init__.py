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
from flask_wtf.csrf import CSRFProtect
import colorlog
from config import get_config

# 初始化扩展 (不绑定到应用)
db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()
cors = CORS()
csrf = CSRFProtect()

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

    config_class = get_config(config_name)
    app.config.from_object(config_class)

    # 初始化扩展
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    # CORS 配置从环境变量读取
    cors_origins = app.config.get('CORS_ORIGINS', '*')
    cors.init_app(app, resources={r"/api/*": {"origins": cors_origins}})
    csrf.init_app(app)

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

    # 注册蓝图 + CSRF 豁免
    register_blueprints(app)

    # 注册根路由 (/)
    register_home_route(app)

    # 向后兼容: /api/* → /api/v1/*
    setup_api_compat_middleware(app)

    # 注册错误处理器
    register_error_handlers(app)

    # 注册上下文处理器
    register_context_processors(app)

    # 注册健康检查端点
    register_health_check(app)

    # 注册 Swagger/OpenAPI 文档
    configure_swagger(app)

    # 注意: 数据库表通过 Flask-Migrate 管理
    # 首次部署运行: flask db upgrade
    # 生成迁移: flask db migrate -m "描述"

    logger.info(f"应用启动成功 - 环境: {config_name}")
    return app


def configure_logging(app):
    """配置应用日志"""
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
    """注册所有蓝图 + CSRF 豁免"""
    from app.routes.auth import auth_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.datasets import datasets_bp
    from app.routes.models import models_bp
    from app.routes.training import training_bp
    from app.routes.comments import comments_bp

    # Web 页面路由
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(datasets_bp, url_prefix='/datasets')
    app.register_blueprint(models_bp, url_prefix='/models')
    app.register_blueprint(training_bp, url_prefix='/training')
    app.register_blueprint(comments_bp, url_prefix='/comments')

    # RESTful API 路由
    from app.routes.api.auth import auth_api_bp
    from app.routes.api.datasets import datasets_api_bp
    from app.routes.api.models import models_api_bp
    from app.routes.api.training import training_api_bp
    from app.routes.api.stream import stream_bp
    from app.routes.api.users import users_api_bp
    from app.routes.api.comments import comments_api_bp

    # API v1 (当前版本)
    app.register_blueprint(auth_api_bp, url_prefix='/api/v1/auth')
    app.register_blueprint(datasets_api_bp, url_prefix='/api/v1/datasets')
    app.register_blueprint(models_api_bp, url_prefix='/api/v1/models')
    app.register_blueprint(training_api_bp, url_prefix='/api/v1/training')
    app.register_blueprint(stream_bp, url_prefix='/api/v1/stream')
    app.register_blueprint(users_api_bp, url_prefix='/api/v1/users')
    app.register_blueprint(comments_api_bp, url_prefix='/api/v1')

    # API 路由豁免 CSRF (使用 JWT/API Key 认证, 非 Session Cookie)
    csrf.exempt(auth_api_bp)
    csrf.exempt(datasets_api_bp)
    csrf.exempt(models_api_bp)
    csrf.exempt(training_api_bp)
    csrf.exempt(stream_bp)
    csrf.exempt(users_api_bp)
    csrf.exempt(comments_api_bp)


def register_home_route(app):
    """注册根路由: 已登录 → 仪表盘, 未登录 → 欢迎首页"""
    from flask import redirect, url_for, render_template
    from flask_login import current_user

    @app.route('/')
    def root():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard.index'))
        return render_template('index.html')


def setup_api_compat_middleware(app):
    """向后兼容: /api/* 内部重写为 /api/v1/*, 并附加 deprecation header"""
    _original_wsgi = app.wsgi_app

    _SWAGGER_PREFIXES = ('/api/docs',)

    def _api_v1_compat_middleware(environ, start_response):
        path = environ.get('PATH_INFO', '')
        if (path.startswith('/api/')
                and not path.startswith('/api/v1/')
                and not path.startswith(_SWAGGER_PREFIXES)):
            environ = dict(environ)
            environ['HTTP_X_API_ORIGINAL_PATH'] = path
            environ['PATH_INFO'] = path.replace('/api/', '/api/v1/', 1)
        return _original_wsgi(environ, start_response)

    app.wsgi_app = _api_v1_compat_middleware

    @app.after_request
    def _api_deprecation_header(response):
        from flask import request as req
        original = req.environ.get('HTTP_X_API_ORIGINAL_PATH', '')
        if (original.startswith('/api/')
                and not original.startswith('/api/v1/')
                and not original.startswith(_SWAGGER_PREFIXES)):
            response.headers['X-API-Deprecated'] = 'Use /api/v1/ instead. Will be removed in v2.0'
            response.headers['Sunset'] = 'Sat, 01 Jan 2027 00:00:00 GMT'
        return response


def register_error_handlers(app):
    """注册错误处理"""
    from flask import render_template, jsonify, request, flash, redirect, url_for
    from werkzeug.exceptions import HTTPException
    from flask_wtf.csrf import CSRFError

    @app.errorhandler(CSRFError)
    def handle_csrf_error(e):
        """CSRF 验证失败 — 刷新页面并提示用户"""
        # API 请求返回 JSON
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'message': 'CSRF 验证失败，请刷新页面后重试。',
            }), 400
        flash('安全验证已过期，请刷新页面后重试。', 'warning')
        return redirect(request.referrer or url_for('auth.login'))

    def _is_api_request():
        """判断当前请求是否为 API 请求"""
        return request.path.startswith('/api/')

    @app.errorhandler(404)
    def not_found(e):
        if _is_api_request():
            return jsonify({
                'success': False,
                'message': '请求的资源不存在。',
                'error': 'Not Found',
            }), 404
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_error(e):
        db.session.rollback()
        app.logger.error(f"Internal Server Error: {e}")
        if _is_api_request():
            return jsonify({
                'success': False,
                'message': '服务器内部错误，请稍后重试。',
                'error': 'Internal Server Error',
            }), 500
        return render_template('errors/500.html'), 500

    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        if _is_api_request():
            return jsonify({
                'success': False,
                'message': e.description or str(e),
                'error': e.name,
                'code': e.code,
            }), e.code
        return render_template('errors/error.html',
            error_code=e.code, error_title=e.name,
            error_message=e.description,
            error_color='text-muted'), e.code


def register_health_check(app):
    """注册健康检查端点 /health 和 /healthz (Kubernetes 兼容)"""
    from flask import jsonify
    from datetime import datetime, timezone

    @app.route('/health')
    @app.route('/healthz')
    def health_check():
        db_ok = True
        db_error = None
        try:
            db.session.execute(db.text('SELECT 1'))
        except Exception as e:
            db_ok = False
            db_error = str(e)

        status_code = 200 if db_ok else 503
        return jsonify({
            'status': 'healthy' if db_ok else 'unhealthy',
            'version': '1.0.0',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'checks': {
                'database': {
                    'status': 'ok' if db_ok else 'error',
                    'error': db_error,
                },
            },
        }), status_code


def register_context_processors(app):
    """注册模板上下文处理器"""
    from datetime import datetime

    @app.context_processor
    def inject_now():
        return {'now': datetime.now()}

    @app.context_processor
    def inject_config():
        from app.models.dataset import CATEGORY_LABELS
        # 模型类型中英文映射 (模板全局可用)
        model_type_labels = {
            'classification': '分类',
            'regression': '回归',
            'clustering': '聚类',
            'nlp': '自然语言处理',
            'computer_vision': '计算机视觉',
            'reinforcement': '强化学习',
            'generative': '生成式',
            'other': '其他',
        }
        return {
            'app_name': 'AI Platform',
            'version': '1.0.0',
            'model_type_labels': model_type_labels,
            'category_labels': CATEGORY_LABELS,
        }



def configure_swagger(app):
    """配置 Swagger/OpenAPI 文档 — 使用 flasgger

    Swagger UI:  GET /api/docs
    OpenAPI JSON: GET /api/docs/swagger.json

    端点文档通过 YAML docstring 定义在每个路由函数中,
    无需额外装饰器。flasgger 自动扫描已注册蓝图中包含
    YAML 文档字符串的路由。
    """
    try:
        from flasgger import Swagger
    except ImportError:
        logger.warning('flasgger 未安装, Swagger UI 不可用。安装: pip install flasgger')
        return

    swagger_config = {
        'headers': [],
        'specs': [
            {
                'endpoint': 'swagger',
                'route': '/api/docs/swagger.json',
                'rule_filter': lambda rule: True,
                'model_filter': lambda tag: True,
            }
        ],
        'static_url_path': '/api/docs/flasgger_static',
        'swagger_ui': True,
        'specs_route': '/api/docs/',
    }

    template = {
        'openapi': '3.0.3',
        'info': {
            'title': 'AI Model Training Platform API',
            'description': (
                'AI 模型训练管理平台 REST API — 支持数据集管理、'
                '多框架模型训练 (sklearn/PyTorch/TensorFlow/ONNX)、'
                '模型注册/测试/对比/导出、超参数自动调优。\n\n'
                '## 认证方式\n'
                '| 方式 | Header | 适用场景 |\n'
                '|------|--------|----------|\n'
                '| JWT Bearer | `Authorization: Bearer <token>` | API 调用 |\n'
                '| API Key | `X-API-Key: <key>` | 第三方集成 |\n'
                '| Session | Cookie (自动) | Web UI |\n\n'
                '## 响应格式\n'
                '所有成功响应: `{"success": true, "data": ...}`\n'
                '所有错误响应: `{"success": false, "message": "...", "error": "..."}`\n\n'
                '## 向后兼容\n'
                '`/api/*` 自动重写为 `/api/v1/*` (附带 deprecation header)。'
            ),
            'version': '1.0.0',
            'contact': {
                'name': 'AI Platform Team',
            },
        },
        'components': {
            'securitySchemes': {
                'BearerAuth': {
                    'type': 'http',
                    'scheme': 'bearer',
                    'bearerFormat': 'JWT',
                    'description': 'JWT Access Token (POST /api/v1/auth/login 获取)',
                },
                'ApiKeyAuth': {
                    'type': 'apiKey',
                    'in': 'header',
                    'name': 'X-API-Key',
                    'description': '用户 API Key (用户设置页获取)',
                },
            },
        },
        'security': [
            {'BearerAuth': []},
            {'ApiKeyAuth': []},
        ],
        'tags': [
            {'name': 'Auth', 'description': '认证 — 登录/Token刷新/当前用户'},
            {'name': 'Users', 'description': '用户管理 — 仅管理员'},
            {'name': 'Datasets', 'description': '数据集 — 上传/分析/导入/智能参数推荐'},
            {'name': 'Models', 'description': '模型 — 注册/预测/评估/导出/部署'},
            {'name': 'Training', 'description': '训练 — 任务管理/超参数调优/AI诊断'},
            {'name': 'Stream', 'description': '实时流 — SSE进度推送 (仅Session认证)'},
            {'name': 'Comments', 'description': '评论 — 模型评论/审核/回复'},
        ],
    }

    Swagger(app, config=swagger_config, template=template)
