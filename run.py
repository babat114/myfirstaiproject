"""
============================================
应用启动入口
运行: python run.py
============================================
"""
import os
from app import create_app, db
from flask_migrate import upgrade

# 从环境变量获取配置，默认为开发环境
config_name = os.environ.get('FLASK_ENV', 'development')

app = create_app(config_name)


if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = config_name == 'development'

    # 自动执行数据库迁移
    with app.app_context():
        try:
            upgrade()
            print('[OK] Database migration check completed')
        except Exception as e:
            if config_name == 'production':
                # 生产环境: 迁移失败必须阻止启动, 避免 schema 不一致导致数据损坏
                print(f'[FATAL] Database migration failed: {e}')
                raise
            print(f'[WARN] Database migration skipped: {e}')

    print(f"""
============================================
  AI Model & Dataset Platform
============================================
  Environment: {config_name}
  URL:         http://{host}:{port}
  API:         http://{host}:{port}/api/
============================================
    """)

    app.run(host=host, port=port, debug=debug)
