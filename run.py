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


# 添加根路由 (首页)
@app.route('/')
def index():
    """首页 - 登录前展示介绍页，登录后跳转仪表盘"""
    from flask import render_template, redirect, url_for
    from flask_login import current_user
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    return render_template('index.html')


if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = config_name == 'development'

    # 自动执行数据库迁移 (开发环境)
    with app.app_context():
        try:
            upgrade()
            print('[OK] Database migration check completed')
        except Exception as e:
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
