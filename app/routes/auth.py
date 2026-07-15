"""
============================================
认证路由
处理用户注册、登录、注销和个人资料
============================================
"""

from urllib.parse import urlparse

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app.services.auth_service import AuthService
from app.utils.decorators import rate_limit

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'])
@rate_limit(max_calls=20, period=60)  # 每IP每分钟最多20次登录尝试
def login():
    """用户登录页面"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))

    if request.method == 'POST':
        login_id = request.form.get('login_id', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'

        if not login_id or not password:
            flash('请填写所有字段。', 'danger')
            return render_template('login.html')

        user, error = AuthService.login(login_id, password)

        if error:
            flash(error, 'danger')
            return render_template('login.html')

        login_user(user, remember=remember)
        next_page = request.args.get('next')
        if next_page:
            # 防止开放重定向: 只允许站内链接 (相对路径或同源)
            parsed = urlparse(next_page)
            if parsed.netloc or parsed.scheme:
                next_page = None
        flash(f'欢迎回来，{user.username}！', 'success')
        return redirect(next_page or url_for('dashboard.index'))

    return render_template('login.html')


@auth_bp.route('/register', methods=['GET', 'POST'])
@rate_limit(max_calls=5, period=300)  # 每IP每5分钟最多5次注册
def register():
    """用户注册页面"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '').strip() or None

        # 表单验证
        errors = []
        if not username or len(username) < 3:
            errors.append('用户名至少需要 3 个字符。')
        if not email or '@' not in email:
            errors.append('请输入有效的邮箱地址。')
        if password != confirm_password:
            errors.append('两次输入的密码不一致。')

        if errors:
            for err in errors:
                flash(err, 'danger')
            return render_template('register.html')

        user, error = AuthService.register(
            username=username,
            email=email,
            password=password,
            full_name=full_name,
        )

        if error:
            flash(error, 'danger')
            return render_template('register.html')

        login_user(user)
        flash(f'注册成功！欢迎加入，{user.username}！', 'success')
        return redirect(url_for('dashboard.index'))

    return render_template('register.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """用户注销"""
    logout_user()
    flash('您已成功注销。', 'info')
    return redirect(url_for('auth.login'))


@auth_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """用户个人资料页面"""
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'update_profile':
            data = {
                'full_name': request.form.get('full_name', '').strip() or None,
                'bio': request.form.get('bio', '').strip() or None,
                'organization': request.form.get('organization', '').strip() or None,
            }
            success, error = AuthService.update_profile(current_user, data)
            if success:
                flash('个人资料已更新。', 'success')
            else:
                flash(error, 'danger')

        elif action == 'change_password':
            old_pw = request.form.get('old_password', '')
            new_pw = request.form.get('new_password', '')
            success, error = AuthService.change_password(current_user, old_pw, new_pw)
            if success:
                flash('密码已修改。', 'success')
            else:
                flash(error, 'danger')

        elif action == 'regenerate_api_key':
            new_key = AuthService.regenerate_api_key(current_user)
            flash(f'新的 API 密钥: {new_key}', 'success')

        return redirect(url_for('auth.profile'))

    return render_template('profile.html', user=current_user)
