/**
 * ============================================
 * AI Platform - 通用前端脚本 v2.0
 * ============================================
 */

document.addEventListener('DOMContentLoaded', function () {
    // 自动关闭告警
    initAlertDismiss();

    // 文件输入框显示文件名
    initFileInputs();

    // 确认删除
    initConfirmDialogs();

    // 表格行点击
    initClickableRows();

    // 自动调整 textarea 高度
    initAutoResize();

    // 初始化所有 tooltip
    initTooltips();

    // 数字动画
    initCountUp();
});

// ============ 告警自动关闭 ============
function initAlertDismiss() {
    document.querySelectorAll('.alert-dismissible').forEach(alert => {
        setTimeout(() => {
            try {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            } catch(e) {}
        }, 5000);
    });
}

// ============ 文件输入 ============
function initFileInputs() {
    document.querySelectorAll('input[type="file"]').forEach(input => {
        input.addEventListener('change', function () {
            const fileName = this.files[0]?.name;
            const label = this.closest('.card-body')?.querySelector('.file-name');
            if (label && fileName) {
                label.textContent = fileName;
                label.classList.remove('text-muted');
            }
            // 也更新相邻的 .selected-file 元素
            const display = this.parentElement.querySelector('.selected-file');
            if (display && fileName) {
                display.textContent = '已选择: ' + fileName;
            }
        });
    });
}

// ============ 全局确认弹窗 (替代浏览器原生 confirm) ============

/**
 * 显示居中确认弹窗，返回 Promise<boolean>
 *
 * 用法1 - async/await:
 *   const ok = await showConfirmModal('确定删除？');
 *   if (ok) { ... }
 *
 * 用法2 - data-confirm 属性 (自动拦截):
 *   <button data-confirm="确定删除此模型？" onclick="deleteModel()">删除</button>
 *   <form data-confirm="确定删除？" method="POST" action="...">...</form>
 */
function showConfirmModal(message, title, confirmText, confirmClass) {
    return new Promise((resolve) => {
        const modalEl = document.getElementById('globalConfirmModal');
        if (!modalEl) {
            // 回退到原生 confirm
            resolve(confirm(message || '确定执行此操作？'));
            return;
        }

        const modal = bootstrap.Modal.getOrCreateInstance(modalEl);
        document.getElementById('confirmModalTitle').textContent = title || '确认操作';
        document.getElementById('confirmModalMessage').textContent = message || '确定执行此操作？';
        const btn = document.getElementById('confirmModalBtn');
        btn.textContent = confirmText || '确认删除';
        btn.className = 'btn btn-sm ' + (confirmClass || 'btn-danger');

        // 移除旧的事件监听
        const newBtn = btn.cloneNode(true);
        btn.parentNode.replaceChild(newBtn, btn);

        // 确认按钮
        newBtn.addEventListener('click', () => {
            modal.hide();
            resolve(true);
        });

        // 取消 (通过 data-bs-dismiss 或点击背景)
        const onCancel = () => resolve(false);
        modalEl.addEventListener('hidden.bs.modal', onCancel, { once: true });

        modal.show();
    });
}

// 挂载到全局
window.showConfirmModal = showConfirmModal;

// 全局拦截: [data-confirm] 属性 — 支持 form/button/a 元素
function initConfirmDialogs() {
    document.addEventListener('click', function (e) {
        const target = e.target.closest('[data-confirm]');
        if (!target) return;

        const message = target.dataset.confirm;
        if (!message) return;

        e.preventDefault();
        e.stopPropagation();

        showConfirmModal(message).then(confirmed => {
            if (!confirmed) return;

            // 如果是表单内的按钮 → 提交表单
            if (target.tagName === 'BUTTON' && target.form) {
                target.form.submit();
                return;
            }

            // 如果是链接 → 导航
            if (target.tagName === 'A' && target.href) {
                window.location = target.href;
                return;
            }

            // 如果是 form → 直接提交
            if (target.tagName === 'FORM') {
                target.submit();
                return;
            }

            // 其他情况：重新派发点击事件（去掉 data-confirm 属性后）
            target.removeAttribute('data-confirm');
            target.click();
            target.setAttribute('data-confirm', message);
        });
    }, true); // 使用捕获阶段以在默认行为之前拦截

    // 同样拦截表单 submit (onSubmit form 也支持 data-confirm)
    document.addEventListener('submit', function (e) {
        const form = e.target.closest('form[data-confirm]');
        if (!form) return;

        // 避免重复拦截（如果 click handler 已经处理了）
        if (form.dataset.confirmHandled === 'true') {
            delete form.dataset.confirmHandled;
            return;
        }

        e.preventDefault();
        showConfirmModal(form.dataset.confirm).then(confirmed => {
            if (confirmed) {
                form.dataset.confirmHandled = 'true';
                form.submit();
            }
        });
    }, true);
}

// ============ 可点击行 ============
function initClickableRows() {
    document.querySelectorAll('tr[data-href]').forEach(row => {
        row.style.cursor = 'pointer';
        row.classList.add('clickable-row');
        row.addEventListener('click', function (e) {
            if (e.target.tagName === 'BUTTON' || e.target.tagName === 'A' ||
                e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' ||
                e.target.closest('button') || e.target.closest('a') ||
                e.target.closest('form')) {
                return;
            }
            window.location = this.dataset.href;
        });
    });
}

// ============ 自动调整 textarea ============
function initAutoResize() {
    document.querySelectorAll('textarea.auto-resize').forEach(ta => {
        ta.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    });
}

// ============ Tooltip 初始化 ============
function initTooltips() {
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
        new bootstrap.Tooltip(el);
    });
}

// ============ 密码可见性切换 (全局函数) ============
window.togglePassword = function (inputId) {
    const el = document.getElementById(inputId);
    if (el) {
        el.type = el.type === 'password' ? 'text' : 'password';
    }
};

// ============ 日期格式化 (全局函数) ============
window.formatDate = function (dateStr) {
    if (!dateStr) return '-';
    const d = new Date(dateStr);
    return d.toLocaleString('zh-CN', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit'
    });
};

// ============ API 请求封装 ============
async function apiRequest(url, options = {}) {
    const defaultOptions = {
        headers: { 'Content-Type': 'application/json' },
    };
    const mergedOptions = { ...defaultOptions, ...options };

    try {
        const response = await fetch(url, mergedOptions);
        const data = await response.json();
        if (!response.ok) throw new Error(data.message || '请求失败');
        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// ============ 居中通知 (替代 ElMessage 放在屏幕正中) ============

/**
 * 在屏幕中央显示浮动通知，自动消失
 *
 * @param {string} message - 消息文本
 * @param {string} type - success | error | warning | info
 * @param {number} duration - 显示毫秒数 (默认 3500)
 *
 * 用法:
 *   showCenteredMessage('训练任务已创建并启动！', 'success');
 *   showCenteredMessage('操作失败，请重试。', 'error', 5000);
 */
function showCenteredMessage(message, type, duration) {
    type = type || 'info';
    duration = duration || 3500;

    const container = document.getElementById('centeredMessages');
    if (!container) {
        // 回退
        if (typeof showToast === 'function') showToast(message, type, duration);
        else alert(message);
        return;
    }

    const icons = {
        success: 'bi-check-circle-fill',
        error: 'bi-x-circle-fill',
        warning: 'bi-exclamation-triangle-fill',
        info: 'bi-info-circle-fill',
    };

    const el = document.createElement('div');
    el.className = 'centered-message msg-' + type;
    el.innerHTML =
        '<i class="bi ' + (icons[type] || icons.info) + ' msg-icon"></i>' +
        '<span class="msg-text">' + message + '</span>' +
        '<i class="bi bi-x msg-close"></i>';

    // 关闭按钮
    el.querySelector('.msg-close').addEventListener('click', function () {
        removeMsg(el);
    });

    container.appendChild(el);

    // 自动消失
    const timer = setTimeout(function () { removeMsg(el); }, duration);
    el._msgTimer = timer;

    // 鼠标悬停时暂停计时
    el.addEventListener('mouseenter', function () { clearTimeout(el._msgTimer); });
    el.addEventListener('mouseleave', function () {
        el._msgTimer = setTimeout(function () { removeMsg(el); }, 1500);
    });
}

function removeMsg(el) {
    if (el._removing) return;
    el._removing = true;
    clearTimeout(el._msgTimer);
    el.classList.add('fade-out');
    setTimeout(function () {
        if (el.parentNode) el.parentNode.removeChild(el);
    }, 260);
}

// 挂载到全局
window.showCenteredMessage = showCenteredMessage;

// ============ Toast 通知 (增强版) ============
function showToast(message, type = 'info', duration = 3500) {
    // 确保容器存在
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    const id = 'toast-' + Date.now();
    const colors = {
        success: 'bg-success text-white',
        error: 'bg-danger text-white',
        warning: 'bg-warning',
        info: 'bg-info text-white',
    };

    const icons = {
        success: 'bi-check-circle',
        error: 'bi-x-circle',
        warning: 'bi-exclamation-triangle',
        info: 'bi-info-circle',
    };

    const html = `
        <div id="${id}" class="toast align-items-center ${colors[type] || colors.info} border-0 animate-fade-in" role="alert">
            <div class="d-flex">
                <div class="toast-body d-flex align-items-center gap-2">
                    <i class="bi ${icons[type] || icons.info}"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

    container.insertAdjacentHTML('beforeend', html);
    const toastEl = document.getElementById(id);
    const toast = new bootstrap.Toast(toastEl, { delay: duration });
    toast.show();
    toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
}

// ============ 数字增长动画 ============
function initCountUp() {
    document.querySelectorAll('.count-up').forEach(el => {
        const target = parseInt(el.dataset.target) || parseInt(el.textContent) || 0;
        const duration = 1000;
        const start = performance.now();
        const initial = 0;

        function update(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3); // ease-out
            const current = Math.round(initial + (target - initial) * eased);
            el.textContent = current;
            if (progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    });
}

// ============ 复制到剪贴板 ============
window.copyToClipboard = function(text, buttonEl) {
    navigator.clipboard.writeText(text).then(() => {
        const origHTML = buttonEl?.innerHTML || '';
        if (buttonEl) {
            buttonEl.innerHTML = '<i class="bi bi-check"></i> 已复制';
            setTimeout(() => { buttonEl.innerHTML = origHTML; }, 2000);
        }
        showToast('已复制到剪贴板', 'success');
    }).catch(() => {
        showToast('复制失败', 'error');
    });
};

// ============ 平滑滚动 ============
window.scrollToElement = function(id) {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
};

// ============ 批量选择 checkbox ============
window.toggleSelectAll = function(checkbox, targetName) {
    document.querySelectorAll(`input[name="${targetName}"]`).forEach(cb => {
        cb.checked = checkbox.checked;
    });
};
