/**
 * ============================================
 * AI Platform — Main Scripts v5.0
 * GitHub-inspired: clean, functional, vanilla JS
 * ============================================
 */

document.addEventListener('DOMContentLoaded', function () {
    initAlertDismiss();
    initFileInputs();
    initConfirmDialogs();
    initClickableRows();
    initAutoResize();
    initTooltips();
    initCountUp();
    initFlashToCentered();   // Bootstrap alert → centered notification
    initBackToTop();          // Vanilla JS back-to-top button
});

// ============ HTML 转义工具函数 (防止 XSS) ============
var _escapeMap = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;' };
function escapeHtml(text) {
    if (typeof text !== 'string') return '';
    return text.replace(/[&<>"']/g, function (c) { return _escapeMap[c] || c; });
}

// ============ Auto-dismiss alerts after 5s ============
function initAlertDismiss() {
    document.querySelectorAll('.alert-dismissible').forEach(function (alert) {
        setTimeout(function () {
            try {
                var bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            } catch(e) {}
        }, 5000);
    });
}

// ============ File input — show selected filename ============
function initFileInputs() {
    document.querySelectorAll('input[type="file"]').forEach(function (input) {
        input.addEventListener('change', function () {
            var fileName = this.files[0]?.name;
            var label = this.closest('.card-body')?.querySelector('.file-name');
            if (label && fileName) {
                label.textContent = fileName;
                label.classList.remove('text-muted');
            }
            var display = this.parentElement?.querySelector('.selected-file');
            if (display && fileName) {
                display.textContent = fileName;
            }
        });
    });
}

// ============ Global confirm modal (replaces native confirm) ============
function showConfirmModal(message, title, confirmText, confirmClass) {
    return new Promise(function (resolve) {
        var modalEl = document.getElementById('globalConfirmModal');
        if (!modalEl) {
            resolve(confirm(message || '确定执行此操作？'));
            return;
        }

        var modal = bootstrap.Modal.getOrCreateInstance(modalEl);
        document.getElementById('confirmModalTitle').textContent = title || '确认操作';
        document.getElementById('confirmModalMessage').textContent = message || '确定执行此操作？';
        var btn = document.getElementById('confirmModalBtn');
        btn.textContent = confirmText || '确认';
        btn.className = 'btn btn-sm ' + (confirmClass || 'btn-danger');

        var newBtn = btn.cloneNode(true);
        btn.parentNode.replaceChild(newBtn, btn);

        newBtn.addEventListener('click', function () {
            modal.hide();
            resolve(true);
        });

        var onCancel = function () { resolve(false); };
        modalEl.addEventListener('hidden.bs.modal', onCancel, { once: true });

        modal.show();
    });
}
window.showConfirmModal = showConfirmModal;

// Global click handler for [data-confirm] attribute
function initConfirmDialogs() {
    document.addEventListener('click', function (e) {
        var target = e.target.closest('[data-confirm]');
        if (!target) return;
        var message = target.dataset.confirm;
        if (!message) return;

        e.preventDefault();
        e.stopPropagation();

        showConfirmModal(message).then(function (confirmed) {
            if (!confirmed) return;
            if (target.tagName === 'BUTTON' && target.form) {
                target.form.submit();
                return;
            }
            if (target.tagName === 'A' && target.href) {
                window.location = target.href;
                return;
            }
            if (target.tagName === 'FORM') {
                target.submit();
                return;
            }
            target.removeAttribute('data-confirm');
            target.click();
            target.setAttribute('data-confirm', message);
        });
    }, true);

    // Also intercept form submit with [data-confirm]
    document.addEventListener('submit', function (e) {
        var form = e.target.closest('form[data-confirm]');
        if (!form) return;
        if (form.dataset.confirmHandled === 'true') {
            delete form.dataset.confirmHandled;
            return;
        }
        e.preventDefault();
        showConfirmModal(form.dataset.confirm).then(function (confirmed) {
            if (confirmed) {
                form.dataset.confirmHandled = 'true';
                form.submit();
            }
        });
    }, true);
}

// ============ Clickable table rows ============
function initClickableRows() {
    document.querySelectorAll('tr[data-href]').forEach(function (row) {
        row.style.cursor = 'pointer';
        row.classList.add('clickable-row');
        row.addEventListener('click', function (e) {
            if (e.target.tagName === 'BUTTON' || e.target.tagName === 'A' ||
                e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' ||
                e.target.closest('button') || e.target.closest('a') ||
                e.target.closest('form')) return;
            window.location = this.dataset.href;
        });
    });
}

// ============ Auto-resize textarea ============
function initAutoResize() {
    document.querySelectorAll('textarea.auto-resize').forEach(function (ta) {
        ta.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    });
}

// ============ Bootstrap tooltips ============
function initTooltips() {
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (el) {
        new bootstrap.Tooltip(el);
    });
}

// ============ Password visibility toggle (global) ============
window.togglePassword = function (inputId) {
    var el = document.getElementById(inputId);
    if (el) { el.type = el.type === 'password' ? 'text' : 'password'; }
};

// ============ Date formatter (global) ============
window.formatDate = function (dateStr) {
    if (!dateStr) return '-';
    var d = new Date(dateStr);
    return d.toLocaleString('zh-CN', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit'
    });
};

// ============ API request helper ============
async function apiRequest(url, options) {
    options = options || {};
    var defaultOptions = { headers: { 'Content-Type': 'application/json' } };
    var mergedOptions = Object.assign({}, defaultOptions, options);
    try {
        var response = await fetch(url, mergedOptions);
        var data = await response.json();
        if (!response.ok) throw new Error(data.message || '请求失败');
        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// ============ Centered notification (replaces alert/ElMessage) ============
function showCenteredMessage(message, type, duration) {
    type = type || 'info';
    duration = duration || 3500;

    var container = document.getElementById('centeredMessages');
    if (!container) {
        if (typeof showToast === 'function') { showToast(message, type, duration); }
        else { alert(message); }
        return;
    }

    var icons = {
        success: 'bi-check-circle-fill',
        error: 'bi-x-circle-fill',
        warning: 'bi-exclamation-triangle-fill',
        info: 'bi-info-circle-fill'
    };

    var el = document.createElement('div');
    el.className = 'centered-message msg-' + type;
    el.innerHTML =
        '<i class="bi ' + (icons[type] || icons.info) + ' msg-icon"></i>' +
        '<span class="msg-text">' + escapeHtml(message) + '</span>' +
        '<i class="bi bi-x msg-close"></i>';

    el.querySelector('.msg-close').addEventListener('click', function () {
        removeMsg(el);
    });

    container.appendChild(el);

    var timer = setTimeout(function () { removeMsg(el); }, duration);
    el._msgTimer = timer;

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

window.showCenteredMessage = showCenteredMessage;

// ============ Flash alert → centered notification ============
function initFlashToCentered() {
    var alerts = document.querySelectorAll('#flashMessagesArea .alert[data-flash-message]');
    alerts.forEach(function (alert) {
        var text = alert.dataset.flashMessage || alert.textContent.trim();
        if (!text) { alert.remove(); return; }

        var type = 'info';
        var cat = alert.dataset.flashCategory;
        if (cat === 'success') type = 'success';
        else if (cat === 'danger' || cat === 'error') type = 'error';
        else if (cat === 'warning') type = 'warning';

        showCenteredMessage(text, type, 4500);
        alert.remove();
    });
}

// ============ Toast notification ============
function showToast(message, type, duration) {
    type = type || 'info';
    duration = duration || 3500;

    var container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    var id = 'toast-' + Date.now();
    var colors = {
        success: 'bg-success text-white',
        error: 'bg-danger text-white',
        warning: 'bg-warning',
        info: 'bg-info text-white'
    };
    var icons = {
        success: 'bi-check-circle',
        error: 'bi-x-circle',
        warning: 'bi-exclamation-triangle',
        info: 'bi-info-circle'
    };

    var html =
        '<div id="' + id + '" class="toast align-items-center ' + (colors[type] || colors.info) + ' border-0 animate-fade-in" role="alert">' +
        '<div class="d-flex">' +
        '<div class="toast-body d-flex align-items-center gap-2">' +
        '<i class="bi ' + (icons[type] || icons.info) + '"></i>' + escapeHtml(message) +
        '</div>' +
        '<button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>' +
        '</div></div>';

    container.insertAdjacentHTML('beforeend', html);
    var toastEl = document.getElementById(id);
    var toast = new bootstrap.Toast(toastEl, { delay: duration });
    toast.show();
    toastEl.addEventListener('hidden.bs.toast', function () { toastEl.remove(); });
}
window.showToast = showToast;

// ============ Count-up number animation ============
function initCountUp() {
    document.querySelectorAll('.count-up').forEach(function (el) {
        var target = parseInt(el.dataset.target) || parseInt(el.textContent) || 0;
        var duration = 1000;
        var start = performance.now();

        function update(now) {
            var elapsed = now - start;
            var progress = Math.min(elapsed / duration, 1);
            var eased = 1 - Math.pow(1 - progress, 3);
            var current = Math.round(target * eased);
            el.textContent = current;
            if (progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    });
}

// ============ Copy to clipboard ============
window.copyToClipboard = function (text, buttonEl) {
    navigator.clipboard.writeText(text).then(function () {
        var origHTML = buttonEl?.innerHTML || '';
        if (buttonEl) {
            buttonEl.innerHTML = '<i class="bi bi-check"></i> 已复制';
            setTimeout(function () { buttonEl.innerHTML = origHTML; }, 2000);
        }
        showCenteredMessage('已复制到剪贴板', 'success', 2000);
    }).catch(function () {
        showCenteredMessage('复制失败', 'error', 3000);
    });
};

// ============ Smooth scroll to element ============
window.scrollToElement = function (id) {
    var el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
};

// ============ Toggle select-all checkboxes ============
window.toggleSelectAll = function (checkbox, targetName) {
    document.querySelectorAll('input[name="' + targetName + '"]').forEach(function (cb) {
        cb.checked = checkbox.checked;
    });
};

// ============ Vanilla JS back-to-top button ============
function initBackToTop() {
    // Create button
    var btn = document.createElement('button');
    btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 12.5a.5.5 0 0 1-.5-.5V3.707L4.854 6.354a.5.5 0 1 1-.708-.708l3.5-3.5a.5.5 0 0 1 .708 0l3.5 3.5a.5.5 0 0 1-.708.708L8.5 3.707V12a.5.5 0 0 1-.5.5z"/></svg>';
    btn.id = 'backToTopBtn';
    btn.title = '回到顶部';
    btn.setAttribute('aria-label', '回到顶部');
    Object.assign(btn.style, {
        position: 'fixed', bottom: '24px', right: '24px', zIndex: '1050',
        width: '36px', height: '36px', borderRadius: '6px',
        border: '1px solid #d0d7de', background: '#fff',
        color: '#656d76', cursor: 'pointer',
        display: 'none', alignItems: 'center', justifyContent: 'center',
        boxShadow: '0 2px 8px rgba(31,35,40,0.12)',
        transition: 'opacity 0.2s ease, transform 0.2s ease'
    });

    btn.addEventListener('mouseenter', function () {
        btn.style.transform = 'translateY(-1px)';
        btn.style.boxShadow = '0 4px 12px rgba(31,35,40,0.18)';
    });
    btn.addEventListener('mouseleave', function () {
        btn.style.transform = '';
        btn.style.boxShadow = '0 2px 8px rgba(31,35,40,0.12)';
    });
    btn.addEventListener('click', function () {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    document.body.appendChild(btn);

    // Show/hide based on scroll position
    window.addEventListener('scroll', function () {
        if (window.scrollY > 300) {
            btn.style.display = 'flex';
        } else {
            btn.style.display = 'none';
        }
    }, { passive: true });
}

// ============ Theme Toggle — light/dark ============
(function() {
    var toggle = document.getElementById('themeToggle');
    var icon = document.getElementById('themeIcon');
    if (!toggle || !icon) return;

    function setTheme(isDark) {
        if (isDark) {
            document.documentElement.setAttribute('data-theme', 'dark');
            icon.className = 'bi bi-sun';
            toggle.title = '切换日间模式';
            localStorage.setItem('theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
            icon.className = 'bi bi-moon-stars';
            toggle.title = '切换夜间模式';
            localStorage.setItem('theme', 'light');
        }
    }

    // Initial icon state
    if (document.documentElement.hasAttribute('data-theme')) {
        icon.className = 'bi bi-sun';
        toggle.title = '切换日间模式';
    }

    toggle.addEventListener('click', function() {
        var isDark = document.documentElement.hasAttribute('data-theme');
        setTheme(!isDark);
    });
})();
