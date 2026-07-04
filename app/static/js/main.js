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
    initScrollReveal();
    initSmoothScroll();
    initNavbarEffect();
    initCardHover();
    initAutoThemeBadges();
    initCountUp();
    initFlashToCentered();
    initBackToTop();
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

// ============ Scroll-triggered reveal (IntersectionObserver) ============
function initScrollReveal() {
    if (!('IntersectionObserver' in window)) {
        document.querySelectorAll('.reveal').forEach(function (el) { el.classList.add('revealed'); });
        return;
    }
    var observer = new IntersectionObserver(
        function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');
                    observer.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.1, rootMargin: '0px 0px -30px 0px' }
    );
    document.querySelectorAll('.reveal').forEach(function (el) { observer.observe(el); });
    document.querySelectorAll('.card:not(.reveal):not(.no-reveal)').forEach(function (card, i) {
        card.classList.add('reveal', 'reveal-up');
        if (i < 6) card.classList.add('reveal-delay-' + (i + 1));
        observer.observe(card);
    });
}

// ============ Smooth scroll for anchor links ============
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
        anchor.addEventListener('click', function (e) {
            var target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// ============ Navbar subtle shadow on scroll ============
function initNavbarEffect() {
    var navbar = document.querySelector('.navbar');
    if (!navbar) return;
    window.addEventListener('scroll', function () {
        if (window.scrollY > 10) {
            navbar.style.boxShadow = '0 1px 0 rgba(255,255,255,0.08), 0 2px 8px rgba(0,0,0,0.2)';
        } else {
            navbar.style.boxShadow = '0 1px 0 rgba(255,255,255,0.08), 0 1px 4px rgba(0,0,0,0.15)';
        }
    }, { passive: true });
}

// ============ Visibility change — respect user's time away ============
(function () {
    document.addEventListener('visibilitychange', function () {
        var animations = document.querySelectorAll('.float-anim, .float-anim-delayed');
        if (document.hidden) {
            animations.forEach(function (el) { el.style.animationPlayState = 'paused'; });
        } else {
            animations.forEach(function (el) { el.style.animationPlayState = 'running'; });
        }
    });
})();

// ============ Card hover — add hoverable class ============
function initCardHover() {
    document.querySelectorAll('.card:not(.no-hover):not(.card-hoverable)').forEach(function (card) {
        if (card.querySelector('a, button, .clickable-row, [data-href]')) {
            card.classList.add('card-hoverable');
        }
    });
}

// ============ Auto-style status badges with soft colors ============
function initAutoThemeBadges() {
    document.querySelectorAll('.badge').forEach(function (badge) {
        var text = badge.textContent.trim().toLowerCase();
        var map = {
            'trained': 'badge-soft-success',
            'deployed': 'badge-soft-success',
            'completed': 'badge-soft-success',
            'running': 'badge-soft-primary',
            'pending': 'badge-soft-info',
            'failed': 'badge-soft-danger',
            'cancelled': 'badge-soft-warning',
            'draft': 'badge-soft-warning',
            'success': 'badge-soft-success',
            'error': 'badge-soft-danger',
            'warning': 'badge-soft-warning',
            'info': 'badge-soft-info',
            'positive': 'badge-soft-success',
            'negative': 'badge-soft-danger',
            'neutral': 'badge-soft-warning',
        };
        for (var key in map) {
            if (text.indexOf(key) !== -1) {
                badge.classList.add('badge-soft', map[key]);
                break;
            }
        }
    });
}

// ============ Count-up number animation (IntersectionObserver) ============
function initCountUp() {
    if (!('IntersectionObserver' in window)) return;
    var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                var el = entry.target;
                var target = parseFloat(el.dataset.target) || 0;
                var decimals = parseInt(el.dataset.decimals) || 0;
                var duration = parseInt(el.dataset.duration) || 800;
                var startTime = null;
                function step(timestamp) {
                    if (!startTime) startTime = timestamp;
                    var progress = Math.min((timestamp - startTime) / duration, 1);
                    var eased = 1 - Math.pow(1 - progress, 3);
                    var current = eased * target;
                    el.textContent = decimals > 0 ? current.toFixed(decimals) : Math.round(current);
                    if (progress < 1) requestAnimationFrame(step);
                }
                requestAnimationFrame(step);
                observer.unobserve(el);
            }
        });
    }, { threshold: 0.3 });
    document.querySelectorAll('.count-up').forEach(function (el) { observer.observe(el); });
}

// ============ Confetti celebration (canvas, no dependencies) ============
window.launchConfetti = function (opts) {
    opts = opts || {};
    var count = opts.count || 80;
    var spread = opts.spread || 80;
    var originX = opts.originX !== undefined ? opts.originX : 0.5;
    var originY = opts.originY !== undefined ? opts.originY : 0.5;
    var colors = opts.colors || ['#1f883d', '#0969da', '#9a6700', '#cf222e', '#8250df', '#bf3989'];
    var canvas = document.getElementById('confetti-canvas');
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'confetti-canvas';
        document.body.appendChild(canvas);
    }
    var ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    var particles = [];
    for (var i = 0; i < count; i++) {
        var angle = (Math.PI / 180) * (Math.random() * spread - spread / 2);
        var velocity = 4 + Math.random() * 4;
        particles.push({
            x: canvas.width * originX,
            y: canvas.height * originY,
            vx: Math.cos(angle) * velocity * (0.6 + Math.random() * 0.4),
            vy: Math.sin(angle) * velocity * (0.6 + Math.random() * 0.4) - 4,
            size: 5 + Math.random() * 5,
            color: colors[Math.floor(Math.random() * colors.length)],
            rotation: Math.random() * 360,
            rotSpeed: (Math.random() - 0.5) * 10,
            opacity: 1,
            shape: Math.random() > 0.5 ? 'circle' : 'rect',
        });
    }
    var gravity = 0.12;
    var drag = 0.98;
    var startTime = Date.now();
    var duration = opts.duration || 2500;
    function animate() {
        var elapsed = Date.now() - startTime;
        if (elapsed > duration) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (var i = particles.length - 1; i >= 0; i--) {
            var p = particles[i];
            p.vx *= drag; p.vy += gravity;
            p.x += p.vx; p.y += p.vy;
            p.rotation += p.rotSpeed;
            p.opacity = Math.max(0, 1 - elapsed / duration);
            if (p.y > canvas.height + 20 || p.opacity <= 0) { particles.splice(i, 1); continue; }
            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate((p.rotation * Math.PI) / 180);
            ctx.globalAlpha = p.opacity;
            ctx.fillStyle = p.color;
            if (p.shape === 'circle') {
                ctx.beginPath(); ctx.arc(0, 0, p.size / 2, 0, Math.PI * 2); ctx.fill();
            } else {
                ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
            }
            ctx.restore();
        }
        requestAnimationFrame(animate);
    }
    animate();
};

// ============ Trigger success celebration (checkmark + confetti) ============
window.celebrateSuccess = function (selector, opts) {
    window.launchConfetti(opts);
    var target = selector ? document.querySelector(selector) : null;
    if (target) target.classList.add('celebration-pulse', 'sparkle-burst');
    var checkEl = document.createElement('div');
    checkEl.className = 'success-checkmark';
    checkEl.style.cssText = 'position:fixed;top:50%;left:50%;margin:-32px 0 0 -32px;z-index:99998;';
    checkEl.innerHTML = '<svg viewBox="0 0 52 52" class="check-circle"><circle class="check-circle" cx="26" cy="26" r="23"/><path class="check-path" d="M14 27l7 7 16-16"/></svg>';
    document.body.appendChild(checkEl);
    checkEl.classList.add('bounce-in');
    setTimeout(function () { if (checkEl.parentNode) checkEl.parentNode.removeChild(checkEl); }, 2000);
};

// ============ Trigger training complete banner ============
window.showTrainingComplete = function (modelName, accuracy) {
    window.celebrateSuccess(null, { count: 120, spread: 120 });
    var banner = document.createElement('div');
    banner.style.cssText = 'position:fixed;top:76px;left:50%;transform:translateX(-50%);z-index:99990;min-width:320px;max-width:500px;padding:16px 24px;border-radius:8px;background:#1f883d;color:#fff;text-align:center;animation:bounce-in 0.4s cubic-bezier(0.34,1.56,0.64,1) both;';
    banner.innerHTML = '<div style="font-size:2rem;margin-bottom:4px;">&#10003;</div><div style="font-size:1rem;font-weight:600;margin-bottom:2px;">训练完成!</div><div style="font-size:0.85rem;opacity:0.9;">' + escapeHtml(modelName || '') + (accuracy ? ' &middot; 准确率 ' + accuracy : '') + '</div>';
    document.body.appendChild(banner);
    setTimeout(function () {
        banner.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        banner.style.opacity = '0';
        banner.style.transform = 'translateX(-50%) translateY(-10px)';
        setTimeout(function () { if (banner.parentNode) banner.parentNode.removeChild(banner); }, 310);
    }, 3000);
};

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
