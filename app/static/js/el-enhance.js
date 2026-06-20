/**
 * ============================================
 * AI Platform - Vue 3 + Element Plus 增强层 v2.0
 * ============================================
 *
 * 功能:
 *   1. el-backtop         回到顶部 (渐变色圆钮)
 *   2. ElMessage          消息通知 (自动接管 Bootstrap alert + showToast)
 *   3. Badge -> ElTag     状态标签样式升级
 *   4. Progress -> ElProgress  进度条样式升级
 *   5. Empty -> ElEmpty   空状态占位
 *   6. 图表响应式缩小      智能调整 canvas 高度
 * ============================================
 */

document.addEventListener('DOMContentLoaded', function () {
    if (typeof Vue === 'undefined' || typeof ElementPlus === 'undefined') {
        console.warn('Vue 3 / ElementPlus not loaded, skipping enhancements');
        return;
    }
    initElEnhanceApp();
    initDOMEnhancements();
    initChartResize();
});

/* ==================================================================
   Part 1: Vue App — Backtop + 消息系统
   ================================================================== */
function initElEnhanceApp() {
    const { createApp } = Vue;

    const app = createApp({
        template: `
            <div class="el-enhance-root">
                <el-backtop :visibility-height="300" :right="36" :bottom="56">
                    <div class="el-backtop-custom">
                        <el-icon><Top /></el-icon>
                    </div>
                </el-backtop>
            </div>
        `,
    });

    app.use(ElementPlus);
    if (window.ElementPlusIconsVue) {
        for (const [name, comp] of Object.entries(window.ElementPlusIconsVue)) {
            app.component(name, comp);
        }
    }

    const container = document.createElement('div');
    container.id = 'el-enhance-container';
    document.body.appendChild(container);
    app.mount('#el-enhance-container');
}

/* ==================================================================
   Part 2: DOM 自动增强 — Bootstrap → Element Plus 视觉升级
   ================================================================== */
function initDOMEnhancements() {
    upgradeBadgesToElTags();
    upgradeProgressBars();
    upgradeEmptyStates();
    upgradeAlertsToElMessage();

    // 暴露全局消息 API (使用居中通知)
    window.$message = {
        success: (m) => showCenteredMessage(m, 'success', 3000),
        error:   (m) => showCenteredMessage(m, 'error',   5000),
        warning: (m) => showCenteredMessage(m, 'warning', 4000),
        info:    (m) => showCenteredMessage(m, 'info',    3000),
    };

    // 接管 showToast → 居中通知
    const orig = window.showToast;
    window.showToast = function (msg, type, dur) {
        if (window.showCenteredMessage && type) {
            showCenteredMessage(msg, type, dur || 3500);
        } else if (window.showCenteredMessage) {
            showCenteredMessage(msg, 'info', dur || 3500);
        } else if (orig) {
            orig(msg, type, dur);
        }
    };
}

/* ---------- Badge -> ElTag 视觉升级 ---------- */
function upgradeBadgesToElTags() {
    // 只升级 Bootstrap badge，不替换 DOM (避免破坏事件绑定)
    document.querySelectorAll('.badge').forEach(badge => {
        badge.classList.add('el-tag-style');

        // 颜色映射 Bootstrap → Element Plus
        const cls = badge.className;
        if (cls.includes('bg-success') || cls.includes('badge-success'))   badge.dataset.elType = 'success';
        else if (cls.includes('bg-danger') || cls.includes('badge-danger'))  badge.dataset.elType = 'danger';
        else if (cls.includes('bg-warning') || cls.includes('badge-warning')) badge.dataset.elType = 'warning';
        else if (cls.includes('bg-info') || cls.includes('badge-info'))    badge.dataset.elType = 'info';
        else if (cls.includes('bg-primary'))  badge.dataset.elType = 'primary';
        else badge.dataset.elType = 'info';
    });
}

/* ---------- Progress Bar -> ElProgress 视觉升级 ---------- */
function upgradeProgressBars() {
    document.querySelectorAll('.progress').forEach(barContainer => {
        barContainer.classList.add('el-progress-style');
        const bar = barContainer.querySelector('.progress-bar');
        if (bar) {
            const width = parseFloat(bar.style.width) || 0;
            barContainer.style.setProperty('--progress-pct', width + '%');

            // 颜色映射
            if (bar.classList.contains('bg-success')) barContainer.dataset.elStatus = 'success';
            else if (bar.classList.contains('bg-danger')) barContainer.dataset.elStatus = 'exception';
            else if (bar.classList.contains('bg-warning')) barContainer.dataset.elStatus = 'warning';
        }
    });
}

/* ---------- Empty State -> ElEmpty ---------- */
function upgradeEmptyStates() {
    document.querySelectorAll('.empty-state, [data-empty]').forEach(el => {
        const text = el.dataset.empty || el.textContent.trim() || '暂无数据';
        const desc = el.dataset.emptyDesc || '';

        // 创建 el-empty 风格占位
        el.innerHTML = `
            <div class="el-empty-wrapper">
                <div class="el-empty-image">
                    <svg viewBox="0 0 64 41" xmlns="http://www.w3.org/2000/svg">
                        <g transform="translate(0 1)" fill="none" fill-rule="evenodd">
                            <ellipse fill="#f5f5f5" cx="32" cy="33" rx="32" ry="7"/>
                            <g fill-rule="nonzero" stroke="#d9d9d9">
                                <path d="M55 12.76L44.854 1.258C44.367.474 43.656 0 42.907 0H21.093c-.749 0-1.46.474-1.947 1.257L9 12.761V22h46v-9.24z"/>
                                <path d="M41.613 15.931c0-1.605.994-2.93 2.227-2.931H55v18.137C55 33.26 53.68 35 52.05 35h-40.1C10.32 35 9 33.259 9 31.137V13h11.16c1.233 0 2.227 1.323 2.227 2.928v.022c0 1.605 1.005 2.901 2.237 2.901h14.752c1.232 0 2.237-1.308 2.237-2.913v-.007z" fill="#fafafa"/>
                            </g>
                        </g>
                    </svg>
                </div>
                <p class="el-empty-text">${text}</p>
                ${desc ? `<p class="el-empty-desc">${desc}</p>` : ''}
            </div>
        `;
        el.classList.add('el-empty-state');
    });
}

/* ---------- Flash Alert -> 居中通知 ---------- */
function upgradeAlertsToElMessage() {
    const alerts = document.querySelectorAll('#flashMessagesArea .alert[data-flash-message]');
    alerts.forEach(alert => {
        const text = alert.dataset.flashMessage || alert.textContent.trim();
        if (!text) { alert.remove(); return; }

        let type = 'info';
        const cat = alert.dataset.flashCategory;
        if (cat === 'success') type = 'success';
        else if (cat === 'danger' || cat === 'error') type = 'error';
        else if (cat === 'warning') type = 'warning';

        // 使用居中通知
        showCenteredMessage(text, type, 4500);
        alert.remove();
    });
}

/* ==================================================================
   Part 3: 图表响应式缩小
   ================================================================== */
function initChartResize() {
    // 限制所有 canvas 最大高度
    document.querySelectorAll('canvas').forEach(canvas => {
        const h = parseInt(canvas.getAttribute('height')) || 0;
        if (h > 240) {
            canvas.setAttribute('height', Math.min(h, 220));
            canvas.style.maxHeight = '320px';
        }
    });

    // 图表容器添加最大高度约束
    document.querySelectorAll('.card-body').forEach(body => {
        const canvas = body.querySelector('canvas');
        if (canvas) {
            body.style.overflow = 'hidden';
        }
    });

    // 响应式: 小屏幕进一步缩小
    const mq = window.matchMedia('(max-width: 768px)');
    function handleResize(e) {
        document.querySelectorAll('canvas').forEach(c => {
            c.style.maxHeight = e.matches ? '220px' : '320px';
        });
    }
    mq.addEventListener('change', handleResize);
    handleResize(mq);
}

/* ==================================================================
   Part 4: 动态样式注入
   ================================================================== */
(function injectStyles() {
    const css = `
        /* -------- Backtop -------- */
        .el-backtop-custom {
            width: 42px; height: 42px;
            background: linear-gradient(135deg, #4e73df, #36b9cc);
            color: #fff; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 20px;
            box-shadow: 0 4px 16px rgba(78,115,223,0.4);
            transition: all .3s ease; cursor: pointer;
        }
        .el-backtop-custom:hover {
            background: linear-gradient(135deg, #2e59d9, #1fa9ba);
            box-shadow: 0 6px 24px rgba(78,115,223,0.6);
            transform: translateY(-2px) scale(1.06);
        }
        .el-backtop { z-index: 1050 !important; }

        /* -------- ElTag style (Bootstrap badge 升级) -------- */
        .badge.el-tag-style {
            border-radius: 4px !important;
            padding: 0.25em 0.6em !important;
            font-weight: 500 !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.3px !important;
            transition: all .15s ease !important;
            border: 1px solid transparent !important;
        }
        .badge.el-tag-style[data-el-type="success"] { background: #e8f5e9 !important; color: #2e7d32 !important; border-color: #c8e6c9 !important; }
        .badge.el-tag-style[data-el-type="danger"]  { background: #ffebee !important; color: #c62828 !important; border-color: #ffcdd2 !important; }
        .badge.el-tag-style[data-el-type="warning"] { background: #fff3e0 !important; color: #e65100 !important; border-color: #ffe0b2 !important; }
        .badge.el-tag-style[data-el-type="info"]    { background: #e3f2fd !important; color: #1565c0 !important; border-color: #bbdefb !important; }
        .badge.el-tag-style[data-el-type="primary"] { background: #ede7f6 !important; color: #4527a0 !important; border-color: #d1c4e9 !important; }

        /* -------- ElProgress style (Bootstrap progress 升级) -------- */
        .progress.el-progress-style {
            height: 8px !important;
            border-radius: 100px !important;
            background: #e9ecef !important;
            position: relative;
            overflow: visible !important;
        }
        .progress.el-progress-style .progress-bar {
            border-radius: 100px !important;
            transition: width .6s ease !important;
            position: relative;
        }
        .progress.el-progress-style[data-el-status="success"] .progress-bar {
            background: linear-gradient(90deg, #66bb6a, #43a047) !important;
        }
        .progress.el-progress-style[data-el-status="exception"] .progress-bar {
            background: linear-gradient(90deg, #ef5350, #e53935) !important;
        }
        .progress.el-progress-style[data-el-status="warning"] .progress-bar {
            background: linear-gradient(90deg, #ffa726, #fb8c00) !important;
        }
        .progress.el-progress-style::after {
            content: attr(data-pct, var(--progress-pct));
            content: var(--progress-pct);
            position: absolute; right: -2px; top: -18px;
            font-size: 10px; color: #666;
        }

        /* -------- ElEmpty style -------- */
        .el-empty-state { text-align: center; padding: 2rem 1.5rem; }
        .el-empty-wrapper { max-width: 260px; margin: 0 auto; }
        .el-empty-image { margin-bottom: 1rem; }
        .el-empty-image svg { width: 80px; height: auto; }
        .el-empty-text { color: #909399; font-size: 0.9rem; margin: 0; }
        .el-empty-desc { color: #c0c4cc; font-size: 0.8rem; margin: 0.25rem 0 0; }

        /* -------- ElAlert style (页面级提示) -------- */
        .el-alert-style {
            border-radius: 8px; padding: 0.6rem 1rem;
            display: flex; align-items: center; gap: 0.5rem;
            font-size: 0.9rem; margin-bottom: 0.5rem;
            border: 1px solid transparent;
        }
        .el-alert-style.el-alert-success { background: #f0f9eb; border-color: #c2e7b0; color: #67c23a; }
        .el-alert-style.el-alert-error   { background: #fef0f0; border-color: #fbc4c4; color: #f56c6c; }
        .el-alert-style.el-alert-warning { background: #fdf6ec; border-color: #f5dab1; color: #e6a23c; }
        .el-alert-style.el-alert-info    { background: #f4f4f5; border-color: #c8c9cc; color: #909399; }

        /* -------- 图表容器响应式 -------- */
        .chart-container {
            position: relative;
            width: 100%;
            max-height: 340px;
            overflow: hidden;
        }
        .chart-container canvas {
            max-height: 320px !important;
            width: 100% !important;
        }
        @media (max-width: 768px) {
            .chart-container { max-height: 240px; }
            .chart-container canvas { max-height: 220px !important; }
        }

        /* -------- 卡片紧凑模式 -------- */
        .card.compact .card-body { padding: 0.75rem 1rem; }
        .card.compact .card-header { padding: 0.5rem 1rem; }

        /* -------- 通用美化 -------- */
        .table > thead { background: #fafafa; }
        .btn { font-weight: 500; letter-spacing: 0.2px; }
        /* 表单聚焦效果统一由 style.css (:root --brand-400) 管理, 此处不再覆写避免冲突 */
    `;

    const style = document.createElement('style');
    style.textContent = css;
    document.head.appendChild(style);
})();
