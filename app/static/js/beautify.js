/**
 * ============================================
 * AI Platform — Frontend Enhancements v5.0
 * GitHub-inspired: scroll reveal · smooth scroll · lazy images
 * ============================================
 */

document.addEventListener('DOMContentLoaded', function () {
    initScrollReveal();
    initSmoothScroll();
    initNavbarEffect();
});

/* ==================================================================
   1. Scroll-triggered reveal (IntersectionObserver)
   ================================================================== */
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

    // Auto-add reveal to cards not already marked
    document.querySelectorAll('.card:not(.reveal):not(.no-reveal)').forEach(function (card, i) {
        card.classList.add('reveal', 'reveal-up');
        if (i < 6) card.classList.add('reveal-delay-' + (i + 1));
        observer.observe(card);
    });
}

/* ==================================================================
   2. Smooth scroll for anchor links
   ================================================================== */
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

/* ==================================================================
   3. Navbar subtle shadow on scroll
   ================================================================== */
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

/* ==================================================================
   4. Visibility change — respect user's time away
   ================================================================== */
(function () {
    document.addEventListener('visibilitychange', function () {
        // Pause/resume any active animations when tab hidden
        var animations = document.querySelectorAll('.float-anim, .float-anim-delayed');
        if (document.hidden) {
            animations.forEach(function (el) { el.style.animationPlayState = 'paused'; });
        } else {
            animations.forEach(function (el) { el.style.animationPlayState = 'running'; });
        }
    });
})();

/* ==================================================================
   5. Lazy image loading
   ================================================================== */
function initLazyImages() {
    if (!('IntersectionObserver' in window)) return;
    var imgObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                var img = entry.target;
                if (img.dataset.src) {
                    img.src = img.dataset.src;
                    img.addEventListener('load', function () { img.classList.add('loaded'); });
                }
                imgObserver.unobserve(img);
            }
        });
    });
    document.querySelectorAll('img[data-src], img.lazy-load').forEach(function (img) {
        imgObserver.observe(img);
    });
}
// Initialize lazy loading when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initLazyImages);
} else {
    initLazyImages();
}
