(function () {
  // Mobile nav toggle
  var toggler = document.querySelector('[data-nav-toggle]');
  var collapse = document.getElementById('navbar-menu');
  if (toggler && collapse) {
    toggler.addEventListener('click', function () {
      var open = collapse.classList.toggle('show');
      toggler.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }

  // Dropdown menus
  document.querySelectorAll('[data-nav-dropdown]').forEach(function (trigger) {
    trigger.addEventListener('click', function (e) {
      e.preventDefault();
      var menu = trigger.parentElement.querySelector('.dropdown-menu');
      if (!menu) return;
      var isOpen = menu.classList.toggle('show');
      trigger.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      document.querySelectorAll('.dropdown-menu.show').forEach(function (other) {
        if (other !== menu) other.classList.remove('show');
      });
    });
  });

  document.addEventListener('click', function (e) {
    if (!e.target.closest('.dropdown')) {
      document.querySelectorAll('.dropdown-menu.show').forEach(function (m) {
        m.classList.remove('show');
      });
    }
  });

  // Dark / light theme toggle
  var themeBtn = document.querySelector('[data-theme-toggle]');
  if (themeBtn) {
    var saved = localStorage.getItem('theme');
    if (saved) document.documentElement.setAttribute('data-theme', saved);

    themeBtn.addEventListener('click', function () {
      var current = document.documentElement.getAttribute('data-theme');
      var isDark = current === 'dark' ||
        (!current && window.matchMedia('(prefers-color-scheme: dark)').matches);
      var next = isDark ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
    });
  }

  // Global '/' shortcut to focus or open search
  document.addEventListener('keydown', function (e) {
    if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return;
    var tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'textarea' || (e.target && e.target.isContentEditable)) return;
    e.preventDefault();
    var searchInput = document.getElementById('search-input') || document.querySelector('[data-pub-search]');
    if (searchInput) {
      searchInput.focus();
      searchInput.select();
    } else {
      var searchLink = document.querySelector('.nav-link--search');
      if (searchLink && searchLink.href) window.location.href = searchLink.href;
    }
  });

  // Scroll reveal animations
  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

    document.querySelectorAll('.animate-on-scroll').forEach(function (el) {
      observer.observe(el);
    });
  } else {
    document.querySelectorAll('.animate-on-scroll').forEach(function (el) {
      el.classList.add('is-visible');
    });
  }

  // Stop card media shimmer once image loads
  document.querySelectorAll('.card__media img').forEach(function (img) {
    var media = img.closest('.card__media');
    if (!media) return;
    var markLoaded = function () { media.classList.add('is-loaded'); };
    if (img.complete) {
      markLoaded();
    } else {
      img.addEventListener('load', markLoaded, { once: true });
      img.addEventListener('error', markLoaded, { once: true });
    }
  });

  // Email copy buttons
  document.querySelectorAll('[data-email-copy]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var email = btn.getAttribute('data-email');
      if (!email) return;
      var done = function () {
        btn.setAttribute('data-copied', 'true');
        var prevLabel = btn.getAttribute('aria-label');
        btn.setAttribute('aria-label', 'Email copied');
        setTimeout(function () {
          btn.removeAttribute('data-copied');
          btn.setAttribute('aria-label', prevLabel);
        }, 1500);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(email).then(done, function () {
          window.prompt('Copy this email:', email);
        });
      } else {
        window.prompt('Copy this email:', email);
      }
    });
  });

  // BibTeX cite buttons
  document.querySelectorAll('.cite-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var bib = btn.getAttribute('data-bibtex');
      if (!bib) return;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(bib).then(function () {
          btn.textContent = 'copied!';
          setTimeout(function () { btn.textContent = 'cite'; }, 1500);
        });
      } else {
        window.prompt('Copy BibTeX:', bib);
      }
    });
  });

  // Back to top button
  var topBtn = document.querySelector('[data-back-to-top]');
  if (topBtn) {
    window.addEventListener('scroll', function () {
      topBtn.classList.toggle('is-visible', window.scrollY > 600);
    }, { passive: true });
    topBtn.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  // Collapsible sections
  document.querySelectorAll('[data-collapse-toggle]').forEach(function (heading) {
    var target = heading.nextElementSibling;
    if (!target) return;
    heading.addEventListener('click', function () {
      heading.classList.toggle('is-collapsed');
      target.classList.toggle('is-collapsed');
    });
  });

  // Blog / Notes filter tabs
  var blogTabs = document.querySelectorAll('[data-blog-filter]');
  if (blogTabs.length > 0) {
    var blogEntries = document.querySelectorAll('[data-post-type]');
    blogTabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        var filter = tab.getAttribute('data-blog-filter');
        blogTabs.forEach(function (t) {
          var active = t === tab;
          t.classList.toggle('is-active', active);
          t.setAttribute('aria-selected', active ? 'true' : 'false');
        });
        blogEntries.forEach(function (entry) {
          var type = entry.getAttribute('data-post-type');
          entry.style.display = (filter === 'all' || type === filter) ? '' : 'none';
        });
      });
    });
  }

  // Publications filter tabs + live search
  var pubTabs = document.querySelectorAll('[data-pub-filter]');
  var pubSearch = document.querySelector('[data-pub-search]');
  if (pubTabs.length > 0 || pubSearch) {
    var pubItems = document.querySelectorAll('[data-pub-item]');
    var activeFilter = 'all';

    var applyPubFilter = function () {
      var q = pubSearch ? pubSearch.value.trim().toLowerCase() : '';
      pubItems.forEach(function (item) {
        var year = parseInt(item.getAttribute('data-pub-year'), 10) || 0;
        var featured = item.getAttribute('data-pub-featured') === 'true';
        var matchesTab = true;
        if (activeFilter === 'featured') matchesTab = featured;
        else if (activeFilter === 'older') matchesTab = year <= 2022;
        else if (activeFilter !== 'all') matchesTab = String(year) === activeFilter;

        var matchesQuery = !q || item.textContent.toLowerCase().indexOf(q) !== -1;
        item.style.display = (matchesTab && matchesQuery) ? '' : 'none';
      });
    };

    pubTabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        activeFilter = tab.getAttribute('data-pub-filter');
        pubTabs.forEach(function (t) {
          var active = t === tab;
          t.classList.toggle('is-active', active);
          t.setAttribute('aria-selected', active ? 'true' : 'false');
        });
        applyPubFilter();
      });
    });

    if (pubSearch) {
      pubSearch.addEventListener('input', applyPubFilter);
    }
  }
})();
