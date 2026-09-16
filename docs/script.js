/* ============================================================
   Broadcast Media Analysis — Interactive Page Script
   Handles tab switching, copy buttons, KaTeX rendering, Mermaid,
   and interactive IoU simulator.
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize KaTeX auto-render if loaded
  if (typeof renderMathInElement === 'function') {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false }
      ],
      throwOnError: false
    });
  }

  // 2. Initialize Mermaid.js
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'dark',
      themeVariables: {
        darkMode: true,
        background: '#0b0f19',
        primaryColor: '#6366f1',
        primaryTextColor: '#f8fafc',
        primaryBorderColor: '#818cf8',
        lineColor: '#38bdf8',
        secondaryColor: '#1e293b',
        tertiaryColor: '#0f172a'
      }
    });
  }

  // 3. Tab Switching for Code Snippets
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');

      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add('active');
      }
    });
  });

  // 4. Code Copy-to-Clipboard
  const copyBtns = document.querySelectorAll('.copy-btn');
  copyBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const activePane = document.querySelector('.tab-pane.active pre code');
      if (activePane) {
        navigator.clipboard.writeText(activePane.innerText).then(() => {
          const originalText = btn.innerHTML;
          btn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
            Copied!
          `;
          btn.style.color = '#34d399';
          setTimeout(() => {
            btn.innerHTML = originalText;
            btn.style.color = '';
          }, 2000);
        });
      }
    });
  });

  // 5. Mobile Navigation Toggle
  const mobileToggle = document.querySelector('.mobile-toggle');
  const navLinks = document.querySelector('.nav-links');

  if (mobileToggle && navLinks) {
    mobileToggle.addEventListener('click', () => {
      navLinks.classList.toggle('open');
    });

    document.querySelectorAll('.nav-link').forEach(link => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('open');
      });
    });
  }

  // 6. Active Section Highlighting on Scroll
  const sections = document.querySelectorAll('section[id]');
  window.addEventListener('scroll', () => {
    const scrollY = window.pageYOffset;
    sections.forEach(current => {
      const sectionHeight = current.offsetHeight;
      const sectionTop = current.offsetTop - 120;
      const sectionId = current.getAttribute('id');
      const navItem = document.querySelector(`.nav-links a[href*='${sectionId}']`);

      if (navItem) {
        if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
          navItem.classList.add('active');
        } else {
          navItem.classList.remove('active');
        }
      }
    });
  });

  // 7. Interactive IoU Simulator
  const iouSlider = document.getElementById('iou-slider');
  const iouDisplay = document.getElementById('iou-val');
  const iouFeedback = document.getElementById('iou-status');

  if (iouSlider && iouDisplay && iouFeedback) {
    const updateIoU = () => {
      const val = parseFloat(iouSlider.value) / 100;
      iouDisplay.textContent = val.toFixed(2);

      if (val >= 0.75) {
        iouFeedback.textContent = 'Excellent Match (Model A Baseline average is ~0.75)';
        iouFeedback.style.color = '#34d399';
      } else if (val >= 0.5) {
        iouFeedback.textContent = 'Valid Detection (PASCAL VOC / COCO Standard Threshold)';
        iouFeedback.style.color = '#38bdf8';
      } else {
        iouFeedback.textContent = 'Poor Overlap / False Positive Threshold (< 0.50)';
        iouFeedback.style.color = '#f87171';
      }
    };

    iouSlider.addEventListener('input', updateIoU);
    updateIoU();
  }
});
