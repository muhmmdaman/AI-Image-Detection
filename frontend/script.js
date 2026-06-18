const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const animate = (fn) => requestAnimationFrame(fn);

const fileInput = $('#file-input');
const uploadBox = $('#upload-box');
const uploadLabel = $('#upload-label');
const preview = $('#preview');
const predictBtn = $('#predict-btn');
const resetBtn = $('#reset-btn');
const resultCard = $('#result');
const resultLabel = $('#result-label');
const resultConfidence = $('#result-confidence');
const meterFill = $('#meter-fill');
const meterState = $('#meter-state');
const logs = $('#logs');
const previewZoom = $('#preview-zoom');
const previewZoomImg = $('#preview-zoom-img');
const zoomClose = $('#zoom-close');
const heatmapImg = $('#ai-heatmap');
const confCanvas = $('#confidenceChart');
const accuracyCanvas = $('#accuracyChart');
const lossCanvas = $('#lossChart');
const themeToggle = $('#theme-toggle');
const universe = $('#universe');

function log(msg) {
  if (!logs) return;
  const el = document.createElement('div');
  el.className = 'log';
  el.textContent = msg;
  el.style.opacity = '0';
  el.style.transform = 'translateY(-6px)';
  logs.prepend(el);
  requestAnimationFrame(() => {
    el.style.transition = 'opacity .42s ease, transform .42s cubic-bezier(.2,.9,.2,1)';
    el.style.opacity = '1';
    el.style.transform = 'translateY(0)';
  });
  const all = logs.querySelectorAll('.log');
  if (all.length > 30) all[all.length - 1].remove();
}

if (!fileInput || !uploadBox) {
  console.warn('Essential DOM elements not found — script aborted.');
}

uploadLabel.addEventListener('click', () => {
  if (!fileInput.disabled) fileInput.click();
});

fileInput.addEventListener('change', (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) handleFile(f);
});

uploadBox.addEventListener('dragenter', (e) => {
  e.preventDefault();
  uploadBox.classList.add('dragging');
});
uploadBox.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadBox.classList.add('dragging');
});
['dragleave', 'dragend', 'drop'].forEach(ev =>
  uploadBox.addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === 'drop' && e.dataTransfer && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
    uploadBox.classList.remove('dragging');
  })
);

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.style.display = 'block';
    preview.style.opacity = '0';
    requestAnimationFrame(() => {
      preview.style.transition = 'opacity .55s cubic-bezier(.2,.9,.2,1), transform .6s ease';
      preview.style.opacity = '1';
      preview.style.transform = 'scale(1)';
    });
    predictBtn.disabled = false;
    resetBtn.style.display = 'inline-block';
    uploadLabel.style.display = 'none';
    log(`Loaded: ${file.name}`);
    flashScanOverlay(700);
  };
  reader.onerror = () => {
    alert('Failed to read file.');
    log('File read error.');
  };
  reader.readAsDataURL(file);
}

function flashScanOverlay(ms = 600) {
  const scan = $('#scan-overlay');
  if (!scan) return;
  scan.style.display = 'flex';
  scan.style.opacity = '0';
  requestAnimationFrame(() => {
    scan.style.transition = 'opacity .18s ease';
    scan.style.opacity = '1';
  });
  setTimeout(() => {
    scan.style.opacity = '0';
    setTimeout(() => (scan.style.display = 'none'), 420);
  }, ms);
}

preview.addEventListener('click', () => {
  if (!preview.src) return;
  previewZoomImg.src = preview.src;
  previewZoom.style.display = 'flex';
  requestAnimationFrame(() => {
    previewZoom.style.opacity = '1';
    previewZoom.style.transform = 'scale(1)';
  });
  document.body.style.overflow = 'hidden';
});
zoomClose.addEventListener('click', closePreviewZoom);
previewZoom.addEventListener('click', (e) => {
  if (e.target === previewZoom) closePreviewZoom();
});

function closePreviewZoom() {
  previewZoom.style.opacity = '0';
  previewZoom.style.transform = 'scale(.98)';
  setTimeout(() => {
    previewZoom.style.display = 'none';
    document.body.style.overflow = '';
  }, 220);
}

previewZoom.addEventListener('mousemove', (e) => {
  const rect = previewZoom.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  const dx = (e.clientX - cx) / rect.width;
  const dy = (e.clientY - cy) / rect.height;
  previewZoomImg.style.transform = `translate3d(${dx * 14}px, ${dy * 14}px, 0) scale(1.01)`;
});
previewZoom.addEventListener('mouseleave', () => {
  previewZoomImg.style.transform = `translate3d(0,0,0) scale(1)`;
});

resetBtn.addEventListener('click', () => {
  preview.src = '';
  preview.style.display = 'none';
  uploadLabel.style.display = 'flex';
  predictBtn.disabled = true;
  resetBtn.style.display = 'none';
  resultCard.style.display = 'none';
  const bars = document.querySelectorAll('.result-graph .bar');
  bars.forEach(b => (b.style.width = '0%'));
  meterFill.style.width = '10%';
  meterState.textContent = 'Ready';
  if (heatmapImg) heatmapImg.classList.remove('visible');
  log('Reset UI');
});

const particles = [];
let particleAnimId = null;

function createParticles(count = 28) {
  if (!universe) return;
  universe.innerHTML = '';
  for (let i = 0; i < count; i++) {
    const d = document.createElement('div');
    d.className = 'particle';
    const s = 2 + Math.random() * 8;
    d.style.width = `${s}px`;
    d.style.height = `${s}px`;
    d.style.left = `${Math.random() * 100}%`;
    d.style.top = `${Math.random() * 100}%`;
    d.dataset.baseX = parseFloat(d.style.left);
    d.dataset.baseY = parseFloat(d.style.top);
    d.style.opacity = `${0.05 + Math.random() * 0.6}`;
    universe.appendChild(d);
    particles.push(d);
  }
}
let lastParticleTime = 0;
function particleLoop(t) {
  if (!lastParticleTime) lastParticleTime = t;
  const dt = t - lastParticleTime;
  lastParticleTime = t;
  particles.forEach((p, i) => {
    const jitter = Math.sin((t / 1000) * (0.2 + i * 0.01) + i) * 6;
    const x = parseFloat(p.dataset.baseX) + jitter * 0.06;
    const y = parseFloat(p.dataset.baseY) + Math.cos((t / 1500) * (0.35 + i * 0.01)) * 3;
    p.style.transform = `translate3d(${(x - parseFloat(p.style.left)) * 1.2}px, ${(y - parseFloat(p.style.top)) * 1.0}px, 0)`;
    p.style.opacity = `${0.04 + Math.abs(Math.sin((t + i * 333) / 1500)) * 0.35}`;
  });
  particleAnimId = requestAnimationFrame(particleLoop);
}

document.addEventListener('mousemove', (e) => {
  const mx = (e.clientX / window.innerWidth - 0.5) * 20;
  const my = (e.clientY / window.innerHeight - 0.5) * 20;
  if (universe) universe.style.transform = `translate3d(${mx}px, ${my}px, 0) rotateZ(${mx * 0.02}deg)`;
});

createParticles(28);
particleAnimId = requestAnimationFrame(particleLoop);

let analyzerTimer = null;
function startAnalyzer() {
  if (!meterFill || !meterState) return;
  meterState.textContent = 'Analyzing...';
  let p = 6;
  clearInterval(analyzerTimer);
  analyzerTimer = setInterval(() => {
    p += Math.random() * 7 + 1;
    if (p >= 96) {
      clearInterval(analyzerTimer);
      p = 96;
    }
    meterFill.style.width = `${Math.round(p)}%`;
    meterFill.style.boxShadow = `0 10px ${20 + p / 2}px rgba(108,99,255,${Math.min(0.25, p / 400)})`;
  }, 260);
}
function completeAnalyzer() {
  if (!meterFill || !meterState) return;
  meterFill.style.width = '100%';
  meterState.textContent = 'Complete';
  meterFill.style.boxShadow = '0 20px 60px rgba(108,99,255,0.18)';
  setTimeout(() => {
    meterFill.style.boxShadow = '0 8px 22px rgba(108,99,255,0.08)';
  }, 1200);
}

function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }
function animateBars(realPct, aiPct, duration = 1200) {
  const r = document.querySelector('.result-graph .bar.real');
  const a = document.querySelector('.result-graph .bar.ai');
  if (!r || !a) return;
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const p = easeOutCubic(progress);
    r.style.width = `${(realPct * p).toFixed(2)}%`;
    a.style.width = `${(aiPct * p).toFixed(2)}%`;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

let confChart = null;
try {
  if (confCanvas) {
    confChart = new Chart(confCanvas, {
      type: 'bar',
      data: {
        labels: ['AI', 'Real'],
        datasets: [{
          data: [50, 50],
          backgroundColor: ['rgba(108,99,255,0.95)', 'rgba(255,111,181,0.95)'],
          borderRadius: 8
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, max: 100, ticks: { color: '#bff6ff' } },
          x: { ticks: { color: '#bff6ff' } }
        }
      }
    });
  }
} catch (e) {
  console.warn('Chart init failed', e);
}

predictBtn.addEventListener('click', async () => {
  if (!fileInput.files[0]) return alert('Please upload an image first.');
  startAnalyzer();
  flashScanOverlay(2400);
  predictBtn.disabled = true;
  log('Uploading image for prediction...');

  const fd = new FormData();
  fd.append('file', fileInput.files[0]);

  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();

    if (data.error) throw new Error(data.error || 'Unknown prediction error');

    clearInterval(analyzerTimer);
    completeAnalyzer();

    const labelRaw = (data.label || '').toString();
    const conf = Math.max(0, Math.min(100, Math.round((data.confidence || 0) * 100)));
    const label = labelRaw.toLowerCase();
    let aiPct = 0, realPct = 0;
    if (label.includes('ai') || label.includes('generated')) {
      aiPct = conf; realPct = 100 - conf;
    } else {
      realPct = conf; aiPct = 100 - conf;
    }


    const feedbackBox = document.getElementById("feedback");
    feedbackBox.style.display = "block";

    document.getElementById("thumbs-up").onclick = async () => {
      await sendFeedback(true);
      feedbackBox.style.display = "none";
      alert("✅ Thanks! Your feedback helps improve the model.");
    };

    document.getElementById("thumbs-down").onclick = async () => {
      await sendFeedback(false);
      feedbackBox.style.display = "none";
      alert("⚠️ Got it! We'll adjust the model accordingly.");
    };

    async function sendFeedback(correct) {
      const fd = new FormData();
      fd.append("file", fileInput.files[0]);
      fd.append("label", correct ? data.label : (data.label.includes("real") ? "ai_generated" : "real"));
      try {
        const res = await fetch("/feedback", { method: "POST", body: fd });
        const info = await res.json();
        console.log("Feedback response:", info);
      } catch (err) {
        console.error("Feedback failed:", err);
      }
    }

    setTimeout(() => {
      resultCard.style.display = 'block';
      resultLabel.textContent = `Prediction: ${labelRaw.toUpperCase()}`;

      const start = performance.now();
      const from = 0;
      const to = conf;
      (function tick(now) {
        const t = Math.min((now - start) / 1200, 1);
        const eased = easeOutCubic(t);
        const val = (from + (to - from) * eased).toFixed(1);
        resultConfidence.textContent = `Confidence: ${val}%`;
        if (t < 1) requestAnimationFrame(tick);
      })(performance.now());

      const feedbackBox = document.getElementById("feedback");
      if (feedbackBox) {
        feedbackBox.style.display = "block";

        const thumbsUp = document.getElementById("thumbs-up");
        const thumbsDown = document.getElementById("thumbs-down");

        thumbsUp.onclick = async () => {
          await sendFeedback(true, data.label);
          feedbackBox.style.display = "none";
          alert("✅ Thanks! Your feedback helps improve the model.");
        };

        thumbsDown.onclick = async () => {
          await sendFeedback(false, data.label);
          feedbackBox.style.display = "none";
          alert("⚠️ Got it! We’ll adjust the model accordingly.");
        };
      }

      async function sendFeedback(correct, label) {
        const fd = new FormData();
        fd.append("file", fileInput.files[0]);
        fd.append("label", correct ? label : (label.includes("real") ? "ai_generated" : "real"));

        try {
          const res = await fetch("/feedback", { method: "POST", body: fd });
          const info = await res.json();
          console.log("Feedback Response:", info);
        } catch (err) {
          console.error("Feedback failed:", err);
        }
      }

      animateBars(realPct, aiPct, 1200);
      resultCard.style.transform = 'translateY(8px) scale(.995)';
      resultCard.style.opacity = '0';
      requestAnimationFrame(() => {
        resultCard.style.transition = 'transform .6s cubic-bezier(.2,.9,.2,1), opacity .5s ease';
        resultCard.style.transform = 'translateY(0) scale(1)';
        resultCard.style.opacity = '1';
      });

      predictBtn.disabled = false;
      log(`Prediction: ${labelRaw} (${conf}%)`);
    }, 540);

    if (data.heatmap && heatmapImg) {
      const newSrc = data.heatmap + '?v=' + Date.now();
      heatmapImg.classList.remove('visible');
      setTimeout(() => {
        heatmapImg.src = newSrc;
        heatmapImg.onload = () => heatmapImg.classList.add('visible');
      }, 420);
      log('Updated Grad-CAM heatmap.');
    }

    if (confChart) {
      confChart.data.datasets[0].data = [aiPct, realPct];
      confChart.update();
      const chartSummary = $('#chart-summary');
      if (chartSummary) {
        const aiBar = chartSummary.querySelector('.ai-fill');
        const realBar = chartSummary.querySelector('.real-fill');
        const aiLabel = chartSummary.querySelector('.ai-val');
        const realLabel = chartSummary.querySelector('.real-val');
        if (aiBar) aiBar.style.width = `${aiPct}%`;
        if (realBar) realBar.style.width = `${realPct}%`;
        if (aiLabel) aiLabel.textContent = `AI: ${aiPct.toFixed(1)}%`;
        if (realLabel) realLabel.textContent = `Real: ${realPct.toFixed(1)}%`;
        chartSummary.querySelectorAll('.summary-labels span').forEach(s => {
          s.classList.add('updated');
          setTimeout(() => s.classList.remove('updated'), 480);
        });
      }
    }

    if (conf >= 96) {
      microPulse();
    }

  } catch (err) {
    console.error(err);
    meterState.textContent = 'Error';
    predictBtn.disabled = false;
    clearInterval(analyzerTimer);
    log(`❌ Error: ${err.message}`);
    alert('Prediction failed — see console or backend logs.');
  }
});

function microPulse() {
  const top = document.querySelector('.topbar');
  if (top) {
    top.style.transition = 'box-shadow .45s ease, transform .45s ease';
    top.style.boxShadow = '0 10px 60px rgba(108,99,255,0.18)';
    top.style.transform = 'translateY(-3px)';
    setTimeout(() => {
      top.style.boxShadow = '';
      top.style.transform = '';
    }, 700);
  }
  if (resultCard) {
    resultCard.animate([
      { transform: 'scale(1)' },
      { transform: 'scale(1.02)' },
      { transform: 'scale(1)' }
    ], { duration: 600, easing: 'cubic-bezier(.2,.9,.2,1)' });
  }
}

setInterval(() => {
  if (!analyzerTimer && meterFill) {
    const p = 10 + Math.abs(Math.sin(Date.now() / 2000) * 12);
    meterFill.style.width = `${p}%`;
  }
}, 900);

(function initRevealObserver() {
  const reveals = $$('.reveal, .insight-item, .tech-card, .stat-block, .chart-card');
  if (!reveals.length) return;
  const obs = new IntersectionObserver((entries, o) => {
    entries.forEach(en => {
      if (en.isIntersecting) {
        en.target.classList.add('inview');
        en.target.style.transition = 'transform .72s cubic-bezier(.2,.9,.2,1), opacity .72s ease';
        en.target.style.opacity = '1';
        en.target.style.transform = 'translateY(0)';
        o.unobserve(en.target);
      }
    });
  }, { threshold: 0.12 });
  reveals.forEach(r => {
    r.style.opacity = '0';
    r.style.transform = 'translateY(18px)';
    obs.observe(r);
  });
})();

window.addEventListener('scroll', throttle(() => {
  const cards = $$('.tech-card');
  cards.forEach((c, idx) => {
    const rect = c.getBoundingClientRect();
    if (rect.top < window.innerHeight * 0.85) {
      c.style.transitionDelay = `${idx * 60}ms`;
      c.classList.add('inview');
      c.style.opacity = '1';
      c.style.transform = 'translateY(0)';
    }
  });
}, 120));

(function initStatsCount() {
  const blocks = $$('.stat-block');
  if (!blocks.length) return;
  const obs = new IntersectionObserver((entries, o) => {
    entries.forEach(en => {
      if (en.isIntersecting) {
        const p = en.target.querySelector('p');
        if (p && !p.dataset.counted) {
          const raw = p.textContent.trim();
          const n = parseFloat(raw.replace(/[^0-9.]/g, '')) || 0;
          animateCountUp(p, n, 1500);
          p.dataset.counted = '1';
        }
        o.unobserve(en.target);
      }
    });
  }, { threshold: 0.3 });
  blocks.forEach(b => obs.observe(b));
})();

function animateCountUp(el, target, duration = 1200) {
  const start = performance.now();
  const from = 0;
  (function step(now) {
    const t = Math.min((now - start) / duration, 1);
    const eased = easeOutCubic(t);
    const val = Math.floor(from + (target - from) * eased);
    el.textContent = val.toLocaleString();
    if (t < 1) requestAnimationFrame(step);
  })(performance.now());
}

(function initPerformanceCharts() {
  try {
    if (!accuracyCanvas || !lossCanvas) return;
    const epochs = Array.from({ length: 20 }, (_, i) => i + 1);
    const trainAcc = [0.78, 0.82, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9995, 1.00].map(v => v * 100);
    const valAcc = [0.74, 0.78, 0.81, 0.83, 0.86, 0.89, 0.90, 0.92, 0.94, 0.95, 0.96, 0.965, 0.97, 0.972, 0.974, 0.9745, 0.975, 0.975, 0.975, 0.975].map(v => v * 100);
    const trainLoss = [0.40, 0.30, 0.24, 0.20, 0.17, 0.14, 0.12, 0.10, 0.08, 0.07, 0.065, 0.062, 0.060, 0.058, 0.057, 0.055, 0.054, 0.053, 0.052, 0.051];
    const valLoss = [0.50, 0.40, 0.32, 0.28, 0.24, 0.20, 0.18, 0.16, 0.14, 0.12, 0.11, 0.105, 0.102, 0.100, 0.098, 0.097, 0.096, 0.095, 0.094, 0.093];

    new Chart(accuracyCanvas, {
      type: 'line',
      data: {
        labels: epochs, datasets: [
          { label: 'Train', data: trainAcc, borderColor: 'rgba(108,99,255,0.95)', backgroundColor: 'rgba(108,99,255,0.12)', fill: true, tension: 0.28 },
          { label: 'Val', data: valAcc, borderColor: 'rgba(255,111,181,0.95)', backgroundColor: 'rgba(255,111,181,0.08)', fill: true, tension: 0.28 }
        ]
      },
      options: { responsive: true, plugins: { legend: { position: 'bottom', labels: { color: '#bff6ff' } } }, scales: { x: { ticks: { color: '#bff6ff' } }, y: { ticks: { color: '#bff6ff' } } } }
    });
    new Chart(lossCanvas, {
      type: 'line',
      data: {
        labels: epochs, datasets: [
          { label: 'Train Loss', data: trainLoss, borderColor: 'rgba(120,120,255,0.9)', backgroundColor: 'rgba(120,120,255,0.06)', fill: true, tension: 0.28 },
          { label: 'Val Loss', data: valLoss, borderColor: 'rgba(255,120,170,0.9)', backgroundColor: 'rgba(255,120,170,0.04)', fill: true, tension: 0.28 }
        ]
      },
      options: { responsive: true, plugins: { legend: { position: 'bottom', labels: { color: '#bff6ff' } } }, scales: { x: { ticks: { color: '#bff6ff' } }, y: { ticks: { color: '#bff6ff' } } } }
    });
  } catch (e) {
    console.warn('Performance charts init failed', e);
  }
})();

if (themeToggle) {
  themeToggle.addEventListener('click', () => {
    document.documentElement.classList.toggle('light-theme');
    themeToggle.animate([{ transform: 'rotate(0deg)' }, { transform: 'rotate(360deg)' }], { duration: 600 });
    log('Toggled theme');
  });
}

function throttle(fn, wait = 100) {
  let last = 0;
  return function (...args) {
    const now = Date.now();
    if (now - last >= wait) {
      last = now;
      fn.apply(this, args);
    }
  };
}
function debounce(fn, wait = 120) {
  let t;
  return function (...args) {
    clearTimeout(t);
    t = setTimeout(() => fn.apply(this, args), wait);
  };
}

document.addEventListener('keydown', (e) => {
  if (e.key === ' ' && document.activeElement.tagName !== 'INPUT') {
    e.preventDefault();
    if (!predictBtn.disabled) predictBtn.click();
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'u') {
    fileInput.click();
  }
});

window.addEventListener('beforeunload', () => {
  if (particleAnimId) cancelAnimationFrame(particleAnimId);
  if (analyzerTimer) clearInterval(analyzerTimer);
});

window.addEventListener('resize', debounce(() => {
}, 200));

log('Interface ready — enhanced script loaded.');
setTimeout(() => log('Tip: drag & drop an image or click the camera to upload.'), 800);
