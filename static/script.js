/* ═══════════════════════════════════════════════════════════════════
   EmotiSense — Living Bokeh Bulbs & Emotion Perception Engine
   ═══════════════════════════════════════════════════════════════════ */

'use strict';

// ─── Emotion Knowledge & Vibrant Bokeh Palettes ─────────────────────
const EMOTIONS = {
  joy: {
    emoji: '😄',
    name: 'Joy',
    headingWord: 'heart',
    desc: 'A gentle warmth of comfort, optimism, and quiet delight.',
    color: '#f59e0b',
    rgb: [245, 158, 11],
    bokehColors: [
      [180, 85, 10],   // Deep warm amber bulb
      [150, 70, 12],   // Golden caramel disc
      [195, 75, 15],   // Dark honey ring
      [130, 60, 15],   // Terracotta ember
      [165, 80, 20],   // Warm cider glow
      [140, 65, 10]    // Deep nocturnal marigold
    ]
  },
  sadness: {
    emoji: '😢',
    name: 'Sadness',
    headingWord: 'soul',
    desc: 'A quiet, deep ache of longing, tender memory, and reflective rain.',
    color: '#60a5fa',
    rgb: [96, 165, 250],
    bokehColors: [
      [30, 58, 138],   // Nocturnal navy bulb
      [25, 70, 190],   // Deep ocean sapphire disc
      [20, 40, 95],    // Dark abyss indigo ring
      [35, 90, 180],   // Muted twilight mist
      [18, 50, 125],   // Deep midnight blue
      [28, 65, 160]    // Midnight rain
    ]
  },
  love: {
    emoji: '❤️',
    name: 'Love',
    headingWord: 'heart',
    desc: 'Radiant tenderness, intimacy, and warm magnetic connection.',
    color: '#fb7185',
    rgb: [251, 113, 133],
    bokehColors: [
      [159, 18, 57],   // Velvet wine disc
      [136, 19, 55],   // Deep ruby crimson ring
      [180, 25, 70],   // Dusky rose bulb
      [115, 12, 45],   // Dark berry ember
      [140, 20, 55],   // Twilight plum
      [125, 15, 45]    // Warm dark blossom
    ]
  },
  anger: {
    emoji: '😠',
    name: 'Anger',
    headingWord: 'fire',
    desc: 'Surging heat, intense defiance, and fierce protective energy.',
    color: '#f87171',
    rgb: [248, 113, 113],
    bokehColors: [
      [153, 27, 27],   // Smoldering cinder disc
      [127, 29, 29],   // Deep garnet ring
      [175, 28, 28],   // Charcoal hearth bulb
      [105, 18, 24],   // Dark volcanic ember
      [135, 22, 22],   // Deep ruby ash
      [115, 15, 20]    // Molten hearth
    ]
  },
  fear: {
    emoji: '😨',
    name: 'Fear',
    headingWord: 'shadows',
    desc: 'Cold instinctive alert, watchful vigilance, and breathless suspension.',
    color: '#c084fc',
    rgb: [192, 132, 252],
    bokehColors: [
      [88, 28, 135],   // Shadow amethyst disc
      [107, 33, 168],  // Deep twilight violet ring
      [60, 10, 105],   // Phantom abyss bulb
      [76, 29, 149],   // Nocturnal plum
      [90, 30, 140],   // Dark slate orchid
      [70, 20, 120]    // Quiet dusk
    ]
  },
  surprise: {
    emoji: '😲',
    name: 'Surprise',
    headingWord: 'wonder',
    desc: 'An inspiring jolt of awe, sudden revelation, and wide-eyed wonder.',
    color: '#2dd4bf',
    rgb: [45, 212, 191],
    bokehColors: [
      [15, 118, 110],  // Deep sea teal disc
      [13, 148, 136],  // Muted ocean jade ring
      [19, 85, 80],    // Dark nocturnal marine bulb
      [20, 110, 120],  // Dusky deep cyan
      [12, 75, 95],    // Shadow lagoon
      [18, 95, 90]     // Quiet sea-glass
    ]
  },
  neutral: {
    emoji: '🤔',
    name: 'Presence',
    headingWord: 'soul',
    desc: 'Calm equilibrium, expectant stillness, and open perception.',
    color: '#94a3b8',
    rgb: [148, 163, 184],
    bokehColors: [
      [20, 45, 90],    // Deep nocturnal sapphire
      [140, 80, 20],   // Warm golden amber bulb
      [90, 25, 60],    // Deep velvet wine ring
      [18, 75, 85],    // Dark sea teal disc
      [45, 60, 90],    // Dusky slate mist
      [115, 65, 20]    // Gentle hearth glint
    ]
  }
};

// ─── DOM References ────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const bokehCanvas = $('bokehCanvas');
const textInput = $('textInput');
const charCount = $('charCount');
const analyzeBtn = $('analyzeBtn');
const statusBadge = $('statusBadge');
const statusText = $('statusText');
const errorToast = $('errorToast');
const errorMsg = $('errorMsg');
const resultSection = $('resultSection');
const headingEm = $('headingEm');

// Centerpiece Orb
const orbWrapper = $('orbWrapper');
const orbEmoji = $('orbEmoji');
const orbRipple1 = $('orbRipple1');
const orbRipple2 = $('orbRipple2');

// Result elements
const bigEmoji = $('bigEmoji');
const bigName = $('bigName');
const confidenceLine = $('confidenceLine');
const emotionDesc = $('emotionDesc');
const analyzedQuote = $('analyzedQuote');
const breakdownRows = $('breakdownRows');

// History elements
const historySection = $('historySection');
const historyList = $('historyList');
const clearHistory = $('clearHistory');

// State
let currentEmotion = 'neutral';
let historyState = [];
const STORAGE_KEY = 'emotisense_history_v2';

// ─── 1. Living Bokeh Canvas Atmosphere Engine ──────────────────────
class BokehEngine {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.particles = [];
    this.glitters = [];
    this.mouse = { x: -9999, y: -9999, targetX: -9999, targetY: -9999 };
    this.currentPalette = EMOTIONS.neutral.bokehColors;
    this.w = 0;
    this.h = 0;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);

    this.init();
  }

  init() {
    this.resize();
    window.addEventListener('resize', () => this.resize());

    // Gentle mouse influence
    window.addEventListener('mousemove', e => {
      this.mouse.targetX = e.clientX;
      this.mouse.targetY = e.clientY;
    });

    window.addEventListener('mouseleave', () => {
      this.mouse.targetX = -9999;
      this.mouse.targetY = -9999;
    });

    this.createParticles();
    this.createGlitters();
    this.animate();
  }

  resize() {
    this.w = window.innerWidth;
    this.h = window.innerHeight;
    this.canvas.width = this.w * this.dpr;
    this.canvas.height = this.h * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);

    if (this.particles.length > 0) {
      this.particles.forEach(p => {
        p.x = Math.random() * this.w;
        p.y = Math.random() * this.h;
      });
    }
  }

  createParticles() {
    // Rich layer of 70-92 bokeh bulbs clearly visible and floating in the atmosphere
    const count = Math.min(Math.max(Math.floor((this.w * this.h) / 13000), 70), 92);
    this.particles = [];

    for (let i = 0; i < count; i++) {
      const paletteCol = this.currentPalette[i % this.currentPalette.length];
      const depth = Math.random(); // 0 (far/blurry) to 1 (near/sharp)

      // 3 types: 'ring' (hollow aperture ring), 'disc' (solid glowing bulb), 'bloom' (soft ambient bokeh)
      let type;
      let baseRadius;
      let baseAlpha;

      const randType = Math.random();
      if (randType < 0.45) {
        // Soft defocused aperture ring (feathered donut)
        type = 'ring';
        baseRadius = 24 + Math.random() * 52;
        baseAlpha = 0.36 + Math.random() * 0.32;
      } else if (randType < 0.84) {
        // Soft glowing disc with luminous center
        type = 'disc';
        baseRadius = 20 + Math.random() * 46;
        baseAlpha = 0.40 + Math.random() * 0.35;
      } else {
        // Soft atmospheric ambient bloom (moderate size, never washes out)
        type = 'bloom';
        baseRadius = 55 + Math.random() * 55;
        baseAlpha = 0.16 + Math.random() * 0.18;
      }

      const hasTearStreak = Math.random() < 0.35;
      const streakAngle = (Math.random() - 0.5) * 0.35; // gentle vertical tilt like eyelashes/tears

      this.particles.push({
        type,
        x: Math.random() * this.w,
        y: Math.random() * this.h,
        baseRadius,
        radius: baseRadius,
        hasTearStreak,
        streakAngle,
        // Gentle upward buoyant float with organic horizontal sway
        vx: (Math.random() - 0.5) * 0.28,
        vy: -(0.2 + depth * 0.4),
        depth,
        baseAlpha,
        alpha: baseAlpha,
        pulseSpeed: 0.008 + Math.random() * 0.015,
        pulsePhase: Math.random() * Math.PI * 2,
        r: paletteCol[0],
        g: paletteCol[1],
        b: paletteCol[2],
        targetR: paletteCol[0],
        targetG: paletteCol[1],
        targetB: paletteCol[2],
        driftOffset: Math.random() * 100,
        swaySpeed: 0.007 + Math.random() * 0.012
      });
    }
  }

  createGlitters() {
    // 70 tiny twinkling sparkling dust motes/stars
    this.glitters = [];
    for (let i = 0; i < 70; i++) {
      this.glitters.push({
        x: Math.random() * this.w,
        y: Math.random() * this.h,
        size: 1 + Math.random() * 2.2,
        vy: -(0.15 + Math.random() * 0.35),
        phase: Math.random() * Math.PI * 2,
        phaseSpeed: 0.02 + Math.random() * 0.04,
        baseAlpha: 0.35 + Math.random() * 0.6
      });
    }
  }

  setPalette(emotionKey) {
    const emotion = EMOTIONS[emotionKey] || EMOTIONS.neutral;
    this.currentPalette = emotion.bokehColors;

    this.particles.forEach((p, i) => {
      const targetCol = this.currentPalette[i % this.currentPalette.length];
      p.targetR = targetCol[0];
      p.targetG = targetCol[1];
      p.targetB = targetCol[2];
    });
  }

  animate() {
    // Smooth mouse easing
    this.mouse.x += (this.mouse.targetX - this.mouse.x) * 0.05;
    this.mouse.y += (this.mouse.targetY - this.mouse.y) * 0.05;

    // Clear canvas
    this.ctx.clearRect(0, 0, this.w, this.h);

    // 1. Draw Sparkling Glitters / Stardust
    this.ctx.globalCompositeOperation = 'source-over';
    for (let i = 0; i < this.glitters.length; i++) {
      const g = this.glitters[i];
      g.y += g.vy;
      g.phase += g.phaseSpeed;
      g.x += Math.sin(g.phase) * 0.25;

      if (g.y < -10) {
        g.y = this.h + 10;
        g.x = Math.random() * this.w;
      }

      const gAlpha = Math.max(0.1, (Math.sin(g.phase) * 0.5 + 0.5) * g.baseAlpha);
      this.ctx.fillStyle = `rgba(255, 255, 255, ${gAlpha})`;
      this.ctx.beginPath();
      this.ctx.arc(g.x, g.y, g.size, 0, Math.PI * 2);
      this.ctx.fill();
    }

    // 2. Draw Moving Bokeh Bulbs (Screen composite mode for authentic vibrant optics)
    this.ctx.globalCompositeOperation = 'screen';

    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];

      // Smooth exponential color interpolation (transforms world in ~1.2s)
      p.r += (p.targetR - p.r) * 0.045;
      p.g += (p.targetG - p.g) * 0.045;
      p.b += (p.targetB - p.b) * 0.045;

      // Continuous upward float with gentle organic sway
      p.pulsePhase += p.pulseSpeed;
      p.driftOffset += p.swaySpeed;
      const wobbleX = Math.sin(p.driftOffset) * 0.45;

      p.x += p.vx + wobbleX;
      p.y += p.vy;

      // Mouse leaning / gentle gravitational interaction
      if (this.mouse.x > -1000) {
        const dx = this.mouse.x - p.x;
        const dy = this.mouse.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 240 && dist > 5) {
          const force = (1 - dist / 240) * 0.35;
          p.x += (dx / dist) * force;
          p.y += (dy / dist) * force;
        }
      }

      // Screen wrapping (continuous upward loop)
      const pad = p.baseRadius * 1.5;
      if (p.y < -pad) {
        p.y = this.h + pad;
        p.x = Math.random() * this.w;
      }
      if (p.x < -pad) p.x = this.w + pad;
      if (p.x > this.w + pad) p.x = -pad;

      // Pulsing alpha & radius
      const pulse = Math.sin(p.pulsePhase);
      p.alpha = Math.max(0.08, p.baseAlpha + pulse * 0.15);
      p.radius = p.baseRadius + pulse * 3;

      const rInt = Math.round(p.r);
      const gInt = Math.round(p.g);
      const bInt = Math.round(p.b);

      // Wet-Eye Tear Refraction Streak (vertical/diagonal light diffraction through tear meniscus)
      if (p.hasTearStreak && p.alpha > 0.22) {
        this.ctx.save();
        this.ctx.translate(p.x, p.y);
        this.ctx.rotate(p.streakAngle);
        const streakH = p.radius * (3.0 + p.depth * 2.8);
        const streakW = p.radius * 0.4;
        const streakGrad = this.ctx.createLinearGradient(0, -streakH, 0, streakH);
        streakGrad.addColorStop(0, 'rgba(0,0,0,0)');
        streakGrad.addColorStop(0.3, `rgba(${rInt}, ${gInt}, ${bInt}, ${p.alpha * 0.28})`);
        streakGrad.addColorStop(0.5, `rgba(255, 255, 255, ${p.alpha * 0.5})`);
        streakGrad.addColorStop(0.7, `rgba(${rInt}, ${gInt}, ${bInt}, ${p.alpha * 0.28})`);
        streakGrad.addColorStop(1, 'rgba(0,0,0,0)');
        this.ctx.fillStyle = streakGrad;
        this.ctx.fillRect(-streakW / 2, -streakH, streakW, streakH * 2);
        this.ctx.restore();
      }

      // Render based on Bokeh Bulb Type (Ultra-soft, dreamy, feathered gradients)
      // Render based on Bokeh Bulb Type (Ultra-soft, dreamy, feathered gradients)
      if (p.type === 'ring') {
        // Defocused aperture ring (feathered donut)
        const ringGrad = this.ctx.createRadialGradient(
          p.x, p.y, p.radius * 0.45,
          p.x, p.y, p.radius
        );
        ringGrad.addColorStop(0, `rgba(${rInt}, ${gInt}, ${bInt}, 0)`);
        ringGrad.addColorStop(0.65, `rgba(${rInt}, ${gInt}, ${bInt}, ${p.alpha * 0.85})`);
        ringGrad.addColorStop(0.85, `rgba(255, 255, 255, ${p.alpha * 0.45})`);
        ringGrad.addColorStop(1, 'rgba(0,0,0,0)');

        this.ctx.fillStyle = ringGrad;
        this.ctx.beginPath();
        this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        this.ctx.fill();

      } else if (p.type === 'disc') {
        // Glowing bokeh disc with luminous center
        const grad = this.ctx.createRadialGradient(
          p.x, p.y, 0,
          p.x, p.y, p.radius
        );
        grad.addColorStop(0, `rgba(255, 255, 255, ${p.alpha * 0.7})`);
        grad.addColorStop(0.25, `rgba(${rInt}, ${gInt}, ${bInt}, ${p.alpha * 0.95})`);
        grad.addColorStop(0.65, `rgba(${rInt}, ${gInt}, ${bInt}, ${p.alpha * 0.4})`);
        grad.addColorStop(0.9, `rgba(${rInt}, ${gInt}, ${bInt}, ${p.alpha * 0.06})`);
        grad.addColorStop(1, 'rgba(0,0,0,0)');

        this.ctx.fillStyle = grad;
        this.ctx.beginPath();
        this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        this.ctx.fill();

      } else {
        // Atmospheric soft ambient bloom
        const grad = this.ctx.createRadialGradient(
          p.x, p.y, 0,
          p.x, p.y, p.radius
        );
        grad.addColorStop(0, `rgba(${rInt}, ${gInt}, ${bInt}, ${p.alpha * 0.55})`);
        grad.addColorStop(0.45, `rgba(${rInt}, ${gInt}, ${bInt}, ${p.alpha * 0.22})`);
        grad.addColorStop(0.8, `rgba(${rInt}, ${gInt}, ${bInt}, ${p.alpha * 0.04})`);
        grad.addColorStop(1, 'rgba(0,0,0,0)');

        this.ctx.fillStyle = grad;
        this.ctx.beginPath();
        this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        this.ctx.fill();
      }
    }

    requestAnimationFrame(() => this.animate());
  }
}

const bokehEngine = new BokehEngine(bokehCanvas);

// ─── 2. Smart Resilient API Fetch ──────────────────────────────────
async function apiFetch(endpoint, options = {}) {
  const isLocalFile = window.location.protocol === 'file:' || !window.location.origin || window.location.origin === 'null';
  const url = isLocalFile ? `http://127.0.0.1:8000${endpoint}` : endpoint;
  return await fetch(url, options);
}

async function checkHealth() {
  try {
    const res = await apiFetch('/health');
    if (!res.ok) throw new Error();
    const data = await res.json();
    if (data.model_loaded) {
      statusBadge.className = 'status-badge online';
      statusText.textContent = 'Model Online';
    } else {
      statusBadge.className = 'status-badge';
      statusText.textContent = 'Model Ready';
    }
  } catch {
    statusBadge.className = 'status-badge offline';
    statusText.textContent = 'Connecting…';
  }
}

// ─── 3. Input Handling & Character Counter ─────────────────────────
textInput.addEventListener('input', () => {
  const len = textInput.value.length;
  charCount.textContent = `${len} / 2000`;
  charCount.style.color = len > 1900 ? '#f87171' : '';
  hideError();
});

textInput.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    triggerAnalysis();
  }
});

// Example Chips
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', e => {
    createChipRipple(e, chip);
    textInput.value = chip.dataset.text;
    textInput.dispatchEvent(new Event('input'));
    textInput.focus();
    triggerAnalysis();
  });
});

function createChipRipple(e, chip) {
  const rect = chip.getBoundingClientRect();
  const circle = document.createElement('span');
  const d = Math.max(rect.width, rect.height);
  circle.style.width = circle.style.height = `${d}px`;
  circle.style.left = `${e.clientX - rect.left - d / 2}px`;
  circle.style.top = `${e.clientY - rect.top - d / 2}px`;
  circle.style.position = 'absolute';
  circle.style.borderRadius = '50%';
  circle.style.background = 'rgba(255, 255, 255, 0.3)';
  circle.style.transform = 'scale(0)';
  circle.style.animation = 'rippleExpand 0.6s linear';
  circle.style.pointerEvents = 'none';

  chip.appendChild(circle);
  setTimeout(() => circle.remove(), 600);
}

// Orb Click Interactive Effect
orbWrapper.addEventListener('click', () => {
  triggerOrb3DFlip();
  triggerOrbRipples();
});

// ─── 4. Analysis Request Execution ─────────────────────────────────
analyzeBtn.addEventListener('click', triggerAnalysis);

async function triggerAnalysis() {
  const text = textInput.value.trim();
  if (!text) {
    showError('Please write or select a sentence to perceive its emotion.');
    textInput.focus();
    return;
  }

  setLoading(true);
  hideError();

  try {
    const res = await apiFetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Perception failed (Status ${res.status})`);
    }

    const data = await res.json();
    renderAnalysisResult(data, text);
    saveToHistory(data, text);

  } catch (err) {
    showError(err.message || 'Could not connect to the neural model.');
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  if (isLoading) {
    analyzeBtn.classList.add('loading');
  } else {
    analyzeBtn.classList.remove('loading');
  }
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorToast.hidden = false;
}

function hideError() {
  errorToast.hidden = true;
}

// ─── 5. Result Rendering & Visual Metamorphosis ─────────────────────
function renderAnalysisResult(data, originalText) {
  const emotionKey = (data.predicted_emotion || 'neutral').toLowerCase();
  const meta = EMOTIONS[emotionKey] || EMOTIONS.neutral;
  const confidencePct = (data.confidence * 100).toFixed(1);

  currentEmotion = emotionKey;

  // 1. Update Body Attribute & Dynamic CSS Theme
  document.body.dataset.emotion = emotionKey;
  document.documentElement.style.setProperty('--emotion-color', meta.color);
  document.documentElement.style.setProperty('--emotion-color-rgb', meta.rgb.join(', '));
  document.documentElement.style.setProperty('--emotion-glow', `rgba(${meta.rgb.join(', ')}, 0.55)`);

  // Update header poetic word if present
  if (headingEm && meta.headingWord) {
    headingEm.textContent = meta.headingWord;
  }

  // 2. Metamorphosis of Living Bokeh Canvas
  bokehEngine.setPalette(emotionKey);

  // 3. Centerpiece Orb 3D Flip & Expanding Ripples
  orbEmoji.textContent = data.emoji || meta.emoji;
  triggerOrb3DFlip();
  triggerOrbRipples();

  // 4. Reveal & Populate Result Cards
  resultSection.hidden = false;

  // Detected Emotion Card
  bigEmoji.textContent = data.emoji || meta.emoji;

  // Typewriter effect for emotion name
  typewriterText(bigName, meta.name);

  confidenceLine.textContent = `${confidencePct}% confidence`;
  emotionDesc.textContent = meta.desc;
  analyzedQuote.textContent = `“${originalText}”`;

  // 5. Breakdown Progress Bars with Staggered Liquid Animation
  renderBreakdownBars(data, emotionKey);

  // Smooth scroll into focus if needed
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function triggerOrb3DFlip() {
  orbEmoji.classList.remove('flip-active');
  void orbEmoji.offsetWidth; // Reflow to restart keyframe
  orbEmoji.classList.add('flip-active');
}

function triggerOrbRipples() {
  orbRipple1.classList.remove('active');
  orbRipple2.classList.remove('active');
  void orbRipple1.offsetWidth;
  void orbRipple2.offsetWidth;
  orbRipple1.classList.add('active');
  orbRipple2.classList.add('active');
}

function typewriterText(element, text) {
  element.textContent = '';
  let i = 0;
  function step() {
    if (i < text.length) {
      element.textContent += text.charAt(i);
      i++;
      setTimeout(step, 45);
    }
  }
  step();
}

function renderBreakdownBars(data, winningEmotionKey) {
  const probs = data.all_probabilites || data.probabilities || {};
  const entries = Object.entries(probs);

  // Sort by probability descending
  entries.sort((a, b) => b[1] - a[1]);

  breakdownRows.innerHTML = '';

  entries.forEach(([key, val], idx) => {
    const emotionMeta = EMOTIONS[key.toLowerCase()] || { emoji: '✨', name: key };
    const pct = (val * 100).toFixed(1);
    const isWinner = key.toLowerCase() === winningEmotionKey;

    const row = document.createElement('div');
    row.className = `breakdown-row ${isWinner ? 'highlight' : ''}`;

    row.innerHTML = `
      <div class="breakdown-label">
        <span class="row-emoji">${emotionMeta.emoji}</span>
        <span>${emotionMeta.name}</span>
      </div>
      <div class="breakdown-track">
        <div class="breakdown-fill ${isWinner ? 'highlight' : ''}" id="bar-${idx}"></div>
      </div>
      <div class="breakdown-pct">${pct}%</div>
    `;

    breakdownRows.appendChild(row);

    // Staggered liquid bar growth
    setTimeout(() => {
      const fillEl = document.getElementById(`bar-${idx}`);
      if (fillEl) {
        fillEl.style.width = `${Math.max(val * 100, 2)}%`;
      }
    }, 60 + idx * 70);
  });
}

// ─── 6. Recent History Management ──────────────────────────────────
function loadHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      historyState = JSON.parse(raw);
      renderHistoryUI();
    }
  } catch (e) {
    historyState = [];
  }
}

function saveToHistory(data, text) {
  const item = {
    id: Date.now(),
    text,
    emotion: data.predicted_emotion,
    emoji: data.emoji || (EMOTIONS[data.predicted_emotion.toLowerCase()] || {}).emoji || '✨',
    confidence: data.confidence,
    data
  };

  // Avoid identical duplicate consecutive item
  if (historyState.length > 0 && historyState[0].text === text) return;

  historyState.unshift(item);
  if (historyState.length > 6) historyState.pop();

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(historyState));
  } catch (e) { }

  renderHistoryUI();
}

function renderHistoryUI() {
  if (!historySection || !historyList) return;
  if (historyState.length === 0) {
    historySection.hidden = true;
    return;
  }

  historySection.hidden = false;
  historyList.innerHTML = '';

  historyState.forEach(item => {
    const el = document.createElement('div');
    el.className = 'history-item';
    const confPct = Math.round(item.confidence * 100);

    el.innerHTML = `
      <div class="history-left">
        <span class="history-badge">${item.emoji} ${item.emotion}</span>
        <span class="history-snippet">“${item.text}”</span>
      </div>
      <span class="history-conf">${confPct}%</span>
    `;

    el.addEventListener('click', () => {
      textInput.value = item.text;
      textInput.dispatchEvent(new Event('input'));
      renderAnalysisResult(item.data, item.text);
    });

    historyList.appendChild(el);
  });
}

if (clearHistory) {
  clearHistory.addEventListener('click', () => {
    historyState = [];
    localStorage.removeItem(STORAGE_KEY);
    renderHistoryUI();
  });
}

// ─── 7. Initialization ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  loadHistory();

  // Check health periodically every 15s
  setInterval(checkHealth, 15000);

  // Handle query parameter demo/test (e.g. ?test=joy or ?test=fear or ?text=...)
  const urlParams = new URLSearchParams(window.location.search);
  const testEmotion = urlParams.get('test');
  if (testEmotion) {
    const chip = document.querySelector(`.chip[data-emotion="${testEmotion.toLowerCase()}"]`) || document.querySelector('.chip');
    textInput.value = urlParams.get('text') || (chip ? chip.dataset.text : "I can't believe how happy I am right now, this is amazing!");
    textInput.dispatchEvent(new Event('input'));
    setTimeout(triggerAnalysis, 300);
  }
});
