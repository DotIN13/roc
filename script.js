// Constants
const POS_RATIO = 0.5;   // 50% positive
const CANVAS_PADDING = 20;

// State
let state = {
    data: [], // Array of { value: float, label: 0|1 }
    separation: 2.0,
    points: 20,
    threshold: 0.5,
    metrics: {},
    educationMode: false,
    revealed: new Set() // 'tp', 'fp', 'fn', 'tn'
};

// DOM Elements
const els = {
    distCanvas: document.getElementById('dist-canvas'),
    rocCanvas: document.getElementById('roc-canvas'),
    handle: document.getElementById('threshold-handle'),
    distWrapper: document.getElementById('dist-wrapper'),
    separationInput: document.getElementById('separation'),
    separationVal: document.getElementById('separation-val'),
    pointsInput: document.getElementById('points'),
    pointsVal: document.getElementById('points-val'),
    thresholdVal: document.getElementById('threshold-val'),

    // Matrix
    tp: document.getElementById('tp-count'),
    fp: document.getElementById('fp-count'),
    fn: document.getElementById('fn-count'),
    tn: document.getElementById('tn-count'),
    // Metrics
    acc: document.getElementById('acc-val'),
    prec: document.getElementById('prec-val'),
    rec: document.getElementById('rec-val'),
    f1: document.getElementById('f1-val'),
    fpr: document.getElementById('fpr-val'),

    // Education Mode
    educationToggle: document.getElementById('education-mode'),
    metricsPanel: document.querySelector('.metrics-panel'),
    matrixCells: {
        tp: document.querySelector('.matrix-cell.tp'),
        fp: document.querySelector('.matrix-cell.fp'),
        fn: document.querySelector('.matrix-cell.fn'),
        tn: document.querySelector('.matrix-cell.tn')
    }
};

// Init
function init() {
    setupEventListeners();
    generateData();
    resizeCanvases();
    update();

    // Resize observer
    new ResizeObserver(() => {
        resizeCanvases();
        draw();
    }).observe(els.distWrapper);
}

function setupEventListeners() {
    // Separation Slider
    els.separationInput.addEventListener('input', (e) => {
        state.separation = parseFloat(e.target.value);
        els.separationVal.textContent = state.separation.toFixed(1);
        generateData();
        resetEducation();
        update();
    });

    // Points Slider
    els.pointsInput.addEventListener('input', (e) => {
        state.points = parseInt(e.target.value);
        els.pointsVal.textContent = state.points;
        generateData();
        resetEducation();
        update();
    });



    // Dragging Logic for Threshold
    let isDragging = false;

    els.handle.addEventListener('mousedown', () => isDragging = true);
    window.addEventListener('mouseup', () => isDragging = false);

    // Allow clicking on the canvas track as well
    els.distWrapper.addEventListener('mousedown', (e) => {
        if (e.target !== els.handle) {
            updateThresholdFromMouse(e);
            isDragging = true;
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        updateThresholdFromMouse(e);
        resetEducation(); // Reset if dragging
        update();
    });

    // Education Mode Toggle
    els.educationToggle.addEventListener('change', (e) => {
        state.educationMode = e.target.checked;
        if (!state.educationMode) {
            state.revealed.clear(); // irrelevant when off, but cleaner
        } else {
            state.revealed.clear(); // Reset to hidden on enable
        }
        update();
    });

    // Matrix Cell Clicks
    ['tp', 'fp', 'fn', 'tn'].forEach(key => {
        els.matrixCells[key].addEventListener('click', () => {
            if (state.educationMode && !state.revealed.has(key)) {
                state.revealed.add(key);
                updateUI();
            }
        });
    });
}

function resetEducation() {
    if (state.educationMode) {
        state.revealed.clear();
    }
}

function updateThresholdFromMouse(e) {
    const rect = els.distWrapper.getBoundingClientRect();
    let x = e.clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    state.threshold = x / rect.width;
    resetEducation();
    update();
}

// Seeded PRNG (Mulberry32)
// This ensures we get the exact same sequence of numbers for the same seed.
function mulberry32(a) {
    return function () {
        var t = a += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
}

// Global RNG instance
let rng = mulberry32(12345);

// Data Generation
function generateData() {
    // RESET RNG so it's deterministic per generation call
    // Using the same seed ensures 'separation' and 'points' produce the same specific dataset each time.
    rng = mulberry32(88675123);

    state.data = [];
    const numPos = Math.floor(state.points * POS_RATIO);
    const numNeg = state.points - numPos;

    // Generate Positives (Higher mean)
    // We map 0..1 range.
    // Separation 2.0 means means are separated by 2 sigmas effectively.
    // Let's keep sigma constant and move means.

    const sigma = 0.12;
    // Centered around 0.5.
    // Offset based on separation.
    const center = 0.5;
    const offset = (state.separation * sigma) / 2;

    const meanPos = Math.min(0.95, center + offset);
    const meanNeg = Math.max(0.05, center - offset);

    for (let i = 0; i < numPos; i++) {
        state.data.push({
            value: randomGaussian(meanPos, sigma),
            label: 1
        });
    }
    for (let i = 0; i < numNeg; i++) {
        state.data.push({
            value: randomGaussian(meanNeg, sigma),
            label: 0
        });
    }

    // Sort for easier plotting? Not strictly necessary but helpful for some viz types.
    // But for histogram we just bin them.
}

// Box-Muller transform for Gaussian noise
function randomGaussian(mean, stdev) {
    const u = 1 - rng();
    const v = rng();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    // Clip to 0-1
    let val = z * stdev + mean;
    return Math.max(0, Math.min(1, val));
}

// Core Update Loop
function update() {
    calculateMetrics();
    updateUI();
    draw();
}

function calculateMetrics() {
    let tp = 0, fp = 0, tn = 0, fn = 0;

    state.data.forEach(p => {
        const predictedPos = p.value >= state.threshold;
        if (p.label === 1) {
            if (predictedPos) tp++; else fn++;
        } else {
            if (predictedPos) fp++; else tn++;
        }
    });

    const total = state.data.length;
    const accuracy = (tp + tn) / total;
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0; // TRP
    const fpr = (fp + tn) > 0 ? fp / (fp + tn) : 0;
    const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

    state.metrics = { tp, fp, fn, tn, accuracy, precision, recall, fpr, f1 };
}

function updateUI() {
    // Threshold Handle pos
    els.handle.style.left = `${state.threshold * 100}%`;
    els.thresholdVal.textContent = state.threshold.toFixed(2);

    // Matrix
    els.tp.textContent = state.metrics.tp;
    els.fp.textContent = state.metrics.fp;
    els.fn.textContent = state.metrics.fn;
    els.tn.textContent = state.metrics.tn;

    // Main Metrics
    els.acc.textContent = state.metrics.accuracy.toFixed(2);
    els.prec.textContent = state.metrics.precision.toFixed(2);
    els.rec.textContent = state.metrics.recall.toFixed(2);
    els.fpr.textContent = state.metrics.fpr.toFixed(2);
    els.f1.textContent = state.metrics.f1.toFixed(2);

    // Education Mode Logic
    if (state.educationMode) {
        // Handle Matrix Cells
        ['tp', 'fp', 'fn', 'tn'].forEach(key => {
            if (state.revealed.has(key)) {
                els.matrixCells[key].classList.remove('masked');
            } else {
                els.matrixCells[key].classList.add('masked');
            }
        });

        // Handle Metrics Panel
        // Show metrics only if ALL 4 cells are revealed
        if (state.revealed.size === 4) {
            els.metricsPanel.classList.remove('masked');
        } else {
            els.metricsPanel.classList.add('masked');
        }
    } else {
        // Clear all masking
        ['tp', 'fp', 'fn', 'tn'].forEach(key => {
            els.matrixCells[key].classList.remove('masked');
        });
        els.metricsPanel.classList.remove('masked');
    }
}

function resizeCanvases() {
    const dpr = window.devicePixelRatio || 1;

    // Dist Canvas
    const dRect = els.distWrapper.getBoundingClientRect();
    els.distCanvas.width = dRect.width * dpr;
    els.distCanvas.height = dRect.height * dpr;
    els.distCanvas.style.width = `${dRect.width}px`;
    els.distCanvas.style.height = `${dRect.height}px`;
    els.distCanvas.getContext('2d').scale(dpr, dpr);

    // ROC Canvas
    const rRect = els.rocCanvas.parentElement.getBoundingClientRect();
    els.rocCanvas.width = rRect.width * dpr;
    els.rocCanvas.height = rRect.height * dpr;
    els.rocCanvas.style.width = `${rRect.width}px`;
    els.rocCanvas.style.height = `${rRect.height}px`;
    els.rocCanvas.getContext('2d').scale(dpr, dpr);
}

function draw() {
    drawDistribution();
    drawROC();
}

function drawDistribution() {
    const ctx = els.distCanvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = els.distCanvas.width / dpr;
    const h = els.distCanvas.height / dpr;

    ctx.clearRect(0, 0, w, h);

    // Draw background zones
    const threshX = state.threshold * w;

    // Left Zone (Predicted Negative)
    ctx.fillStyle = 'rgba(245, 158, 11, 0.1)'; // faint amber
    ctx.fillRect(0, 0, threshX, h);

    // Right Zone (Predicted Positive)
    ctx.fillStyle = 'rgba(139, 92, 246, 0.1)'; // faint violet
    ctx.fillRect(threshX, 0, w - threshX, h);

    // Draw Dots
    // We can spread them vertically randomly or build a histogram stack.
    // "Beeswarm" or stacked dots look nice.
    // Let's do a simple stacked histogram style where dots stack up.

    const buckets = 50;
    const bucketWidth = w / buckets;
    // Map of bucketIndex -> [items]
    const slots = Array(buckets).fill().map(() => []);

    state.data.forEach(p => {
        const bucketIdx = Math.min(buckets - 1, Math.floor(p.value * buckets));
        slots[bucketIdx].push(p);
    });

    const dotRadius = Math.min(6, (bucketWidth * 0.8) / 2);

    slots.forEach((items, bIdx) => {
        const cx = bIdx * bucketWidth + bucketWidth / 2;
        items.forEach((item, i) => {
            const cy = h - 10 - (i * (dotRadius * 2 + 2));

            ctx.beginPath();
            ctx.arc(cx, cy, dotRadius, 0, Math.PI * 2);

            // Color based on DATA TRUTH
            if (item.label === 1) {
                // Positive
                ctx.fillStyle = '#8b5cf6'; // violet
                // If it is on the LEFT of threshold (FN), maybe style differently?
                // The requirements say "See X on dots". 
                // Let's stick to true class colors for now, maybe add an X if misclassified.
            } else {
                ctx.fillStyle = '#f59e0b'; // amber
            }
            ctx.fill();

            // Check misclassification for visual flare
            const predictedPos = item.value >= state.threshold;
            const isMisclassified = (item.label === 1 && !predictedPos) || (item.label === 0 && predictedPos);

            if (isMisclassified) {
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();

                // Maybe draw an X
                ctx.beginPath();
                ctx.moveTo(cx - 3, cy - 3);
                ctx.lineTo(cx + 3, cy + 3);
                ctx.moveTo(cx + 3, cy - 3);
                ctx.lineTo(cx - 3, cy + 3);
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        });
    });

    // Draw Threshold Line
    ctx.beginPath();
    ctx.moveTo(threshX, 0);
    ctx.lineTo(threshX, h);
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.stroke();
}

function drawROC() {
    const ctx = els.rocCanvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = els.rocCanvas.width / dpr;
    const h = els.rocCanvas.height / dpr;
    const padding = 40;
    const availableW = w - padding * 2;
    const availableH = h - padding * 2;

    ctx.clearRect(0, 0, w, h);

    // Axes
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, h - padding);
    ctx.lineTo(w - padding, h - padding);
    ctx.strokeStyle = '#ccc';
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#666';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('FPR (1 - Specificity)', w / 2, h - 10);

    ctx.save();
    ctx.translate(15, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('TPR (Recall)', 0, 0);
    ctx.restore();

    // Calculate ROC Curve Points
    // We vary threshold from 0 to 1 and calculate TPR/FPR
    // Optimally, we sort data by value and sweep.
    const sortedData = [...state.data].sort((a, b) => b.value - a.value);

    let tp = 0;
    let fp = 0;
    const totalPos = state.data.filter(d => d.label === 1).length;
    const totalNeg = state.data.filter(d => d.label === 0).length;

    const points = [];
    points.push({ x: 0, y: 0 }); // Start at 0,0 (Threshold > 1)

    // Sweep
    // If threshold is just below item i, then item i is predicted positive.
    for (let i = 0; i < sortedData.length; i++) {
        if (sortedData[i].label === 1) tp++;
        else fp++;

        points.push({
            x: fp / totalNeg,
            y: tp / totalPos
        });
    }

    ctx.beginPath();
    ctx.moveTo(padding, h - padding); // 0,0

    points.forEach(p => {
        const px = padding + p.x * availableW;
        const py = (h - padding) - p.y * availableH;
        ctx.lineTo(px, py);
    });

    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 3;
    ctx.stroke();

    // Draw diagonal chance line
    ctx.beginPath();
    ctx.setLineDash([5, 5]);
    ctx.moveTo(padding, h - padding);
    ctx.lineTo(w - padding, padding);
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Current Point
    const currCtxX = padding + state.metrics.fpr * availableW;
    const currCtxY = (h - padding) - state.metrics.recall * availableH;

    // Drop lines
    ctx.beginPath();
    ctx.moveTo(currCtxX, h - padding);
    ctx.lineTo(currCtxX, currCtxY);
    ctx.lineTo(padding, currCtxY);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)';
    ctx.stroke();

    // Red Dot
    ctx.beginPath();
    ctx.arc(currCtxX, currCtxY, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#ef4444'; // red-500
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
}

// Start
init();
