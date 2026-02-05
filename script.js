// Constants
const DATA_POINTS = 200; // Total points
const POS_RATIO = 0.5;   // 50% positive
const CANVAS_PADDING = 20;

// State
let state = {
    data: [], // Array of { value: float, label: 0|1 }
    separation: 2.0,
    threshold: 0.5,
    metrics: {}
};

// DOM Elements
const els = {
    distCanvas: document.getElementById('dist-canvas'),
    rocCanvas: document.getElementById('roc-canvas'),
    handle: document.getElementById('threshold-handle'),
    distWrapper: document.getElementById('dist-wrapper'),
    separationInput: document.getElementById('separation'),
    separationVal: document.getElementById('separation-val'),
    thresholdVal: document.getElementById('threshold-val'),
    regenBtn: document.getElementById('regen-btn'),
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
    fpr: document.getElementById('fpr-val')
};

// Init
function init() {
    setupEventListeners();
    generateData();
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
        update();
    });

    // Regen Button
    els.regenBtn.addEventListener('click', () => {
        generateData();
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
    });
}

function updateThresholdFromMouse(e) {
    const rect = els.distWrapper.getBoundingClientRect();
    let x = e.clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    state.threshold = x / rect.width;
    update();
}

// Data Generation
function generateData() {
    state.data = [];
    const numPos = Math.floor(DATA_POINTS * POS_RATIO);
    const numNeg = DATA_POINTS - numPos;

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
    const u = 1 - Math.random();
    const v = Math.random();
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
}

function resizeCanvases() {
    const dRect = els.distWrapper.getBoundingClientRect();
    els.distCanvas.width = dRect.width;
    els.distCanvas.height = dRect.height;

    // ROC is usually fixed aspect or controlled by CSS to be square-ish
    // But let's ensure high DPI
    const rRect = els.rocCanvas.parentElement.getBoundingClientRect();
    els.rocCanvas.width = rRect.width;
    els.rocCanvas.height = rRect.height;
}

function draw() {
    drawDistribution();
    drawROC();
}

function drawDistribution() {
    const ctx = els.distCanvas.getContext('2d');
    const w = els.distCanvas.width;
    const h = els.distCanvas.height;

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
    const w = els.rocCanvas.width;
    const h = els.rocCanvas.height;
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
