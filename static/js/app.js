/**
 * ChronoPredict — Frontend Application Logic
 * Handles disease selection, form submission, results rendering,
 * risk gauge animation, model comparison, and SHAP chart.
 */

document.addEventListener("DOMContentLoaded", () => {
    initThemeToggle();
    initNavbar();
    initDiseaseSelector();
    initPredictForm();
    initModelComparison();
    initParticles();
});

// ─── Theme Toggle ─────────────────────────────────────────────────────
function initThemeToggle() {
    const toggle = document.getElementById("themeToggle");
    const icon = document.getElementById("themeIcon");
    const label = document.getElementById("themeLabel");

    // Check saved preference
    const savedTheme = localStorage.getItem("chronopredict-theme");
    if (savedTheme === "dark") {
        document.documentElement.setAttribute("data-theme", "dark");
        icon.textContent = "☀️";
        label.textContent = "Light";
    }

    toggle.addEventListener("click", () => {
        // Add transition class for smooth switch
        document.body.classList.add("theme-transitioning");

        const isDark = document.documentElement.getAttribute("data-theme") === "dark";

        if (isDark) {
            document.documentElement.removeAttribute("data-theme");
            icon.textContent = "\u{1F319}";  // moon
            label.textContent = "Dark";
            localStorage.setItem("chronopredict-theme", "light");
        } else {
            document.documentElement.setAttribute("data-theme", "dark");
            icon.textContent = "\u{2600}\u{FE0F}";  // sun
            label.textContent = "Light";
            localStorage.setItem("chronopredict-theme", "dark");
        }

        // Update particles
        updateParticleColors();

        // Remove transition class after animation
        setTimeout(() => {
            document.body.classList.remove("theme-transitioning");
        }, 500);
    });
}

function isDarkMode() {
    return document.documentElement.getAttribute("data-theme") === "dark";
}

function updateParticleColors() {
    const particles = document.querySelectorAll("#particles > div");
    particles.forEach(p => {
        const opacity = Math.random() * 0.3 + 0.05;
        if (isDarkMode()) {
            p.style.background = `rgba(139, 92, 246, ${opacity})`;
        } else {
            p.style.background = `rgba(99, 102, 241, ${opacity})`;
        }
    });
}

// ─── Navbar Scroll Effect ─────────────────────────────────────────────
function initNavbar() {
    const navbar = document.getElementById("navbar");
    const navLinks = document.querySelectorAll(".nav-link");

    window.addEventListener("scroll", () => {
        navbar.classList.toggle("scrolled", window.scrollY > 50);
    });

    // Active link tracking
    navLinks.forEach(link => {
        link.addEventListener("click", (e) => {
            navLinks.forEach(l => l.classList.remove("active"));
            link.classList.add("active");
        });
    });
}

// ─── Disease Card Selection ───────────────────────────────────────────
function initDiseaseSelector() {
    const cards = document.querySelectorAll(".disease-card");
    const hiddenInput = document.getElementById("selectedDisease");

    cards.forEach(card => {
        card.addEventListener("click", () => {
            cards.forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            hiddenInput.value = card.dataset.disease;
        });
    });
}

// ─── Prediction Form ──────────────────────────────────────────────────
function initPredictForm() {
    const form = document.getElementById("predictForm");
    const btn = document.getElementById("predictBtn");
    const resultsPanel = document.getElementById("resultsPanel");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Collect form data
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = value;
        });

        // Show loading
        btn.classList.add("loading");
        btn.disabled = true;

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });

            const result = await response.json();

            if (result.error) {
                showError(result.error);
            } else {
                renderResults(result);
            }
        } catch (err) {
            showError("Failed to connect to the server. Make sure the pipeline has been run.");
        } finally {
            btn.classList.remove("loading");
            btn.disabled = false;
        }
    });
}

// ─── Render Prediction Results ────────────────────────────────────────
function renderResults(result) {
    const panel = document.getElementById("resultsPanel");
    panel.style.display = "block";

    // Scroll to results
    setTimeout(() => {
        panel.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);

    // Model info
    document.getElementById("resultModel").textContent =
        `Model: ${result.model_used} (${result.model_accuracy}% accuracy)`;

    // Risk gauge
    animateGauge(result.risk_score, result.risk_color);

    // Risk label
    const riskLabel = document.getElementById("riskLabel");
    riskLabel.textContent = result.risk_level;
    riskLabel.style.color = result.risk_color;

    // Disease name
    document.getElementById("riskDisease").textContent = result.disease;

    // Contributing factors
    renderFactors(result.factors);
}

// ─── Animate Risk Gauge ───────────────────────────────────────────────
function animateGauge(score, color) {
    const gaugeValue = document.getElementById("gaugeValue");
    const gaugeFill = document.getElementById("gaugeFill");

    // SVG arc math: total arc length ≈ 251
    const totalArc = 251;
    const offset = totalArc - (score / 100) * totalArc;

    // Create gradient dynamically
    const svg = document.querySelector(".gauge");
    let defs = svg.querySelector("defs");
    if (!defs) {
        defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
        svg.prepend(defs);
    }
    defs.innerHTML = `
        <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#10b981" />
            <stop offset="50%" stop-color="#f59e0b" />
            <stop offset="100%" stop-color="#ef4444" />
        </linearGradient>
    `;

    // Animate fill
    gaugeFill.style.stroke = `url(#gaugeGradient)`;
    gaugeFill.style.strokeDashoffset = totalArc;

    requestAnimationFrame(() => {
        gaugeFill.style.strokeDashoffset = offset;
    });

    // Animate number
    let current = 0;
    const step = score / 60;
    const interval = setInterval(() => {
        current += step;
        if (current >= score) {
            current = score;
            clearInterval(interval);
        }
        gaugeValue.textContent = `${Math.round(current)}%`;
    }, 16);
}

// ─── Render Contributing Factors ──────────────────────────────────────
function renderFactors(factors) {
    const list = document.getElementById("factorsList");
    list.innerHTML = "";

    if (!factors || factors.length === 0) {
        list.innerHTML = `<p style="color: var(--text-muted); font-size: 0.9rem;">
            No SHAP explanation available for this model.</p>`;
        return;
    }

    factors.forEach((factor, i) => {
        const isPositive = factor.impact > 0;
        const item = document.createElement("div");
        item.className = "factor-item";
        item.style.animationDelay = `${i * 0.1}s`;

        item.innerHTML = `
            <div class="factor-bar ${isPositive ? 'positive' : 'negative'}"></div>
            <div class="factor-info">
                <div class="factor-name">${factor.name}</div>
                <div class="factor-impact">
                    ${isPositive ? '↑ Increases' : '↓ Decreases'} risk
                    (impact: ${Math.abs(factor.impact).toFixed(3)})
                </div>
            </div>
        `;

        list.appendChild(item);
    });
}

// ─── Model Comparison ─────────────────────────────────────────────────
function initModelComparison() {
    const tabs = document.querySelectorAll(".model-tab");

    // Load data on first tab click or page load
    let modelsData = null;

    async function loadModels() {
        try {
            const response = await fetch("/models");
            modelsData = await response.json();

            // Show first disease
            const firstDisease = tabs[0]?.dataset.disease;
            if (firstDisease && modelsData[firstDisease]) {
                renderModelTable(firstDisease, modelsData[firstDisease]);
            } else {
                document.getElementById("modelLoading").innerHTML =
                    `<p style="color: var(--text-muted);">No trained models yet. Run <code>python run_pipeline.py</code> first.</p>`;
            }
        } catch {
            document.getElementById("modelLoading").innerHTML =
                `<p style="color: var(--text-muted);">Could not load model data.</p>`;
        }
    }

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");

            const disease = tab.dataset.disease;
            if (modelsData && modelsData[disease]) {
                renderModelTable(disease, modelsData[disease]);
            }
        });
    });

    loadModels();
}

function renderModelTable(disease, data) {
    const loading = document.getElementById("modelLoading");
    const wrap = document.getElementById("modelTableWrap");
    const body = document.getElementById("modelTableBody");
    const badge = document.getElementById("bestModelBadge");
    const chartWrap = document.getElementById("featureChartWrap");

    loading.style.display = "none";
    wrap.style.display = "block";

    body.innerHTML = "";

    const metrics = data.metrics;
    const bestModel = data.best_model;

    for (const [modelName, m] of Object.entries(metrics)) {
        const isBest = modelName === bestModel;
        const row = document.createElement("tr");
        if (isBest) row.className = "best-row";

        row.innerHTML = `
            <td>${isBest ? '⭐ ' : ''}${modelName}</td>
            <td class="metric-value ${getMetricClass(m.accuracy)}">${(m.accuracy * 100).toFixed(1)}%</td>
            <td class="metric-value ${getMetricClass(m.precision)}">${(m.precision * 100).toFixed(1)}%</td>
            <td class="metric-value ${getMetricClass(m.recall)}">${(m.recall * 100).toFixed(1)}%</td>
            <td class="metric-value ${getMetricClass(m.f1)}">${(m.f1 * 100).toFixed(1)}%</td>
            <td class="metric-value ${getMetricClass(m.roc_auc)}">${(m.roc_auc * 100).toFixed(1)}%</td>
        `;

        body.appendChild(row);
    }

    badge.innerHTML = `🏆 Best Model: <strong>${bestModel}</strong> for ${data.display_name}`;

    // Feature importance chart
    if (data.feature_importance) {
        chartWrap.style.display = "block";
        renderFeatureChart(data.feature_importance);
    } else {
        chartWrap.style.display = "none";
    }
}

function getMetricClass(value) {
    if (value >= 0.85) return "metric-good";
    if (value >= 0.70) return "metric-ok";
    return "metric-poor";
}

// ─── Feature Importance Chart (Canvas) ────────────────────────────────
function renderFeatureChart(importance) {
    const canvas = document.getElementById("featureChart");
    const ctx = canvas.getContext("2d");

    // High-DPI support
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    ctx.clearRect(0, 0, width, height);

    const entries = Object.entries(importance).slice(0, 10);
    const maxVal = Math.max(...entries.map(e => e[1]));

    const barHeight = 24;
    const gap = 10;
    const labelWidth = 180;
    const chartLeft = labelWidth + 10;
    const chartWidth = width - chartLeft - 40;
    const startY = 20;

    entries.forEach(([name, value], i) => {
        const y = startY + i * (barHeight + gap);
        const barWidth = (value / maxVal) * chartWidth;

        // Gradient bar
        const gradient = ctx.createLinearGradient(chartLeft, 0, chartLeft + barWidth, 0);
        gradient.addColorStop(0, "#6366f1");
        gradient.addColorStop(1, "#a855f7");

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.roundRect(chartLeft, y, barWidth, barHeight, 4);
        ctx.fill();

        // Label — use theme-aware color
        ctx.fillStyle = isDarkMode() ? "#94a3b8" : "#475569";
        ctx.font = "12px Inter, sans-serif";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";

        // Truncate long names
        let displayName = name.length > 22 ? name.substring(0, 20) + "..." : name;
        ctx.fillText(displayName, labelWidth, y + barHeight / 2);

        // Value — use theme-aware color
        ctx.fillStyle = isDarkMode() ? "#e2e8f0" : "#1e293b";
        ctx.textAlign = "left";
        ctx.fillText(value.toFixed(4), chartLeft + barWidth + 8, y + barHeight / 2);
    });
}

// ─── Floating Particles ───────────────────────────────────────────────
function initParticles() {
    const container = document.getElementById("particles");
    const count = 30;

    for (let i = 0; i < count; i++) {
        const particle = document.createElement("div");
        const size = Math.random() * 3 + 1;

        Object.assign(particle.style, {
            position: "absolute",
            width: `${size}px`,
            height: `${size}px`,
            borderRadius: "50%",
            background: isDarkMode()
                ? `rgba(139, 92, 246, ${Math.random() * 0.3 + 0.05})`
                : `rgba(99, 102, 241, ${Math.random() * 0.3 + 0.05})`,
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            animation: `float ${Math.random() * 20 + 10}s ease-in-out infinite`,
            animationDelay: `${Math.random() * 10}s`,
        });

        container.appendChild(particle);
    }

    // Add float animation
    const style = document.createElement("style");
    style.textContent = `
        @keyframes float {
            0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.5; }
            25% { transform: translate(${rand()}px, ${rand()}px) scale(1.2); opacity: 0.8; }
            50% { transform: translate(${rand()}px, ${rand()}px) scale(0.8); opacity: 0.3; }
            75% { transform: translate(${rand()}px, ${rand()}px) scale(1.1); opacity: 0.7; }
        }
    `;
    document.head.appendChild(style);
}

function rand() {
    return (Math.random() - 0.5) * 100;
}

// ─── Error Display ────────────────────────────────────────────────────
function showError(message) {
    const panel = document.getElementById("resultsPanel");
    panel.style.display = "block";
    panel.innerHTML = `
        <div class="result-card" style="text-align: center; padding: 40px;">
            <div style="font-size: 2rem; margin-bottom: 16px;">⚠️</div>
            <h3 style="margin-bottom: 12px;">Prediction Error</h3>
            <p style="color: var(--text-secondary);">${message}</p>
            <p style="color: var(--text-muted); margin-top: 12px; font-size: 0.85rem;">
                Make sure you've run <code>python run_pipeline.py</code> to train the models first.
            </p>
        </div>
    `;
    panel.scrollIntoView({ behavior: "smooth" });
}
