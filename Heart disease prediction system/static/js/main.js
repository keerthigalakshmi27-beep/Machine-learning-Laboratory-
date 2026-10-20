/**
 * CardioVision AI - Frontend Application Controller
 * Handles 5-Stage ML Pipeline, Live In-App Code Studio, SVG Visualizations,
 * Real-time inference, and Medical Report generation.
 */

// Global Application State
const AppState = {
    currentImage: "/static/samples/normal_sinus_rhythm.png",
    stages: {},
    features: null,
    featureVector: null,
    evaluation: null,
    currentDiagnosis: null,
    codeTemplates: {},
    activeSampleId: "sample-1",
    isProcessing: false
};

// Clinical Feature Descriptions
const FEATURE_DESCRIPTIONS = {
    "mean_intensity": "Average optical density / cardiac tissue opacity",
    "std_dev": "Pixel intensity dispersion & contrast variance",
    "skewness": "Asymmetry in cardiac chamber grayscale distribution",
    "kurtosis": "Sharpness of waveform peaks and boundary gradients",
    "entropy": "Randomness / structural disorder index of cardiac tissue",
    "glcm_contrast": "Spatial frequency & local texture intensity variations",
    "glcm_dissimilarity": "Linear distance between adjacent gray-level pairs",
    "glcm_homogeneity": "Uniformity of myocardial tissue & baseline consistency",
    "glcm_energy": "Angular Second Moment (orderliness of texture pattern)",
    "glcm_correlation": "Linear gray-tone dependency across neighbor pixels",
    "peak_count": "Detected electrical depolarization spikes (R-waves)",
    "heart_rate_bpm": "Estimated Heart Rate calculated from waveform frequency",
    "cardiothoracic_ratio": "Cardiac silhouette area / total thoracic width",
    "st_elevation_index": "Deviation from isoelectric baseline (myocardial injury)"
};

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
    initTabNavigation();
    initSampleSelector();
    initDropzone();
    initPreprocessorControls();
    initFeatureControls();
    initTrainingControls();
    initDiagnosisControls();
    initCodeStudio();
    initGuideCopyButtons();

    // Trigger initial automated load
    loadSampleImage("normal_sinus_rhythm.png");
    loadInitialModelEvaluation();
});

/* ==========================================================================
   Tab Navigation
   ========================================================================== */
function initTabNavigation() {
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabPanels = document.querySelectorAll(".tab-panel");

    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const targetId = tab.dataset.tab;
            navTabs.forEach(t => t.classList.remove("active"));
            tabPanels.forEach(p => p.classList.remove("active"));

            tab.classList.add("active");
            const targetPanel = document.getElementById(targetId);
            if (targetPanel) targetPanel.classList.add("active");
        });
    });

    // Subtabs within cards
    const subtabs = document.querySelectorAll(".subtab");
    subtabs.forEach(st => {
        st.addEventListener("click", () => {
            const container = st.closest(".card-body");
            const targetId = st.dataset.subtab;

            container.querySelectorAll(".subtab").forEach(t => t.classList.remove("active"));
            container.querySelectorAll(".subtab-panel").forEach(p => p.classList.remove("active"));

            st.classList.add("active");
            const panel = document.getElementById(targetId);
            if (panel) panel.classList.add("active");
        });
    });
}

/* ==========================================================================
   Stage 1: Input Image & Preset Selection
   ========================================================================== */
function initSampleSelector() {
    const samplePills = document.querySelectorAll(".sample-pill");
    samplePills.forEach(pill => {
        pill.addEventListener("click", () => {
            samplePills.forEach(p => p.classList.remove("active"));
            pill.classList.add("active");

            // Update patient metadata banner
            const name = pill.textContent.trim();
            const age = pill.dataset.age;
            const gender = pill.dataset.gender;
            const bp = pill.dataset.bp;
            const hr = pill.dataset.hr;

            document.getElementById("pMetaName").textContent = name;
            document.getElementById("pMetaDemographics").textContent = `${age}y / ${gender}`;
            document.getElementById("pMetaBp").textContent = bp;
            document.getElementById("pMetaHr").textContent = hr;

            // Hub patient details
            document.getElementById("hubPatientName").textContent = `Patient: ${name}`;
            document.getElementById("hubAge").textContent = age;
            document.getElementById("hubGender").textContent = gender;
            document.getElementById("hubBp").textContent = bp;
            document.getElementById("hubHr").textContent = hr;

            loadSampleImage(pill.dataset.file);
        });
    });

    // Stepper node click
    document.querySelectorAll(".step-node").forEach(node => {
        node.addEventListener("click", () => {
            const step = node.dataset.step;
            const card = document.getElementById(`card-stage-${step}`);
            if (card) {
                card.scrollIntoView({ behavior: "smooth", block: "center" });
            }
        });
    });

    // Full Pipeline Auto-Run Button
    document.getElementById("btnAutoRunPipeline").addEventListener("click", runFullPipelineAuto);
    document.getElementById("btnResetPipeline").addEventListener("click", () => {
        loadSampleImage("normal_sinus_rhythm.png");
    });
}

function loadSampleImage(filename) {
    fetch("/api/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_file: filename })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            AppState.currentImage = data.image_data;
            updatePreviewImage(data.image_data, "Raw Input Image");
            setStageBadge(1, "Ready", "badge-ready");
            setStageConnector(1, true);
        }
    })
    .catch(err => console.error("Error loading sample image:", err));
}

function initDropzone() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");

    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
            uploadFile(e.target.files[0]);
        }
    });
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    setStageBadge(1, "Uploading...", "badge-pending");

    fetch("/api/upload", {
        method: "POST",
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            AppState.currentImage = data.image_data;
            updatePreviewImage(data.image_data, `Uploaded: ${file.name}`);
            setStageBadge(1, "Loaded", "badge-ready");
            setStageConnector(1, true);

            // Update patient metadata for uploaded file
            document.getElementById("pMetaName").textContent = `Upload: ${file.name}`;
            document.getElementById("pMetaDemographics").textContent = "Clinical Custom File";
        } else {
            alert("Upload failed: " + data.error);
            setStageBadge(1, "Error", "badge-pending");
        }
    })
    .catch(err => {
        console.error("Upload error:", err);
        setStageBadge(1, "Error", "badge-pending");
    });
}

/* ==========================================================================
   Stage 2: Preprocessor Controls
   ========================================================================== */
function initPreprocessorControls() {
    const sContrast = document.getElementById("sliderContrast");
    const sNoise = document.getElementById("sliderNoise");
    const sThresh = document.getElementById("sliderThreshold");

    sContrast.addEventListener("input", (e) => document.getElementById("valContrast").textContent = `${e.target.value}x`);
    sNoise.addEventListener("input", (e) => document.getElementById("valNoise").textContent = `${e.target.value} px`);
    sThresh.addEventListener("input", (e) => document.getElementById("valThreshold").textContent = e.target.value);

    document.getElementById("btnRunPreprocess").addEventListener("click", () => executePreprocessing());

    // Preprocessor Gallery Tabs
    const preTabs = document.querySelectorAll(".pre-tab");
    preTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            preTabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");

            const stageKey = tab.dataset.stage;
            if (AppState.stages && AppState.stages[stageKey]) {
                updatePreviewImage(AppState.stages[stageKey], tab.textContent);
            }
        });
    });
}

function executePreprocessing(callback) {
    if (!AppState.currentImage) return;

    setStageBadge(2, "Processing...", "badge-pending");

    const payload = {
        image_data: AppState.currentImage,
        contrast: parseFloat(document.getElementById("sliderContrast").value),
        noise_radius: parseFloat(document.getElementById("sliderNoise").value),
        threshold: parseInt(document.getElementById("sliderThreshold").value)
    };

    return fetch("/api/preprocess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            AppState.stages = data.stages;
            setStageBadge(2, "Complete", "badge-ready");
            setStageConnector(2, true);

            // Active stage preview tab
            const activeTab = document.querySelector(".pre-tab.active");
            const stageKey = activeTab ? activeTab.dataset.stage : "original";
            updatePreviewImage(AppState.stages[stageKey] || AppState.stages.original, activeTab ? activeTab.textContent : "Stage Preview");

            if (callback) callback();
            return data;
        } else {
            alert("Preprocessing error: " + data.error);
        }
    })
    .catch(err => console.error("Preprocessing error:", err));
}

function updatePreviewImage(src, tagText) {
    const preview = document.getElementById("previewImage");
    const tag = document.getElementById("currentStageTag");
    preview.src = src;
    if (tagText) tag.textContent = tagText;
}

/* ==========================================================================
   Stage 3: Feature Extraction
   ========================================================================== */
function initFeatureControls() {
    document.getElementById("btnRunFeatures").addEventListener("click", () => executeFeatureExtraction());
}

function executeFeatureExtraction(callback) {
    if (!AppState.currentImage) return;

    return fetch("/api/features", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_data: AppState.stages.grayscale || AppState.currentImage })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            AppState.features = data.features;
            AppState.featureVector = data.feature_vector;

            renderFeatureSummary(data.features, data.summary);
            renderGlcmHeatmap(data.glcm_matrix_preview);
            renderFeatureVectorTable(data.feature_names, data.feature_vector);

            setStageConnector(3, true);

            // Update BPM in header
            if (data.features.heart_rate_bpm) {
                document.getElementById("liveBpm").textContent = `${Math.round(data.features.heart_rate_bpm)} BPM`;
            }

            if (callback) callback();
            return data;
        } else {
            alert("Feature extraction error: " + data.error);
        }
    })
    .catch(err => console.error("Feature extraction error:", err));
}

function renderFeatureSummary(features, summary) {
    document.getElementById("fHomogeneity").textContent = `${Math.round(features.glcm_homogeneity * 100)}%`;
    document.getElementById("fContrast").textContent = features.glcm_contrast.toFixed(2);
    document.getElementById("fEntropy").textContent = features.entropy.toFixed(2);
    document.getElementById("fBpm").textContent = `${Math.round(features.heart_rate_bpm)} BPM`;

    // Clinical Hub CTR
    document.getElementById("hubCtr").textContent = `${features.cardiothoracic_ratio} (${features.cardiothoracic_ratio > 0.5 ? "Enlarged" : "Normal"})`;
    document.getElementById("hubSt").textContent = `${features.st_elevation_index.toFixed(3)} mV`;
}

function renderGlcmHeatmap(matrix) {
    const grid = document.getElementById("glcmGrid");
    grid.innerHTML = "";

    if (!matrix || !matrix.length) return;

    for (let r = 0; r < matrix.length; r++) {
        for (let c = 0; c < matrix[r].length; c++) {
            const val = matrix[r][c]; // 0 - 100
            const cell = document.createElement("div");
            cell.className = "glcm-cell";

            // Heatmap color interpolation
            const intensity = Math.min(1, val / 25);
            cell.style.background = `rgba(0, 240, 255, ${0.1 + intensity * 0.9})`;
            cell.title = `GLCM (${r},${c}): ${val.toFixed(2)}%`;
            cell.textContent = val > 1 ? val.toFixed(0) : "";
            grid.appendChild(cell);
        }
    }
}

function renderFeatureVectorTable(names, values) {
    const tbody = document.querySelector("#featureVectorTable tbody");
    tbody.innerHTML = "";

    names.forEach((name, idx) => {
        const val = values[idx];
        const row = document.createElement("tr");

        const tdName = document.createElement("td");
        tdName.textContent = name;

        const tdVal = document.createElement("td");
        tdVal.textContent = val;

        const tdDesc = document.createElement("td");
        tdDesc.textContent = FEATURE_DESCRIPTIONS[name] || "Clinical indicator";

        row.appendChild(tdName);
        row.appendChild(tdVal);
        row.appendChild(tdDesc);
        tbody.appendChild(row);
    });
}

/* ==========================================================================
   Stage 4: ML Training Controls
   ========================================================================== */
function initTrainingControls() {
    const selectAlgorithm = document.getElementById("selectAlgorithm");
    selectAlgorithm.addEventListener("change", updateHyperparamInputs);

    document.getElementById("btnTrainModel").addEventListener("click", () => executeTraining());
}

function updateHyperparamInputs() {
    const algo = document.getElementById("selectAlgorithm").value;
    const p1 = document.getElementById("hpParam1");
    const p2 = document.getElementById("hpParam2");

    if (algo.includes("Random Forest")) {
        p1.innerHTML = `<label>Trees (n_estimators):</label><input type="number" id="inputParam1" value="100" min="10" max="500">`;
        p2.innerHTML = `<label>Max Depth:</label><input type="number" id="inputParam2" value="10" min="2" max="30">`;
    } else if (algo.includes("SVM")) {
        p1.innerHTML = `<label>Penalty (C):</label><input type="number" id="inputParam1" value="1.0" step="0.1" min="0.1" max="100">`;
        p2.innerHTML = `<label>Kernel:</label><input type="text" id="inputParam2" value="rbf" readonly style="width:60px">`;
    } else if (algo.includes("KNN")) {
        p1.innerHTML = `<label>Neighbors (k):</label><input type="number" id="inputParam1" value="5" min="1" max="25">`;
        p2.innerHTML = `<label>Weights:</label><input type="text" id="inputParam2" value="uniform" readonly style="width:75px">`;
    } else if (algo.includes("Decision Tree")) {
        p1.innerHTML = `<label>Max Depth:</label><input type="number" id="inputParam1" value="8" min="2" max="25">`;
        p2.innerHTML = `<label>Criterion:</label><input type="text" id="inputParam2" value="gini" readonly style="width:60px">`;
    } else {
        p1.innerHTML = `<label>Regularization (C):</label><input type="number" id="inputParam1" value="1.0" step="0.2" min="0.1">`;
        p2.innerHTML = `<label>Max Iterations:</label><input type="number" id="inputParam2" value="1000" min="100">`;
    }
}

function executeTraining(callback) {
    const algo = document.getElementById("selectAlgorithm").value;
    const split = parseFloat(document.getElementById("selectSplit").value);
    const p1Val = document.getElementById("inputParam1")?.value || 100;
    const p2Val = document.getElementById("inputParam2")?.value || 10;

    const progressWrap = document.getElementById("trainProgressWrap");
    const progressFill = document.getElementById("trainProgressFill");
    const progressText = document.getElementById("trainProgressText");

    progressWrap.style.display = "block";
    progressFill.style.width = "25%";
    progressText.textContent = `Optimizing ${algo} hyperparameters...`;

    const payload = {
        algorithm: algo,
        test_size: split,
        params: {
            n_estimators: p1Val,
            max_depth: p2Val,
            C: p1Val,
            n_neighbors: p1Val
        }
    };

    setTimeout(() => { progressFill.style.width = "65%"; progressText.textContent = "Fitting classifier on 800+ cardiac samples..."; }, 250);

    return fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(data => {
        progressFill.style.width = "100%";
        progressText.textContent = `Training complete! Model accuracy: ${data.metrics.accuracy}%`;

        setTimeout(() => { progressWrap.style.display = "none"; }, 1200);

        if (data.success) {
            AppState.evaluation = data.metrics;
            updateMetricsDisplay(data.metrics);
            setStageBadge(4, "Trained", "badge-ready");
            setStageConnector(4, true);

            if (callback) callback();
            return data;
        } else {
            alert("Training error: " + data.error);
        }
    })
    .catch(err => console.error("Training error:", err));
}

function loadInitialModelEvaluation() {
    // Initial fetch to populate default metrics
    fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ algorithm: "Random Forest" })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            AppState.evaluation = data.metrics;
            updateMetricsDisplay(data.metrics);
        }
    })
    .catch(err => console.log("Evaluation preload:", err));
}

function updateMetricsDisplay(metrics) {
    document.getElementById("evalAcc").textContent = `${metrics.accuracy}%`;
    document.getElementById("evalPrec").textContent = `${metrics.precision}%`;
    document.getElementById("evalRec").textContent = `${metrics.recall}%`;
    document.getElementById("evalF1").textContent = `${metrics.f1_score}%`;

    // Render Confusion Matrix
    renderConfusionMatrix(metrics.confusion_matrix);

    // Render ROC Curves
    renderRocCurves(metrics.roc_curves);

    // Render Classification Report Table
    renderClassificationReport(metrics.per_class);
}

/* ==========================================================================
   Stage 5: Patient Diagnosis & Performance Evaluation
   ========================================================================== */
function initDiagnosisControls() {
    document.getElementById("btnPredictPatient").addEventListener("click", () => executeDiagnosis());

    // Print / Modal controls
    const modal = document.getElementById("reportModal");
    document.getElementById("btnPrintReport").addEventListener("click", () => {
        populatePrintableReport();
        modal.style.display = "flex";
    });
    document.getElementById("btnCloseModal").addEventListener("click", () => modal.style.display = "none");
    document.getElementById("btnDismissModal").addEventListener("click", () => modal.style.display = "none");
}

function executeDiagnosis() {
    if (!AppState.featureVector) {
        // Run feature extraction first
        executeFeatureExtraction(() => executeDiagnosis());
        return;
    }

    return fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feature_vector: AppState.featureVector })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            AppState.currentDiagnosis = data.diagnosis;
            renderDiagnosisHero(data.diagnosis);
            setStageConnector(5, true);

            // Scroll to card 5
            document.getElementById("card-stage-5").scrollIntoView({ behavior: "smooth", block: "nearest" });
            return data;
        } else {
            alert("Diagnosis error: " + data.error);
        }
    })
    .catch(err => console.error("Diagnosis error:", err));
}

function renderDiagnosisHero(diag) {
    const fillCircle = document.getElementById("gaugeFillCircle");
    const confText = document.getElementById("diagConfidence");
    const badge = document.getElementById("diagRiskBadge");
    const title = document.getElementById("diagPredictionTitle");
    const summary = document.getElementById("diagSummary");
    const rec = document.getElementById("diagRecommendation");
    const modelTag = document.getElementById("diagModelTag");

    // Animate circular gauge
    const radius = 50;
    const circumference = 2 * Math.PI * radius; // ~314.16
    const offset = circumference - (diag.confidence / 100) * circumference;

    fillCircle.style.strokeDashoffset = offset;
    fillCircle.style.stroke = diag.risk_color;

    confText.textContent = `${diag.confidence}%`;
    title.textContent = diag.prediction;
    summary.textContent = diag.clinical_summary;
    rec.textContent = diag.recommendation;
    modelTag.textContent = `Model: ${diag.model_used}`;

    // Risk badge styling
    badge.textContent = diag.risk_level.toUpperCase();
    badge.className = "risk-badge";
    if (diag.risk_level.includes("Low")) badge.classList.add("badge-green");
    else if (diag.risk_level.includes("Critical")) badge.classList.add("badge-crimson");
    else if (diag.risk_level.includes("Moderate")) badge.classList.add("badge-amber");
    else badge.classList.add("badge-purple");

    // Hub alerts
    document.getElementById("hubFindingTitle").textContent = diag.prediction;
    document.getElementById("hubFindingText").textContent = diag.clinical_summary;

    // Render probability distribution bars
    renderProbabilityBars(diag.probabilities);
}

function renderProbabilityBars(probs) {
    const container = document.getElementById("probBarsList");
    container.innerHTML = "";

    for (const [className, percent] of Object.entries(probs)) {
        const row = document.createElement("div");
        row.className = "prob-row";

        row.innerHTML = `
            <span class="prob-class-name" title="${className}">${className}</span>
            <div class="prob-track">
                <div class="prob-bar-fill" style="width: ${percent}%;"></div>
            </div>
            <span class="prob-num">${percent}%</span>
        `;

        container.appendChild(row);
    }
}

/* ==========================================================================
   Automated Full 5-Stage Pipeline Execution
   ========================================================================== */
function runFullPipelineAuto() {
    const btn = document.getElementById("btnAutoRunPipeline");
    btn.disabled = true;
    btn.innerHTML = `<span class="btn-icon">⏳</span> Running Pipeline...`;

    // Step 1: Preprocess
    executePreprocessing(() => {
        // Step 2: Extract Features
        setTimeout(() => {
            executeFeatureExtraction(() => {
                // Step 3: Run Diagnosis
                setTimeout(() => {
                    executeDiagnosis();
                    btn.disabled = false;
                    btn.innerHTML = `<span class="btn-icon">⚡</span> Run Complete 5-Stage Pipeline`;
                }, 300);
            });
        }, 300);
    });
}

function setStageBadge(step, text, className) {
    const badge = document.getElementById(`stage${step}-badge`);
    if (badge) {
        badge.textContent = text;
        badge.className = `card-status-badge ${className}`;
    }
}

function setStageConnector(step, active) {
    const conn = document.getElementById(`conn-${step}`);
    if (conn && active) conn.classList.add("active");

    const node = document.querySelector(`.step-node[data-step="${step}"]`);
    if (node && active) node.classList.add("completed");
}

/* ==========================================================================
   Visualizations: Confusion Matrix & ROC Curves
   ========================================================================== */
function renderConfusionMatrix(cmData) {
    if (!cmData || !cmData.matrix) return;

    const wrap = document.getElementById("cmTableWrap");
    const matrix = cmData.matrix;
    const labels = cmData.labels;

    let html = `<table class="cm-table"><thead><tr><th>Actual \\ Predicted</th>`;
    labels.forEach(lbl => html += `<th>${lbl}</th>`);
    html += `</tr></thead><tbody>`;

    for (let r = 0; r < matrix.length; r++) {
        html += `<tr><th>${labels[r]}</th>`;
        for (let c = 0; c < matrix[r].length; c++) {
            const count = matrix[r][c];
            const isDiagonal = (r === c);
            const bgColor = isDiagonal ? "rgba(0, 255, 157, 0.25)" : (count > 0 ? "rgba(255, 51, 102, 0.25)" : "rgba(255, 255, 255, 0.03)");
            html += `<td class="cm-cell" style="background:${bgColor}" title="True: ${labels[r]}, Pred: ${labels[c]}">${count}</td>`;
        }
        html += `</tr>`;
    }

    html += `</tbody></table>`;
    wrap.innerHTML = html;
}

function renderRocCurves(rocData) {
    if (!rocData || !rocData.length) return;

    const svg = document.getElementById("rocSvgChart");
    const legend = document.getElementById("rocLegend");
    svg.innerHTML = "";
    legend.innerHTML = "";

    const colors = ["#00f0ff", "#00ff9d", "#ff3366", "#ffb703", "#a855f7"];
    const w = 400, h = 300;
    const pad = 40;

    // Draw axes
    svg.innerHTML += `
        <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h - pad}" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
        <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${pad}" stroke="rgba(255,255,255,0.15)" stroke-dasharray="4" stroke-width="1"/>
        <text x="${w / 2}" y="${h - 10}" fill="#94a3b8" font-size="11" text-anchor="middle">False Positive Rate (1 - Specificity)</text>
        <text x="15" y="${h / 2}" fill="#94a3b8" font-size="11" text-anchor="middle" transform="rotate(-90, 15, ${h / 2})">True Positive Rate (Sensitivity)</text>
    `;

    rocData.forEach((curve, idx) => {
        const color = colors[idx % colors.length];
        const points = curve.points;
        let d = "";

        points.forEach((pt, pIdx) => {
            const svgX = pad + pt.x * (w - 2 * pad);
            const svgY = (h - pad) - pt.y * (h - 2 * pad);
            if (pIdx === 0) d += `M ${svgX} ${svgY} `;
            else d += `L ${svgX} ${svgY} `;
        });

        svg.innerHTML += `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.5" />`;

        // Legend item
        const legItem = document.createElement("div");
        legItem.className = "roc-legend-item";
        legItem.innerHTML = `<span class="legend-color-dot" style="background:${color}"></span> ${curve.class_name} (AUC: ${curve.auc})`;
        legend.appendChild(legItem);
    });
}

function renderClassificationReport(perClass) {
    if (!perClass) return;

    const tbody = document.querySelector("#classificationReportTable tbody");
    tbody.innerHTML = "";

    perClass.forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong>${row.class}</strong></td>
            <td>${row.precision}%</td>
            <td>${row.recall}%</td>
            <td style="color:var(--cyan-primary); font-weight:700;">${row.f1_score}%</td>
            <td>${row.support}</td>
        `;
        tbody.appendChild(tr);
    });
}

/* ==========================================================================
   Tab 4: Interactive Code Studio & Terminal Runner
   ========================================================================== */
function initCodeStudio() {
    const fileSelector = document.getElementById("codeFileSelector");
    const textarea = document.getElementById("codeTextarea");
    const lineNumbers = document.getElementById("codeLineNumbers");

    // Fetch code templates from backend
    fetch("/api/code_templates")
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                AppState.codeTemplates = data.templates;
                // Load default selected
                if (AppState.codeTemplates["train_model.py"]) {
                    textarea.value = AppState.codeTemplates["train_model.py"];
                    updateLineNumbers();
                }
            }
        })
        .catch(err => console.error("Templates fetch error:", err));

    fileSelector.addEventListener("change", (e) => {
        const val = e.target.value;
        if (val === "custom") {
            textarea.value = `# Write custom Python 3 code for testing:\nimport sys\nfrom feature_extraction import FeatureExtractor\n\nprint("Executing custom diagnostic script...")\nprint("Python version:", sys.version)\n`;
        } else if (AppState.codeTemplates[val]) {
            textarea.value = AppState.codeTemplates[val];
        }
        updateLineNumbers();
    });

    textarea.addEventListener("input", updateLineNumbers);
    textarea.addEventListener("scroll", () => {
        lineNumbers.scrollTop = textarea.scrollTop;
    });

    // Run Code action
    document.getElementById("btnRunCode").addEventListener("click", runCodeInStudio);
    document.getElementById("btnResetCode").addEventListener("click", () => {
        const val = fileSelector.value;
        if (AppState.codeTemplates[val]) {
            textarea.value = AppState.codeTemplates[val];
            updateLineNumbers();
        }
    });

    document.getElementById("btnCopyCode").addEventListener("click", () => {
        navigator.clipboard.writeText(textarea.value);
        alert("Code copied to clipboard!");
    });

    document.getElementById("btnClearTerminal").addEventListener("click", () => {
        document.getElementById("terminalOutput").textContent = "[Terminal console cleared]";
    });
}

function updateLineNumbers() {
    const textarea = document.getElementById("codeTextarea");
    const lineNumbers = document.getElementById("codeLineNumbers");
    const lineCount = textarea.value.split("\n").length;
    let nums = "";
    for (let i = 1; i <= lineCount; i++) {
        nums += i + "\n";
    }
    lineNumbers.textContent = nums;
}

function runCodeInStudio() {
    const code = document.getElementById("codeTextarea").value;
    const terminal = document.getElementById("terminalOutput");
    const statusBadge = document.getElementById("termStatusBadge");
    const execTime = document.getElementById("termExecTime");
    const btn = document.getElementById("btnRunCode");

    btn.disabled = true;
    statusBadge.textContent = "Running...";
    statusBadge.className = "term-badge badge-pending";
    terminal.textContent = "[Executing code with Python 3.14...]\n";

    fetch("/api/run_code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: code })
    })
    .then(r => r.json())
    .then(data => {
        btn.disabled = false;
        execTime.textContent = `${data.execution_time_ms} ms`;

        if (data.success) {
            statusBadge.textContent = "Success (0)";
            statusBadge.className = "term-badge badge-ready";
        } else {
            statusBadge.textContent = `Exit (${data.returncode})`;
            statusBadge.className = "term-badge badge-crimson";
        }

        terminal.textContent = data.output;
        terminal.scrollTop = terminal.scrollHeight;
    })
    .catch(err => {
        btn.disabled = false;
        statusBadge.textContent = "Error";
        terminal.textContent = "Execution request failed: " + err;
    });
}

/* ==========================================================================
   Tab 5: Compile & Run Guide Helpers
   ========================================================================== */
function initGuideCopyButtons() {
    document.querySelectorAll(".btn-copy-code").forEach(btn => {
        btn.addEventListener("click", () => {
            const text = btn.dataset.copy;
            navigator.clipboard.writeText(text);
            const orig = btn.textContent;
            btn.textContent = "Copied!";
            setTimeout(() => { btn.textContent = orig; }, 1500);
        });
    });
}

/* ==========================================================================
   Printable Medical Report Generation
   ========================================================================== */
function populatePrintableReport() {
    const dateStr = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
    document.getElementById("reportDate").textContent = `Date: ${dateStr}`;

    const pName = document.getElementById("pMetaName").textContent;
    document.getElementById("repPatientName").textContent = pName;
    document.getElementById("repDemographics").textContent = document.getElementById("pMetaDemographics").textContent;
    document.getElementById("repBp").textContent = document.getElementById("pMetaBp").textContent;

    if (AppState.currentDiagnosis) {
        document.getElementById("repFindingTitle").textContent = AppState.currentDiagnosis.prediction;
        document.getElementById("repConfidence").textContent = `${AppState.currentDiagnosis.confidence}%`;
        document.getElementById("repModel").textContent = AppState.currentDiagnosis.model_used;
        document.getElementById("repRisk").textContent = AppState.currentDiagnosis.risk_level;
        document.getElementById("repRecommendation").textContent = AppState.currentDiagnosis.recommendation;
    }

    if (AppState.features) {
        const grid = document.getElementById("repFeaturesGrid");
        grid.innerHTML = "";
        const keys = ["mean_intensity", "glcm_contrast", "glcm_homogeneity", "entropy", "heart_rate_bpm", "cardiothoracic_ratio"];
        keys.forEach(k => {
            const item = document.createElement("div");
            item.className = "rep-feat-item";
            item.innerHTML = `<strong>${k}:</strong> ${AppState.features[k]}`;
            grid.appendChild(item);
        });
    }
}
