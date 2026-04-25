const dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const browseLink  = document.getElementById('browse-link');
const resultsCard = document.getElementById('results-card');
const statusCard  = document.getElementById('status-card');
const loadingOverlay = document.getElementById('loading-overlay');

const previewImg    = document.getElementById('preview-img');
const heatmapImg    = document.getElementById('heatmap-img');
const diagnosisHero = document.getElementById('diagnosis-hero');
const diagnosisName = document.getElementById('diagnosis-name');
const confidenceVal = document.getElementById('confidence-val');
const confidenceBar = document.getElementById('confidence-bar');
const confRing      = document.getElementById('conf-ring');
const ringPct       = document.getElementById('ring-pct');
const fullAnalysis  = document.getElementById('full-analysis');

const CIRC = 169.6;

const diagColors = {
    covid:        { bg: 'covid-bg',     txt: 'covid-txt',     score: '#DC2626', ring: '#DC2626' },
    normal:       { bg: 'normal-bg',    txt: 'normal-txt',    score: '#059669', ring: '#10B981' },
    pneumonia:    { bg: 'pneumonia-bg', txt: 'pneumonia-txt', score: '#D97706', ring: '#F59E0B' },
    tuberculosis: { bg: 'tb-bg',        txt: 'tb-txt',        score: '#7C3AED', ring: '#8B5CF6' }
};

browseLink.onclick = (e) => { e.stopPropagation(); fileInput.click(); };
dropZone.onclick   = () => fileInput.click();

fileInput.onchange = (e) => {
    if (e.target.files.length > 0) handleUpload(e.target.files[0]);
};

dropZone.ondragover  = (e) => { e.preventDefault(); dropZone.classList.add('active'); };
dropZone.ondragleave = ()  => dropZone.classList.remove('active');
dropZone.ondrop      = (e) => {
    e.preventDefault();
    dropZone.classList.remove('active');
    if (e.dataTransfer.files.length > 0) handleUpload(e.dataTransfer.files[0]);
};

async function handleUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => { previewImg.src = e.target.result; };
    reader.readAsDataURL(file);

    statusCard.classList.add('hidden');
    resultsCard.classList.remove('hidden');
    loadingOverlay.classList.remove('hidden');

    resetDiagnosisHero();
    heatmapImg.src = '';
    confidenceBar.style.width = '0%';
    confidenceVal.innerText = '—';
    confRing.style.strokeDashoffset = CIRC;
    confRing.style.stroke = '#2563EB';
    ringPct.innerText = '0%';
    fullAnalysis.innerHTML = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res  = await fetch('/predict', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        displayResults(data);
    } catch (err) {
        console.error(err);
        loadingOverlay.classList.add('hidden');
        diagnosisName.innerText = 'Error';
        alert('Inference failed: ' + err.message);
    }
}

function displayResults(data) {
    loadingOverlay.classList.add('hidden');

    const key    = data.diagnosis.toLowerCase();
    const colors = diagColors[key] || { bg:'', txt:'', score:'var(--blue)', ring:'var(--blue)' };
    const pct    = parseFloat(data.confidence);

    diagnosisHero.className = 'diagnosis-hero ' + colors.bg;
    diagnosisName.className = 'dh-name ' + colors.txt;
    diagnosisName.innerText = data.diagnosis.toUpperCase();

    confRing.style.stroke = colors.ring;
    setTimeout(() => {
        const offset = CIRC - (pct / 100) * CIRC;
        confRing.style.strokeDashoffset = offset;
        ringPct.innerText = Math.round(pct) + '%';
    }, 60);

    confidenceVal.innerText = data.confidence;
    setTimeout(() => { confidenceBar.style.width = data.confidence; }, 60);

    heatmapImg.src = data.heatmap_url;

    const scoreColors = {
        covid: '#DC2626', normal: '#059669',
        pneumonia: '#D97706', tuberculosis: '#7C3AED'
    };
    let html = '';
    for (const [k, v] of Object.entries(data.all_scores)) {
        const c = scoreColors[k] || 'var(--txt-primary)';
        html += `<div class="bk-item">
                    <div class="bk-name">${k}</div>
                    <div class="bk-score" style="color:${c}">${v}</div>
                 </div>`;
    }
    fullAnalysis.innerHTML = html;
}

function resetDiagnosisHero() {
    diagnosisHero.className = 'diagnosis-hero';
    diagnosisName.className = 'dh-name';
    diagnosisName.innerText = '—';
}

function resetApp() {
    resultsCard.classList.add('hidden');
    statusCard.classList.remove('hidden');
    fileInput.value = '';
    previewImg.src = '';
    heatmapImg.src = '';
    loadingOverlay.classList.add('hidden');
    resetDiagnosisHero();
    confidenceBar.style.width = '0%';
    confidenceVal.innerText = '—';
    confRing.style.strokeDashoffset = CIRC;
    ringPct.innerText = '0%';
    fullAnalysis.innerHTML = '';
}
