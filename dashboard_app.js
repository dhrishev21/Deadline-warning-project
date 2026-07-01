// State management
let state = {
    projects: [],
    filteredProjects: [],
    featureImportances: [],
    backtestTimeline: [],
    shapData: {},
    recommendationsData: {},
    engineeringMetrics: {},
    scenarioResult: null,
    
    // Filters
    searchQuery: '',
    riskLevel: 'All',
    minTeamSize: 3,
    minVelocity: 0.5,
    
    // Selection
    selectedProjectId: null,
    
    // Table sorting & pagination
    sortColumn: 'risk_score',
    sortDirection: 'desc',
    currentPage: 1,
    pageSize: 8,
    
    // Chart instances
    charts: {
        riskDist: null,
        featureImportance: null,
        backtest: null,
        shapChart: null,
        riskComparison: null,
        contributionComparison: null,
        forecast: null,
        monteCarlo: null
    }
};

const COLORS = {
    High: '#f43f5e',
    Medium: '#f59e0b',
    Low: '#10b981',
    Primary: '#38bdf8',
    PrimaryGlow: 'rgba(56, 189, 248, 0.15)',
    HighGlow: 'rgba(244, 63, 94, 0.15)',
    MedGlow: 'rgba(245, 158, 11, 0.15)',
    LowGlow: 'rgba(16, 185, 129, 0.15)',
    TextMain: '#f8fafc',
    TextMuted: '#94a3b8',
    GridLine: 'rgba(255, 255, 255, 0.05)'
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    loadData();
    setupEventListeners();
});

// Fetch data from APIs
async function loadData() {
    try {
        const [apiProjRes, featuresRes, backtestRes, metricsRes] = await Promise.all([
            fetch('/api/projects'),
            fetch('data/feature_importances.json'),
            fetch('data/backtest_timeline.json'),
            fetch('/api/engineering-metrics')
        ]);
        
        const apiData = await apiProjRes.json();
        state.projects = apiData.projects;
        state.shapData = apiData.shap;
        state.recommendationsData = apiData.recommendations;
        
        state.featureImportances = await featuresRes.json();
        state.backtestTimeline = await backtestRes.json();
        state.engineeringMetrics = await metricsRes.json();
        
        state.filteredProjects = [...state.projects];
        if (state.projects.length > 0) {
            state.selectedProjectId = state.projects[0].project_id;
        }
        
        // Initial draw
        populateProjectSelect();
        applyFilters();
        initCharts();
        updateEngineeringMetrics();
        updateIntelligencePanels();
        
        setTimeout(showWarningToast, 1500);
        
    } catch (err) {
        console.error('Error loading dashboard data files:', err);
    }
}

// Setup Event Listeners
function setupEventListeners() {
    // Basic filters
    document.getElementById('projectSearch').addEventListener('input', (e) => {
        state.searchQuery = e.target.value.trim().toLowerCase();
        state.currentPage = 1;
        applyFilters();
    });
    
    const riskBtns = document.querySelectorAll('[data-risk]');
    riskBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            riskBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.riskLevel = btn.getAttribute('data-risk');
            state.currentPage = 1;
            applyFilters();
        });
    });
    
    document.getElementById('teamSizeSlider').addEventListener('input', (e) => {
        state.minTeamSize = parseInt(e.target.value);
        document.getElementById('teamSizeVal').textContent = state.minTeamSize === 3 ? 'Any' : `â‰¥ ${state.minTeamSize}`;
        state.currentPage = 1;
        applyFilters();
    });
    
    document.getElementById('velocitySlider').addEventListener('input', (e) => {
        state.minVelocity = parseFloat(e.target.value);
        document.getElementById('velocityVal').textContent = state.minVelocity.toFixed(2);
        state.currentPage = 1;
        applyFilters();
    });
    
    document.getElementById('resetFiltersBtn').addEventListener('click', () => {
        document.getElementById('projectSearch').value = '';
        state.searchQuery = '';
        riskBtns.forEach(b => b.classList.remove('active'));
        document.querySelector('[data-risk="All"]').classList.add('active');
        state.riskLevel = 'All';
        
        document.getElementById('teamSizeSlider').value = 3;
        state.minTeamSize = 3;
        document.getElementById('teamSizeVal').textContent = 'Any';
        
        document.getElementById('velocitySlider').value = 0.5;
        state.minVelocity = 0.5;
        document.getElementById('velocityVal').textContent = '0.50';
        
        state.currentPage = 1;
        applyFilters();
    });
    
    // Project Selection for Intelligence Panel
    document.getElementById('projectSelect').addEventListener('change', (e) => {
        state.selectedProjectId = parseInt(e.target.value);
        updateIntelligencePanels();
    });
    
    // Scenario Simulators
    const scenarioInputs = ['scenarioTeam', 'scenarioVelocity', 'scenarioScope', 'scenarioDeadline', 'scenarioBugs'];
    scenarioInputs.forEach(id => {
        const el = document.getElementById(id);
        el.addEventListener('input', (e) => {
            document.getElementById(id + 'Val').textContent = e.target.value;
            debounceSimulate();
        });
    });
    
    const askBtn = document.getElementById('askAssistantBtn');
    if (askBtn) {
        askBtn.addEventListener('click', askProjectAssistant);
    }
    
    // Table Sorting
    document.querySelectorAll('#projectsTable th').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.getAttribute('data-col');
            if (!col) return;
            if (state.sortColumn === col) {
                state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                state.sortColumn = col;
                state.sortDirection = 'desc';
            }
            document.querySelectorAll('#projectsTable th').forEach(h => h.classList.remove('sorted-asc', 'sorted-desc'));
            th.classList.add(state.sortDirection === 'asc' ? 'sorted-asc' : 'sorted-desc');
            sortAndRenderTable();
        });
    });
    
    // Pagination
    document.getElementById('prevPageBtn').addEventListener('click', () => {
        if (state.currentPage > 1) { state.currentPage--; renderTableOnly(); }
    });
    document.getElementById('nextPageBtn').addEventListener('click', () => {
        const maxPage = Math.ceil(state.filteredProjects.length / state.pageSize) || 1;
        if (state.currentPage < maxPage) { state.currentPage++; renderTableOnly(); }
    });
}

function applyFilters() {
    state.filteredProjects = state.projects.filter(p => {
        const matchSearch = state.searchQuery === '' || p.project_id.toString().includes(state.searchQuery);
        const matchRisk = state.riskLevel === 'All' || p.risk_level === state.riskLevel;
        const matchTeam = p.team_size >= state.minTeamSize;
        const matchVel = p.sprint_velocity >= state.minVelocity;
        return matchSearch && matchRisk && matchTeam && matchVel;
    });
    updateKPIs();
    sortAndRenderTable();
    updateRiskDistChart();
    updatePortfolioDrivers();
}

function updateKPIs() {
    const total = state.filteredProjects.length;
    const highRiskCount = state.filteredProjects.filter(p => p.risk_level === 'High').length;
    const avgRisk = total > 0 ? (state.filteredProjects.reduce((sum, p) => sum + p.risk_score, 0) / total) : 0;
    
    document.getElementById('kpiTotalProjects').textContent = total;
    document.getElementById('kpiAvgRisk').textContent = `${(avgRisk * 100).toFixed(1)}%`;
    document.getElementById('kpiHighRisk').textContent = highRiskCount;
}

function sortAndRenderTable() {
    const col = state.sortColumn;
    const dir = state.sortDirection === 'asc' ? 1 : -1;
    state.filteredProjects.sort((a, b) => {
        let valA = a[col];
        let valB = b[col];
        if (typeof valA === 'string') return valA.localeCompare(valB) * dir;
        return (valA - valB) * dir;
    });
    renderTableOnly();
}

function renderTableOnly() {
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';
    const total = state.filteredProjects.length;
    const maxPage = Math.ceil(total / state.pageSize) || 1;
    if (state.currentPage > maxPage) state.currentPage = maxPage;
    if (state.currentPage < 1) state.currentPage = 1;
    
    const startIndex = (state.currentPage - 1) * state.pageSize;
    const endIndex = Math.min(startIndex + state.pageSize, total);
    const pageItems = state.filteredProjects.slice(startIndex, endIndex);
    
    if (pageItems.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">No projects match the active filters.</td></tr>`;
    } else {
        pageItems.forEach(p => {
            const riskColor = COLORS[p.risk_level] || COLORS.Primary;
            const riskPct = (p.risk_score * 100).toFixed(0);
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>#${p.project_id}</td>
                <td>${p.team_size} members</td>
                <td>${p.sprint_velocity.toFixed(2)}</td>
                <td>${(p.tasks_completed_pct || 0).toFixed(1)}%</td>
                <td>${p.bugs_open}</td>
                <td style="color: ${p.completion_gap < 0 ? COLORS.High : COLORS.Low}">${((p.completion_gap||0) * 100).toFixed(1)}%</td>
                <td>
                    <div class="score-bar-container">
                        <div class="score-bar-bg"><div class="score-bar-fill" style="width: ${riskPct}%; background-color: ${riskColor};"></div></div>
                        <span class="score-text" style="color: ${riskColor};">${riskPct}%</span>
                    </div>
                </td>
                <td><span class="badge badge-${(p.risk_level||'Low').toLowerCase()}">${p.risk_level}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }
    document.getElementById('currentPageNum').textContent = state.currentPage;
    document.getElementById('totalPageNum').textContent = maxPage;
    document.getElementById('prevPageBtn').disabled = state.currentPage === 1;
    document.getElementById('nextPageBtn').disabled = state.currentPage === maxPage;
    document.getElementById('tableSummaryText').textContent = total > 0 ? `Showing ${startIndex + 1}-${endIndex} of ${total} projects` : `Showing 0-0 of 0 projects`;
}

function updateEngineeringMetrics() {
    const em = state.engineeringMetrics;
    if(!em) return;
    const gh = em.github || {};
    const jira = em.jira || {};
    document.getElementById('metricCommits').textContent = gh.commits_per_week || '--';
    document.getElementById('metricOpenPrs').textContent = gh.pull_requests_opened || '--';
    document.getElementById('metricBlocked').textContent = jira.blocked_issues || '--';
    document.getElementById('metricSprintVelocity').textContent = jira.sprint_velocity ? jira.sprint_velocity.toFixed(1) : '--';
    document.getElementById('metricBugTrend').textContent = gh.issue_creation_rate || jira.bug_count || '--';
}

function populateProjectSelect() {
    const sel = document.getElementById('projectSelect');
    sel.innerHTML = '';
    state.projects.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.project_id;
        opt.textContent = `Project #${p.project_id} (${p.risk_level})`;
        sel.appendChild(opt);
    });
    if(state.selectedProjectId) sel.value = state.selectedProjectId;
}

function updateIntelligencePanels() {
    if (!state.selectedProjectId) return;
    const pid = state.selectedProjectId;
    const pInfo = state.projects.find(p => p.project_id === pid);
    if(pInfo) {
        resetScenarioControls(pInfo);
        document.getElementById('selectedRiskScore').textContent = `${(pInfo.risk_score * 100).toFixed(0)}% Risk`;
        document.getElementById('selectedRiskScore').style.color = COLORS[pInfo.risk_level];
    }
    
    // SHAP Explainability
    if (state.shapData.projects) {
        const shapProj = state.shapData.projects.find(p => p.project_id === pid);
        if (shapProj) {
            renderContributors('positiveContributors', shapProj.top_positive, 'increases_risk');
            renderContributors('negativeContributors', shapProj.top_negative, 'reduces_risk');
            updateShapChart(shapProj.contributors);
        }
    }
    
    // Recommendations
    if (state.recommendationsData.projects) {
        const recProj = state.recommendationsData.projects.find(p => p.project_id === pid);
        renderRecommendations(recProj ? recProj.recommendations : []);
    }
    
    // Run an initial simulation call
    debounceSimulate();
    loadCommandCenter();
}

function renderContributors(elId, list, type) {
    const el = document.getElementById(elId);
    el.innerHTML = '';
    list.forEach(item => {
        const div = document.createElement('div');
        div.style.marginBottom = '0.4rem';
        div.style.fontSize = '0.8rem';
        const color = type === 'increases_risk' ? COLORS.High : COLORS.Low;
        const sign = type === 'increases_risk' ? '+' : '-';
        div.innerHTML = `<span style="color:${color}; font-weight:bold;">${sign}${Math.abs(item.impact_pct).toFixed(1)}%</span> ${item.description}`;
        el.appendChild(div);
    });
}

function renderRecommendations(recs) {
    const el = document.getElementById('recommendationCards');
    el.innerHTML = '';
    if (!recs || recs.length === 0) {
        el.innerHTML = `<div class="chart-subtitle">No recommendations available for this profile.</div>`;
        return;
    }
    recs.forEach(r => {
        const div = document.createElement('div');
        div.className = 'recommendation-card';
        div.innerHTML = `
            <div class="recommendation-title">${r.recommendation}</div>
            <div class="recommendation-meta">
                <span>New risk: <span style="color:var(--text-main);">${(r.expected_new_risk * 100).toFixed(1)}%</span></span>
                <span style="color:var(--risk-low);">Improvement: ${(r.expected_risk_reduction * 100).toFixed(1)}%</span>
            </div>
            <div class="recommendation-meta">
                <span>Difficulty: ${r.implementation_difficulty || 'Medium'}</span>
                <span>Priority: ${Number(r.priority_score || 0).toFixed(1)}</span>
            </div>
        `;
        el.appendChild(div);
    });
}
let simTimeout = null;
function debounceSimulate() {
    clearTimeout(simTimeout);
    simTimeout = setTimeout(runSimulation, 500);
}

async function runSimulation() {
    if (!state.selectedProjectId) return;
    const reqBody = {
        project_id: state.selectedProjectId,
        overrides: {
            team_size: parseInt(document.getElementById('scenarioTeam').value),
            sprint_velocity: parseFloat(document.getElementById('scenarioVelocity').value),
            scope_changes: parseInt(document.getElementById('scenarioScope').value),
            deadline_extension_days: parseInt(document.getElementById('scenarioDeadline').value),
            bugs_open: parseInt(document.getElementById('scenarioBugs').value)
        }
    };
    
    try {
        const res = await fetch('/api/scenario', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(reqBody)
        });
        const data = await res.json();
        state.scenarioResult = data;
        
        // Update UI
        document.getElementById('scenarioCurrentRisk').textContent = `${(data.current.risk_score * 100).toFixed(1)}%`;
        document.getElementById('scenarioNewRisk').textContent = `${(data.scenario.risk_score * 100).toFixed(1)}%`;
        document.getElementById('scenarioDifference').textContent = `${(data.difference.risk_delta * 100).toFixed(1)}%`;
        document.getElementById('scenarioImprovement').textContent = `${data.difference.improvement_pct.toFixed(1)}%`;
        
        renderRecommendations(data.scenario.recommendations || []);
        updateSimulationCharts();
    } catch (e) {
        console.error("Simulation error:", e);
    }
}

// Charting Defaults
Chart.defaults.color = COLORS.TextMuted;
Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(13, 20, 38, 0.95)';
Chart.defaults.plugins.tooltip.borderColor = COLORS.GridLine;
Chart.defaults.plugins.tooltip.borderWidth = 1;

function initCharts() {
    const distCtx = document.getElementById('riskDistChart').getContext('2d');
    state.charts.riskDist = new Chart(distCtx, {
        type: 'doughnut',
        data: { labels: ['High Risk', 'Medium Risk', 'Low Risk'], datasets: [{data:[0,0,0], backgroundColor:[COLORS.High, COLORS.Medium, COLORS.Low], borderColor:'rgba(11,15,25,0.8)', borderWidth:2}] },
        options: { responsive: true, maintainAspectRatio: false, cutout: '68%', plugins:{ legend:{position:'bottom', labels:{boxWidth:12, font:{size:11}}}} }
    });
    
    const impCtx = document.getElementById('featureImportanceChart').getContext('2d');
    state.charts.featureImportance = new Chart(impCtx, {
        type: 'bar',
        data: { labels: [], datasets: [{label:'Importance', data:[], backgroundColor:COLORS.Primary, borderRadius:4}] },
        options: { indexAxis:'y', responsive: true, maintainAspectRatio: false, plugins:{legend:{display:false}}, scales:{x:{grid:{color:COLORS.GridLine}},y:{grid:{display:false}}} }
    });
    
    const btCtx = document.getElementById('backtestChart').getContext('2d');
    state.charts.backtest = new Chart(btCtx, {
        type: 'line',
        data: { labels: [], datasets: [{label:'Risk', data:[], borderColor:COLORS.High, backgroundColor:'rgba(244,63,94,0.08)', fill:true, tension:0.3, borderWidth:3}] },
        options: { responsive: true, maintainAspectRatio: false, scales:{y:{min:0, max:1}} }
    });
    
    const shapCtx = document.getElementById('shapChart').getContext('2d');
    state.charts.shapChart = new Chart(shapCtx, {
        type: 'bar',
        data: { labels: [], datasets: [{label:'Impact', data:[], backgroundColor:[]}] },
        options: { indexAxis:'y', responsive: true, maintainAspectRatio: false, plugins:{legend:{display:false}}, scales:{x:{grid:{color:COLORS.GridLine}},y:{grid:{display:false}}} }
    });
    
    const rcCtx = document.getElementById('riskComparisonChart').getContext('2d');
    state.charts.riskComparison = new Chart(rcCtx, {
        type: 'bar',
        data: { labels: ['Current', 'Scenario'], datasets: [{data:[], backgroundColor:[COLORS.High, COLORS.Primary]}] },
        options: { responsive: true, maintainAspectRatio: false, plugins:{legend:{display:false}, title:{display:true, text:'Risk Comparison'}}, scales:{y:{min:0,max:1, ticks:{callback:v=>`${(v*100).toFixed(0)}%`}}} }
    });

    const ccCtx = document.getElementById('contributionComparisonChart').getContext('2d');
    state.charts.contributionComparison = new Chart(ccCtx, {
        type: 'bar',
        data: { labels: [], datasets: [
            {label:'Current', data:[], backgroundColor:COLORS.High},
            {label:'Scenario', data:[], backgroundColor:COLORS.Primary}
        ] },
        options: { indexAxis:'y', responsive: true, maintainAspectRatio: false, plugins:{title:{display:true, text:'Feature Contribution Comparison'}}, scales:{x:{grid:{color:COLORS.GridLine}, ticks:{callback:v=>`${v.toFixed(0)}%`}}, y:{grid:{display:false}}} }
    });
    
    updateRiskDistChart();
    
    // Feature Importances
    if (state.featureImportances.length > 0) {
        const sorted = [...state.featureImportances].sort((a,b)=>b.importance-a.importance);
        state.charts.featureImportance.data.labels = sorted.map(f=>formatFeatureName(f.feature));
        state.charts.featureImportance.data.datasets[0].data = sorted.map(f=>f.importance);
        state.charts.featureImportance.data.datasets[0].backgroundColor = sorted.map((v,i)=>i<3?COLORS.High:COLORS.Primary);
        state.charts.featureImportance.update();
    }
    
    // Backtest
    if (state.backtestTimeline.length > 0) {
        state.charts.backtest.data.labels = state.backtestTimeline.map(t=>`Week ${t.week}`);
        state.charts.backtest.data.datasets[0].data = state.backtestTimeline.map(t=>t.risk_score);
        state.charts.backtest.update();
    }
}

function updateRiskDistChart() {
    if (!state.charts.riskDist) return;
    const counts = { High: 0, Medium: 0, Low: 0 };
    state.filteredProjects.forEach(p => counts[p.risk_level] = (counts[p.risk_level]||0)+1);
    state.charts.riskDist.data.datasets[0].data = [counts.High, counts.Medium, counts.Low];
    state.charts.riskDist.update();
}

function updateShapChart(contributors) {
    if (!state.charts.shapChart || !contributors) return;
    const items = [...contributors].sort((a,b)=>Math.abs(b.shap_value) - Math.abs(a.shap_value)).slice(0,5);
    state.charts.shapChart.data.labels = items.map(c=>c.label);
    state.charts.shapChart.data.datasets[0].data = items.map(c=>c.shap_value);
    state.charts.shapChart.data.datasets[0].backgroundColor = items.map(c=>c.shap_value > 0 ? COLORS.High : COLORS.Low);
    state.charts.shapChart.update();
}

function updateSimulationCharts() {
    if (!state.scenarioResult || !state.charts.riskComparison) return;
    const cur = state.scenarioResult.current.risk_score;
    const sce = state.scenarioResult.scenario.risk_score;
    state.charts.riskComparison.data.datasets[0].data = [cur, sce];
    state.charts.riskComparison.data.datasets[0].backgroundColor = [COLORS.High, sce < cur ? COLORS.Low : COLORS.High];
    state.charts.riskComparison.update();

    if (state.charts.contributionComparison) {
        const current = [...(state.scenarioResult.current.explanation?.contributors || [])]
            .sort((a, b) => Math.abs(b.impact_pct) - Math.abs(a.impact_pct))
            .slice(0, 5);
        const scenarioMap = new Map((state.scenarioResult.scenario.explanation?.contributors || []).map(c => [c.feature, c]));
        state.charts.contributionComparison.data.labels = current.map(c => c.label);
        state.charts.contributionComparison.data.datasets[0].data = current.map(c => c.impact_pct);
        state.charts.contributionComparison.data.datasets[1].data = current.map(c => scenarioMap.get(c.feature)?.impact_pct || 0);
        state.charts.contributionComparison.update();
    }
}
function resetScenarioControls(project) {
    if (!project) return;
    document.getElementById('scenarioTeam').value = Math.round(project.team_size || 5);
    document.getElementById('scenarioVelocity').value = Number(project.sprint_velocity || 1).toFixed(2);
    document.getElementById('scenarioScope').value = Math.round(project.scope_changes || 0);
    document.getElementById('scenarioDeadline').value = 14;
    document.getElementById('scenarioBugs').value = Math.round(project.bugs_open || 0);
    document.getElementById('scenarioTeamVal').textContent = document.getElementById('scenarioTeam').value;
    document.getElementById('scenarioVelocityVal').textContent = document.getElementById('scenarioVelocity').value;
    document.getElementById('scenarioScopeVal').textContent = document.getElementById('scenarioScope').value;
    document.getElementById('scenarioDeadlineVal').textContent = `${document.getElementById('scenarioDeadline').value} days`;
    document.getElementById('scenarioBugsVal').textContent = document.getElementById('scenarioBugs').value;
}

function updatePortfolioDrivers() {
    const el = document.getElementById('portfolioContributors');
    if (!el || !state.shapData.projects) return;
    const filteredIds = new Set(state.filteredProjects.map(p => p.project_id));
    const byFeature = new Map();
    state.shapData.projects.forEach(project => {
        if (!filteredIds.has(project.project_id)) return;
        (project.contributors || []).forEach(c => {
            if (!byFeature.has(c.feature)) byFeature.set(c.feature, {label: c.label, total: 0, abs: 0, count: 0});
            const item = byFeature.get(c.feature);
            item.total += c.impact_pct;
            item.abs += Math.abs(c.impact_pct);
            item.count += 1;
        });
    });
    const drivers = [...byFeature.values()].map(item => ({
        label: item.label,
        impact: item.total / item.count,
        abs: item.abs / item.count
    })).sort((a, b) => b.abs - a.abs).slice(0, 5);
    el.innerHTML = drivers.length ? drivers.map(item => `
        <div class="contributor-item ${item.impact >= 0 ? 'positive' : 'negative'}">
            <span>${item.label}</span>
            <span>${item.impact >= 0 ? '+' : ''}${item.impact.toFixed(1)}%</span>
        </div>
    `).join('') : '<div class="chart-subtitle">No portfolio drivers for active filters.</div>';
}
function formatFeatureName(name) {
    return name.split('_').map(w=>w==='pct'?'%':w.charAt(0).toUpperCase()+w.slice(1)).join(' ');
}

function showWarningToast() {
    const alert = document.getElementById('alertToast');
    if(alert) {
        alert.classList.add('show');
        setTimeout(() => alert.classList.remove('show'), 8000);
    }
}









async function loadCommandCenter() {
    if (!state.selectedProjectId) return;
    setCommandCenterLoading();
    await Promise.allSettled([
        loadForecastPanel(),
        loadMonteCarloPanel(),
        loadSimilarityPanel(),
        loadPortfolioPanel(),
        loadDriftPanel(),
        loadScenarioLabPanel()
    ]);
}

function setCommandCenterLoading() {
    const ids = ['forecastTrend', 'similarInsight', 'driftStatus', 'scenarioLabSummary'];
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = 'Loading...';
    });
}

async function loadForecastPanel() {
    const data = await fetchJson(`/api/forecast?project_id=${state.selectedProjectId}&horizon_weeks=4`);
    document.getElementById('forecastTrend').textContent = data.escalation_warning || 'No high-risk threshold crossing in forecast horizon.';
    document.getElementById('forecastTrendValue').textContent = trendLabel(data.risk_trend);
    document.getElementById('forecastHighDate').textContent = shortDate(data.predicted_high_risk_date) || '--';
    document.getElementById('forecastCurrentRisk').textContent = `${data.current_risk.toFixed(1)}%`;
    const last = data.forecast[data.forecast.length - 1];
    document.getElementById('forecastWeek4Risk').textContent = last ? `${last.risk.toFixed(1)}%` : '--';
    renderForecastChart(data);
}

async function loadMonteCarloPanel() {
    const data = await fetchJson(`/api/monte-carlo?project_id=${state.selectedProjectId}&simulations=1000`);
    document.getElementById('mcP50').textContent = shortDate(data.p50_completion);
    document.getElementById('mcP80').textContent = shortDate(data.p80_completion);
    document.getElementById('mcP90').textContent = shortDate(data.p90_completion);
    document.getElementById('mcOnTime').textContent = `${(data.probability_on_time * 100).toFixed(0)}%`;
    renderMonteCarloChart(data);
}

async function loadSimilarityPanel() {
    const data = await fetchJson(`/api/similar-projects?project_id=${state.selectedProjectId}&top_n=5&algorithm=cosine_similarity`);
    document.getElementById('similarInsight').textContent = data.insight;
    const body = document.getElementById('similarProjectsBody');
    body.innerHTML = data.similar_projects.map(item => `
        <tr>
            <td>${item.project_name}</td>
            <td>${item.similarity_pct.toFixed(1)}%</td>
            <td>${item.outcome}</td>
        </tr>
    `).join('');
}

async function loadPortfolioPanel() {
    const data = await fetchJson('/api/portfolio');
    const metrics = data.portfolio_metrics;
    document.getElementById('portfolioRisk').textContent = `${metrics.portfolio_risk_score}%`;
    document.getElementById('portfolioHighRisk').textContent = metrics.high_risk_projects;
    document.getElementById('portfolioDelay').textContent = `${metrics.total_expected_delay_days}d`;
    document.getElementById('portfolioUtilization').textContent = `${metrics.resource_utilization}%`;
    const body = document.getElementById('portfolioBody');
    body.innerHTML = data.command_center.slice(0, 8).map(item => `
        <tr>
            <td>${item.project}</td>
            <td>${item.department}</td>
            <td>${item.risk.toFixed(1)}%</td>
            <td>${trendLabel(item.forecast)}</td>
            <td>${item.delay_days}d</td>
            <td>${item.priority_score.toFixed(2)}</td>
        </tr>
    `).join('');
}

async function loadDriftPanel() {
    const data = await fetchJson('/api/drift?simulate_drift=true');
    document.getElementById('driftStatus').textContent = `Status: ${data.overall_status}. Samples analyzed: ${data.samples_analyzed}.`;
    const alerts = data.alerts && data.alerts.length ? data.alerts : [{severity: 'stable', message: 'No critical drift alerts.', recommendation: 'continue monitoring'}];
    document.getElementById('driftAlerts').innerHTML = alerts.slice(0, 5).map(alert => `
        <div class="alert-row"><strong>${alert.severity}</strong>: ${alert.message}<br><span class="chart-subtitle">${alert.recommendation}</span></div>
    `).join('');
}

async function loadScenarioLabPanel() {
    const payload = {
        project_id: state.selectedProjectId,
        scenarios: [
            {name: 'Scenario A: Add 2 Developers', overrides: {team_size: Math.min(20, (getSelectedProject()?.team_size || 5) + 2)}},
            {name: 'Scenario B: Reduce Scope 15%', overrides: {scope_changes: Math.max(0, Math.round((getSelectedProject()?.scope_changes || 0) * 0.85))}},
            {name: 'Scenario C: Extend 2 Weeks', overrides: {deadline_extension_days: 14}}
        ]
    };
    const data = await fetchJson('/api/scenario-lab', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    document.getElementById('scenarioLabSummary').textContent = `Best scenario: ${data.best_scenario}`;
    document.getElementById('scenarioLabBody').innerHTML = data.comparison_table.map(row => `
        <tr>
            <td>${row.scenario}</td>
            <td>${Number(row.risk_score).toFixed(1)}%</td>
            <td>${Number(row.risk_delta).toFixed(1)}%</td>
            <td>${Number(row.improvement_pct).toFixed(1)}%</td>
            <td>${row.key_changes}</td>
        </tr>
    `).join('');
}

async function askProjectAssistant() {
    const answer = document.getElementById('assistantAnswer');
    const question = document.getElementById('assistantQuestion').value || 'Why is this project risky?';
    answer.textContent = 'Retrieving project context...';
    const data = await fetchJson('/api/assistant', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({project_id: state.selectedProjectId, question})
    });
    answer.innerHTML = `<div>${escapeHtml(data.response).replace(/\n/g, '<br>')}</div><div class="chart-subtitle">Sources: ${data.sources_used.join(', ')}</div>`;
}

function renderForecastChart(data) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    const labels = ['Now', ...data.forecast.map(item => `W${item.week}`)];
    const risks = [data.current_risk, ...data.forecast.map(item => item.risk)];
    const lower = [data.current_risk, ...data.forecast.map(item => item.confidence_lower)];
    const upper = [data.current_risk, ...data.forecast.map(item => item.confidence_upper)];
    if (state.charts.forecast) state.charts.forecast.destroy();
    state.charts.forecast = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets: [
            {label: 'Upper confidence', data: upper, borderColor: 'rgba(56,189,248,0.25)', backgroundColor: 'rgba(56,189,248,0.08)', fill: '+1', pointRadius: 0},
            {label: 'Lower confidence', data: lower, borderColor: 'rgba(56,189,248,0.25)', backgroundColor: 'rgba(56,189,248,0.08)', fill: false, pointRadius: 0},
            {label: 'Forecast risk', data: risks, borderColor: COLORS.High, backgroundColor: COLORS.HighGlow, tension: 0.25, borderWidth: 3},
            {label: 'High threshold', data: labels.map(() => 65), borderColor: 'rgba(244,63,94,0.5)', borderDash: [4,4], pointRadius: 0}
        ]},
        options: {responsive: true, maintainAspectRatio: false, scales: {y: {min: 0, max: 100, ticks: {callback: v => `${v}%`}, grid: {color: COLORS.GridLine}}, x: {grid: {display: false}}}}
    });
}

function renderMonteCarloChart(data) {
    const ctx = document.getElementById('monteCarloChart').getContext('2d');
    const cdf = data.cdf || [];
    if (state.charts.monteCarlo) state.charts.monteCarlo.destroy();
    state.charts.monteCarlo = new Chart(ctx, {
        type: 'line',
        data: {labels: cdf.map(item => shortDate(item.date)), datasets: [{label: 'Cumulative delivery probability', data: cdf.map(item => item.probability * 100), borderColor: COLORS.Low, backgroundColor: COLORS.LowGlow, fill: true, tension: 0.25}]},
        options: {responsive: true, maintainAspectRatio: false, scales: {y: {min: 0, max: 100, ticks: {callback: v => `${v}%`}, grid: {color: COLORS.GridLine}}, x: {grid: {display: false}, ticks: {maxTicksLimit: 6}}}}
    });
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) throw new Error(`${url} failed with ${response.status}`);
    return response.json();
}

function getSelectedProject() {
    return state.projects.find(project => project.project_id === state.selectedProjectId);
}

function shortDate(value) {
    if (!value) return '--';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '--';
    return date.toLocaleDateString(undefined, {month: 'short', day: 'numeric'});
}

function trendLabel(value) {
    if (value === 'up' || value === 'increasing') return '↑ Increasing';
    if (value === 'down' || value === 'decreasing') return '↓ Decreasing';
    if (value === 'flat' || value === 'stable') return '→ Stable';
    return value || '--';
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"]/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[char]));
}

