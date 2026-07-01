// features_app.js — Advanced Intelligence UI logic

let featuresState = {
    charts: {
        forecast: null,
        monteCarlo: null
    }
};

document.addEventListener('DOMContentLoaded', () => {
    // Hook into projectSelect change from dashboard_app.js
    const projSelect = document.getElementById('projectSelect');
    if (projSelect) {
        projSelect.addEventListener('change', (e) => {
            const pid = e.target.value;
            loadAdvancedFeatures(pid);
        });
    }

    // Attach to assistant button
    const askBtn = document.getElementById('askAssistantBtn');
    const questionInput = document.getElementById('assistantQuestion');
    if (askBtn) {
        askBtn.addEventListener('click', askAssistant);
    }
    if (questionInput) {
        questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') askAssistant();
        });
    }

    // Initialize charts
    initFeatureCharts();
    
    // Load portfolio & drift
    loadPortfolioData();
    loadDriftData();

    // Load initial project after delay
    setTimeout(() => {
        if(typeof state !== 'undefined' && state.selectedProjectId) {
            loadAdvancedFeatures(state.selectedProjectId);
        }
    }, 1000);
});

function initFeatureCharts() {
    const fcCtx = document.getElementById('forecastChart');
    if (fcCtx) {
        featuresState.charts.forecast = new Chart(fcCtx.getContext('2d'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: { 
                responsive: true, maintainAspectRatio: false, 
                scales: { y: { min: 0, max: 100, title: {display: true, text: 'Risk %'} } },
                plugins: { legend: { display: false } }
            }
        });
    }

    const mcCtx = document.getElementById('monteCarloChart');
    if (mcCtx) {
        featuresState.charts.monteCarlo = new Chart(mcCtx.getContext('2d'), {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: { 
                responsive: true, maintainAspectRatio: false,
                scales: { y: { display: false }, x: { title: {display: false} } },
                plugins: { legend: { display: false } }
            }
        });
    }
}

async function loadAdvancedFeatures(projectId) {
    if (!projectId) return;

    // Load Forecast
    fetch(`/api/forecast?project_id=${projectId}`)
        .then(r => r.json())
        .then(data => updateForecastUI(data))
        .catch(e => console.error("Forecast Error:", e));

    // Load Monte Carlo
    fetch(`/api/monte-carlo?project_id=${projectId}`)
        .then(r => r.json())
        .then(data => updateMonteCarloUI(data))
        .catch(e => console.error("Monte Carlo Error:", e));

    // Load Similar Projects
    fetch(`/api/similar-projects?project_id=${projectId}`)
        .then(r => r.json())
        .then(data => updateSimilarityUI(data))
        .catch(e => console.error("Similarity Error:", e));

    // Load Scenario Lab
    const sampleScenarios = [
        { name: "+3 Devs", overrides: { team_size: (state.projectData.team_size || 5) + 3 } },
        { name: "Scope Cut", overrides: { scope_changes: 0 } },
        { name: "2 Week Delay", overrides: { days_elapsed: (state.projectData.days_elapsed || 0) + 14 } }
    ];
    fetch('/api/scenario-lab', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({project_id: parseInt(projectId), scenarios: sampleScenarios})
    })
    .then(r => r.json())
    .then(data => updateScenarioLabUI(data))
    .catch(e => console.error("Scenario Lab Error:", e));
        
    // Reset Chat Assistant
    const answerDiv = document.getElementById('assistantAnswer');
    if(answerDiv) {
        answerDiv.innerHTML = 'Ask a project question to retrieve a grounded answer.';
        answerDiv.className = 'assistant-answer loading-text';
    }
}

function updateForecastUI(data) {
    if (!data.forecast) return;
    
    document.getElementById('forecastTrend').textContent = data.escalation_warning || 'No critical escalations';
    
    document.getElementById('forecastTrendValue').textContent = data.risk_trend.toUpperCase();
    document.getElementById('forecastHighDate').textContent = data.predicted_high_risk_date ? data.predicted_high_risk_date.substring(5) : 'N/A';
    document.getElementById('forecastCurrentRisk').textContent = data.current_risk + '%';
    
    const week4 = data.forecast.find(f => f.week === 4);
    if(week4) document.getElementById('forecastWeek4Risk').textContent = week4.risk + '%';
    
    const labels = ["Current", ...data.forecast.map(f => `W${f.week}`)];
    const riskData = [data.current_risk, ...data.forecast.map(f => f.risk)];
    const lowerData = [data.current_risk, ...data.forecast.map(f => f.confidence_lower)];
    const upperData = [data.current_risk, ...data.forecast.map(f => f.confidence_upper)];

    if(featuresState.charts.forecast) {
        featuresState.charts.forecast.data = {
            labels: labels,
            datasets: [
                {
                    label: 'Risk',
                    data: riskData,
                    borderColor: COLORS.High,
                    backgroundColor: COLORS.High,
                    tension: 0.3,
                    borderWidth: 3,
                    zIndex: 2
                },
                {
                    label: 'Upper',
                    data: upperData,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(244, 63, 94, 0.1)',
                    fill: '+1',
                    pointRadius: 0
                },
                {
                    label: 'Lower',
                    data: lowerData,
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    fill: false,
                    pointRadius: 0
                }
            ]
        };
        featuresState.charts.forecast.update();
    }
}

function updateMonteCarloUI(data) {
    document.getElementById('mcP50').textContent = data.p50_days + 'd';
    document.getElementById('mcP80').textContent = data.p80_days + 'd';
    document.getElementById('mcP90').textContent = data.p90_days + 'd';
    document.getElementById('mcOnTime').textContent = (data.delivery_probability * 100).toFixed(0) + '%';

    if (data.histogram && data.histogram.length > 0 && featuresState.charts.monteCarlo) {
        const labels = data.histogram.map(h => `${h.bin_start}`);
        const counts = data.histogram.map(h => h.count);

        featuresState.charts.monteCarlo.data = {
            labels: labels,
            datasets: [{
                label: 'Simulations',
                data: counts,
                backgroundColor: COLORS.Primary,
                borderRadius: 2
            }]
        };
        featuresState.charts.monteCarlo.update();
    }
}

function updateSimilarityUI(data) {
    document.getElementById('similarInsight').textContent = data.insight;
    const tbody = document.getElementById('similarProjectsBody');
    if(!tbody) return;
    
    tbody.innerHTML = '';
    
    if (data.similar_projects && data.similar_projects.length > 0) {
        data.similar_projects.forEach(p => {
            const tr = document.createElement('tr');
            const color = p.delayed ? COLORS.High : COLORS.Low;
            tr.innerHTML = `
                <td>#${p.project_id} - ${p.project_name}</td>
                <td>${p.similarity_pct}%</td>
                <td style="color: ${color};">${p.outcome}</td>
            `;
            tbody.appendChild(tr);
        });
    } else {
        tbody.innerHTML = `<tr><td colspan="3">No similar projects found.</td></tr>`;
    }
}

async function askAssistant() {
    const input = document.getElementById('assistantQuestion');
    const question = input.value.trim();
    if (!question || typeof state === 'undefined' || !state.selectedProjectId) return;
    
    const answerDiv = document.getElementById('assistantAnswer');
    answerDiv.className = 'assistant-answer loading-text';
    answerDiv.innerHTML = 'Thinking...';
    
    try {
        const res = await fetch('/api/assistant', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({project_id: state.selectedProjectId, question: question})
        });
        const data = await res.json();
        
        const formattedResponse = data.response.replace(/\n/g, '<br>');
        
        answerDiv.className = 'assistant-answer';
        answerDiv.innerHTML = `
            ${formattedResponse}
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; text-align:right;">[Mode: ${data.generation_mode}]</div>
        `;
    } catch (e) {
        answerDiv.className = 'assistant-answer';
        answerDiv.innerHTML = `<span style="color:var(--risk-high);">Error: Failed to reach assistant.</span>`;
    }
}

async function loadPortfolioData() {
    try {
        const res = await fetch('/api/portfolio');
        const data = await res.json();
        
        if (data.portfolio_metrics) {
            document.getElementById('portfolioRisk').textContent = data.portfolio_metrics.portfolio_risk_score + '%';
            document.getElementById('portfolioHighRisk').textContent = data.portfolio_metrics.high_risk_projects;
            document.getElementById('portfolioDelay').textContent = data.portfolio_metrics.total_expected_delay_days + 'd';
            document.getElementById('portfolioUtilization').textContent = data.portfolio_metrics.total_resources;
        }

        const tbody = document.getElementById('portfolioBody');
        if(!tbody) return;
        
        tbody.innerHTML = '';
        if (data.top_priority_projects) {
            data.top_priority_projects.forEach(p => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>Project #${p.project_id}</td>
                    <td>${p.business_impact}/10</td>
                    <td><span class="badge badge-${p.risk_level.toLowerCase()}">${p.risk_level}</span></td>
                    <td>--</td>
                    <td>--</td>
                    <td><strong style="color:var(--text-main);">${round(p.priority_score, 1)}</strong></td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (e) {
        console.error("Portfolio Error:", e);
    }
}

async function loadDriftData() {
    try {
        const res = await fetch('/api/drift');
        const data = await res.json();
        
        if (data.drift_report) {
            const statusEl = document.getElementById('driftStatus');
            if(statusEl) statusEl.textContent = `Status: ${data.drift_report.overall_status.toUpperCase()}`;

            const listEl = document.getElementById('driftAlerts');
            if(!listEl) return;
            listEl.innerHTML = '';
            
            data.drift_report.feature_drift.forEach(f => {
                const color = f.status === 'critical' ? COLORS.High : (f.status === 'warning' ? COLORS.Medium : COLORS.Low);
                const div = document.createElement('div');
                div.className = 'alert-item';
                div.innerHTML = `
                    <span class="alert-label" style="background:${color}20; color:${color}">${f.status.toUpperCase()}</span>
                    <span>${f.feature.replace(/_/g, ' ')} shifted (PSI: ${f.psi.toFixed(3)})</span>
                `;
                listEl.appendChild(div);
            });
        }
    } catch (e) {
        console.error("Drift Error:", e);
    }
}

function round(val, dec) {
    return Number(Math.round(val+'e'+dec)+'e-'+dec);
}

function updateScenarioLabUI(data) {
    const summary = document.getElementById('scenarioLabSummary');
    if (summary) summary.textContent = `Baseline Risk: ${(data.current.risk_score * 100).toFixed(1)}% | Best option: ${data.best_scenario}`;
    
    const tbody = document.getElementById('scenarioLabBody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    data.comparison_table.forEach(s => {
        const tr = document.createElement('tr');
        const improvement = s.Delta.startsWith('-') ? s.Delta.substring(1) : '0%';
        const color = s.Delta.startsWith('-') ? COLORS.Low : COLORS.High;
        
        tr.innerHTML = `
            <td><strong>${s.Scenario}</strong></td>
            <td>${s.Risk}</td>
            <td style="color:${color}">${s.Delta}</td>
            <td style="color:${color}">${improvement}</td>
            <td style="font-size:0.75rem; color:var(--text-muted);">${s['Key Changes']}</td>
        `;
        tbody.appendChild(tr);
    });
}
