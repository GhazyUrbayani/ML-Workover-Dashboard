<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Workover Optimization Dashboard | IPFEST 2026</title>
    
    <!-- Chart.js untuk visualisasi -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        /* Header */
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
        }
        
        .header h1 {
            font-size: 2.2rem;
            color: #2d3748;
            margin-bottom: 8px;
        }
        
        .header p {
            color: #718096;
            font-size: 1rem;
        }
        
        /* KPI Cards Row */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .kpi-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: transform 0.2s;
        }
        
        .kpi-card:hover {
            transform: translateY(-5px);
        }
        
        .kpi-card.success {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        
        .kpi-card.warning {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        
        .kpi-card.info {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        
        .kpi-label {
            font-size: 0.85rem;
            opacity: 0.9;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .kpi-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .kpi-delta {
            font-size: 0.9rem;
            opacity: 0.85;
        }
        
        /* Charts Grid */
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .chart-card {
            background: #f7fafc;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .chart-card h3 {
            font-size: 1.1rem;
            color: #2d3748;
            margin-bottom: 15px;
            text-align: center;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .chart-card.full-width {
            grid-column: span 2;
        }
        
        canvas {
            max-height: 300px;
        }
        
        /* Recommendations Table */
        .table-container {
            background: #f7fafc;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        
        .table-container h3 {
            font-size: 1.1rem;
            color: #2d3748;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        
        th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        
        td {
            padding: 10px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        tr:hover {
            background: #edf2f7;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .badge.high {
            background: #38ef7d;
            color: #1a472a;
        }
        
        .badge.medium {
            background: #ffd93d;
            color: #744210;
        }
        
        .badge.low {
            background: #ff6b6b;
            color: #742a2a;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #e2e8f0;
            color: #718096;
            font-size: 0.85rem;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
            
            .chart-card.full-width {
                grid-column: span 1;
            }
            
            .kpi-row {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <div class="header">
            <h1>🛢️ ML Workover Optimization Dashboard</h1>
            <p>Real-Time Well Intervention Intelligence | IPFEST 2026 Hackathon</p>
        </div>
        
        <!-- KPI Cards -->
        <div class="kpi-row" id="kpiRow">
            <!-- Diisi oleh JavaScript -->
        </div>
        
        <!-- Charts Grid -->
        <div class="charts-grid">
            <!-- Cost Optimization Chart -->
            <div class="chart-card">
                <h3>💰 Cost Optimization Impact</h3>
                <canvas id="costChart"></canvas>
            </div>
            
            <!-- ROC Curve -->
            <div class="chart-card">
                <h3>📈 Model Performance (ROC Curve)</h3>
                <canvas id="rocChart"></canvas>
            </div>
            
            <!-- Confusion Matrix -->
            <div class="chart-card">
                <h3>🎯 Prediction Accuracy Matrix</h3>
                <canvas id="confusionChart"></canvas>
            </div>
            
            <!-- Cluster Distribution -->
            <div class="chart-card">
                <h3>🔬 Well Cluster Performance</h3>
                <canvas id="clusterChart"></canvas>
            </div>
        </div>
        
        <!-- Top Recommendations Table -->
        <div class="table-container">
            <h3>⭐ Top 10 Well Recommendations (High Priority)</h3>
            <table id="recommendationsTable">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Well Name</th>
                        <th>Success Probability</th>
                        <th>Cluster</th>
                        <th>Advisory</th>
                        <th>Potential Saving</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                    <!-- Diisi oleh JavaScript -->
                </tbody>
            </table>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>© 2026 Hengker Berkelas Team | Powered by LightGBM & Chart.js | Last Updated: <span id="lastUpdate"></span></p>
        </div>
    </div>
    
    <script>
        // ========================================
        // DATA SIMULATION (Ganti dengan data real dari API/JSON)
        // ========================================
        const dashboardData = {
            kpis: {
                totalWells: 999,
                baselineSuccessRate: 58.7,
                mlSuccessRate: 82.8,
                rocAuc: 0.828,
                totalCostBaseline: 167740000,
                totalCostOptimized: 96355000,
                costSaving: 71385000,
                savingPercent: 42.6,
                highPriorityWells: 165,
                testSize: 300
            },
            confusionMatrix: {
                tn: 83,
                fp: 37,
                fn: 15,
                tp: 165
            },
            rocCurve: {
                fpr: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0],
                tpr: [0.0, 0.5, 0.7, 0.78, 0.83, 0.87, 0.91, 0.95, 0.97, 0.99, 1.0]
            },
            clusterPerformance: [
                { cluster: 'Low Performer', count: 198, successRate: 66.7 },
                { cluster: 'High Performer', count: 2, successRate: 0.0 }
            ],
            topWells: [
                { rank: 1, well: 'WELL6', prob: 99.9, cluster: 'Low', advisory: 'Strongly Recommend', saving: 150000 },
                { rank: 2, well: 'WELL885', prob: 99.9, cluster: 'Low', advisory: 'Strongly Recommend', saving: 145000 },
                { rank: 3, well: 'WELL946', prob: 99.8, cluster: 'Low', advisory: 'Strongly Recommend', saving: 140000 },
                { rank: 4, well: 'WELL741', prob: 99.7, cluster: 'Low', advisory: 'Strongly Recommend', saving: 135000 },
                { rank: 5, well: 'WELL93', prob: 99.6, cluster: 'Low', advisory: 'Strongly Recommend', saving: 130000 },
                { rank: 6, well: 'WELL66', prob: 99.5, cluster: 'Low', advisory: 'Strongly Recommend', saving: 125000 },
                { rank: 7, well: 'WELL359', prob: 99.4, cluster: 'Low', advisory: 'Strongly Recommend', saving: 120000 },
                { rank: 8, well: 'WELL371', prob: 99.3, cluster: 'Low', advisory: 'Strongly Recommend', saving: 115000 },
                { rank: 9, well: 'WELL299', prob: 99.2, cluster: 'Low', advisory: 'Strongly Recommend', saving: 110000 },
                { rank: 10, well: 'WELL167', prob: 99.1, cluster: 'Low', advisory: 'Strongly Recommend', saving: 105000 }
            ]
        };
        
        // ========================================
        // RENDER KPI CARDS
        // ========================================
        function renderKPIs() {
            const kpiRow = document.getElementById('kpiRow');
            const kpis = dashboardData.kpis;
            
            kpiRow.innerHTML = `
                <div class="kpi-card success">
                    <div class="kpi-label">ROC AUC Score</div>
                    <div class="kpi-value">${kpis.rocAuc.toFixed(3)}</div>
                    <div class="kpi-delta">Model Performance</div>
                </div>
                
                <div class="kpi-card">
                    <div class="kpi-label">Success Rate (ML)</div>
                    <div class="kpi-value">${kpis.mlSuccessRate.toFixed(1)}%</div>
                    <div class="kpi-delta">↑ ${(kpis.mlSuccessRate - kpis.baselineSuccessRate).toFixed(1)}% vs Baseline</div>
                </div>
                
                <div class="kpi-card warning">
                    <div class="kpi-label">Cost Saving</div>
                    <div class="kpi-value">$${(kpis.costSaving / 1e6).toFixed(1)}M</div>
                    <div class="kpi-delta">${kpis.savingPercent.toFixed(1)}% Reduction</div>
                </div>
                
                <div class="kpi-card info">
                    <div class="kpi-label">High Priority Wells</div>
                    <div class="kpi-value">${kpis.highPriorityWells}</div>
                    <div class="kpi-delta">Out of ${kpis.testSize} Wells</div>
                </div>
            `;
        }
        
        // ========================================
        // CHART 1: Cost Optimization (Bar Chart)
        // ========================================
        function renderCostChart() {
            const ctx = document.getElementById('costChart').getContext('2d');
            const kpis = dashboardData.kpis;
            
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Baseline (Actual)', 'Optimized (ML)'],
                    datasets: [{
                        label: 'Total Cost (USD)',
                        data: [kpis.totalCostBaseline, kpis.totalCostOptimized],
                        backgroundColor: ['#ef4444', '#10b981'],
                        borderColor: ['#b91c1c', '#047857'],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => `$${(ctx.raw / 1e6).toFixed(2)}M`
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                callback: (value) => `$${(value / 1e6).toFixed(0)}M`
                            }
                        }
                    }
                }
            });
        }
        
        // ========================================
        // CHART 2: ROC Curve (Line Chart)
        // ========================================
        function renderROCChart() {
            const ctx = document.getElementById('rocChart').getContext('2d');
            const roc = dashboardData.rocCurve;
            
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: roc.fpr,
                    datasets: [
                        {
                            label: `ROC Curve (AUC = ${dashboardData.kpis.rocAuc})`,
                            data: roc.tpr,
                            borderColor: '#f97316',
                            backgroundColor: 'rgba(249, 115, 22, 0.2)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4
                        },
                        {
                            label: 'Random Classifier',
                            data: roc.fpr,
                            borderColor: '#6366f1',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: true, position: 'bottom' }
                    },
                    scales: {
                        x: { title: { display: true, text: 'False Positive Rate' } },
                        y: { title: { display: true, text: 'True Positive Rate' } }
                    }
                }
            });
        }
        
        // ========================================
        // CHART 3: Confusion Matrix (Bar Chart)
        // ========================================
        function renderConfusionChart() {
            const ctx = document.getElementById('confusionChart').getContext('2d');
            const cm = dashboardData.confusionMatrix;
            
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['True Negative', 'False Positive', 'False Negative', 'True Positive'],
                    datasets: [{
                        label: 'Count',
                        data: [cm.tn, cm.fp, cm.fn, cm.tp],
                        backgroundColor: ['#3b82f6', '#f59e0b', '#ef4444', '#10b981'],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: { beginAtZero: true }
                    }
                }
            });
        }
        
        // ========================================
        // CHART 4: Cluster Performance (Doughnut Chart)
        // ========================================
        function renderClusterChart() {
            const ctx = document.getElementById('clusterChart').getContext('2d');
            const clusters = dashboardData.clusterPerformance;
            
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: clusters.map(c => `${c.cluster} (${c.successRate}%)`),
                    datasets: [{
                        data: clusters.map(c => c.count),
                        backgroundColor: ['#6366f1', '#ec4899'],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom' }
                    }
                }
            });
        }
        
        // ========================================
        // RENDER TOP RECOMMENDATIONS TABLE
        // ========================================
        function renderTable() {
            const tbody = document.getElementById('tableBody');
            const wells = dashboardData.topWells;
            
            tbody.innerHTML = wells.map(w => `
                <tr>
                    <td><strong>${w.rank}</strong></td>
                    <td>${w.well}</td>
                    <td><strong>${w.prob.toFixed(1)}%</strong></td>
                    <td>${w.cluster} Performer</td>
                    <td><span class="badge high">${w.advisory}</span></td>
                    <td>$${w.saving.toLocaleString()}</td>
                </tr>
            `).join('');
        }
        
        // ========================================
        // INITIALIZE DASHBOARD
        // ========================================
        function initDashboard() {
            renderKPIs();
            renderCostChart();
            renderROCChart();
            renderConfusionChart();
            renderClusterChart();
            renderTable();
            
            // Update timestamp
            document.getElementById('lastUpdate').textContent = new Date().toLocaleString('id-ID');
        }
        
        // Run on page load
        initDashboard();
    </script>
</body>
</html>
