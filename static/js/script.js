/* ===================================================================
   Customer Segmentation JavaScript Interactions
   =================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // Auto-dismiss alert banners after 6 seconds
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            if (bsAlert) bsAlert.close();
        }, 6000);
    });

    // Initialize Interactive Dashboard Charts if chart canvases are present
    initDashboardCharts();

    // Table Search filter
    initTableSearch();
});

/**
 * Pre-fills the customer prediction form with predefined persona profiles.
 */
function fillSampleData(persona) {
    const presets = {
        'high_value': {
            age: 38,
            gender: 'Female',
            income: 2100000,
            spending: 88,
            frequency: 30,
            aov: 8500,
            total_purchases: 95,
            recency: 7,
            online: 54,
            offline: 41,
            visits: 30
        },
        'budget': {
            age: 44,
            gender: 'Male',
            income: 420000,
            spending: 20,
            frequency: 7,
            aov: 1200,
            total_purchases: 14,
            recency: 42,
            online: 4,
            offline: 10,
            visits: 7
        },
        'trendsetter': {
            age: 23,
            gender: 'Female',
            income: 720000,
            spending: 86,
            frequency: 22,
            aov: 2800,
            total_purchases: 56,
            recency: 9,
            online: 46,
            offline: 10,
            visits: 42
        },
        'conservative': {
            age: 56,
            gender: 'Male',
            income: 2200000,
            spending: 21,
            frequency: 5,
            aov: 7400,
            total_purchases: 13,
            recency: 58,
            online: 6,
            offline: 7,
            visits: 9
        },
        'at_risk': {
            age: 47,
            gender: 'Female',
            income: 950000,
            spending: 46,
            frequency: 4,
            aov: 2400,
            total_purchases: 6,
            recency: 195,
            online: 3,
            offline: 3,
            visits: 4
        }
    };

    const data = presets[persona];
    if (!data) return;

    if (document.getElementById('age')) document.getElementById('age').value = data.age;
    if (document.getElementById('gender')) document.getElementById('gender').value = data.gender;
    if (document.getElementById('income')) document.getElementById('income').value = data.income;
    if (document.getElementById('spending')) document.getElementById('spending').value = data.spending;
    if (document.getElementById('frequency')) document.getElementById('frequency').value = data.frequency;
    if (document.getElementById('aov')) document.getElementById('aov').value = data.aov;
    if (document.getElementById('total_purchases')) document.getElementById('total_purchases').value = data.total_purchases;
    if (document.getElementById('recency')) document.getElementById('recency').value = data.recency;
    if (document.getElementById('online')) document.getElementById('online').value = data.online;
    if (document.getElementById('offline')) document.getElementById('offline').value = data.offline;
    if (document.getElementById('visits')) document.getElementById('visits').value = data.visits;

    // Highlight inputs briefly
    const form = document.getElementById('predictionForm');
    if (form) {
        form.classList.add('border-primary');
        setTimeout(() => form.classList.remove('border-primary'), 800);
    }
}

/**
 * Initializes Interactive Charts via AJAX endpoint /api/cluster_data
 */
function initDashboardCharts() {
    const distCanvas = document.getElementById('interactiveDistChart');
    const elbowCanvas = document.getElementById('interactiveElbowChart');

    if (!distCanvas && !elbowCanvas) return;

    fetch('/api/cluster_data')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error("API Error:", data.error);
                return;
            }

            // 1. Cluster Distribution Chart
            if (distCanvas && data.cluster_counts) {
                const labels = data.cluster_counts.map(c => `Cluster ${c.Cluster_ID}: ${c.Segment_Name}`);
                const counts = data.cluster_counts.map(c => c.Customer_Count);
                const colors = ['#4f46e5', '#f59e0b', '#06b6d4', '#10b981', '#64748b', '#ec4899', '#8b5cf6'];

                new Chart(distCanvas, {
                    type: 'doughnut',
                    data: {
                        labels: labels,
                        datasets: [{
                            data: counts,
                            backgroundColor: colors.slice(0, counts.length),
                            borderWidth: 2,
                            borderColor: '#ffffff'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11 } } },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        const count = context.raw || 0;
                                        const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                        const pct = ((count / total) * 100).toFixed(1);
                                        return ` ${count} customers (${pct}%)`;
                                    }
                                }
                            }
                        },
                        cutout: '65%'
                    }
                });
            }

            // 2. Interactive Elbow / Silhouette Curve
            if (elbowCanvas && data.elbow_k && data.elbow_wcss) {
                new Chart(elbowCanvas, {
                    type: 'line',
                    data: {
                        labels: data.elbow_k.map(k => `K=${k}`),
                        datasets: [
                            {
                                label: 'WCSS / Inertia',
                                data: data.elbow_wcss,
                                borderColor: '#4f46e5',
                                backgroundColor: 'rgba(79, 70, 229, 0.1)',
                                tension: 0.3,
                                fill: true,
                                yAxisID: 'y'
                            },
                            {
                                label: 'Silhouette Score',
                                data: data.silhouette_k,
                                borderColor: '#06b6d4',
                                backgroundColor: 'rgba(6, 182, 212, 0.1)',
                                tension: 0.3,
                                borderDash: [5, 5],
                                yAxisID: 'y1'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: { mode: 'index', intersect: false },
                        scales: {
                            y: {
                                type: 'linear',
                                display: true,
                                position: 'left',
                                title: { display: true, text: 'WCSS (Inertia)' }
                            },
                            y1: {
                                type: 'linear',
                                display: true,
                                position: 'right',
                                title: { display: true, text: 'Silhouette Score' },
                                grid: { drawOnChartArea: false }
                            }
                        }
                    }
                });
            }
        })
        .catch(err => console.log('Chart API Error:', err));
}

/**
 * Client-side table search filter
 */
function initTableSearch() {
    const searchInput = document.getElementById('tableSearchInput');
    const table = document.getElementById('searchableTable');

    if (!searchInput || !table) return;

    searchInput.addEventListener('keyup', () => {
        const query = searchInput.value.toLowerCase().trim();
        const rows = table.querySelectorAll('tbody tr');

        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(query) ? '' : 'none';
        });
    });
}
