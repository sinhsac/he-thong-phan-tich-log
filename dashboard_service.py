# dashboard_service.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import os
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DashboardService")

# Khởi tạo FastAPI app
app = FastAPI(
    title="Log Analyzer Dashboard",
    description="Dashboard để hiển thị kết quả phân tích log và dự đoán xu hướng",
    version="1.0.0"
)

# Địa chỉ API của log analyzer service
LOG_ANALYZER_API = "http://localhost:8000"

# Tạo thư mục static nếu chưa tồn tại
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Thiết lập đường dẫn cho static files và templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Tạo file HTML template mặc định
DEFAULT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Log Analyzer Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .card {
            margin-bottom: 20px;
        }
        .error-badge {
            font-size: 0.8em;
            margin-right: 5px;
        }
        .dashboard-header {
            background-color: #343a40;
            color: white;
            padding: 15px 0;
            margin-bottom: 20px;
        }
        .chart-container {
            height: 300px;
        }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <div class="container">
            <h1>Log Analyzer Dashboard</h1>
            <p>Real-time monitoring and analytics for system logs</p>
        </div>
    </div>

    <div class="container">
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Log Summary</h5>
                    </div>
                    <div class="card-body">
                        <div id="logSummaryChart" class="chart-container"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Error Distribution</h5>
                    </div>
                    <div class="card-body">
                        <div id="errorDistributionChart" class="chart-container"></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Error Forecast</h5>
                    </div>
                    <div class="card-body">
                        <div id="forecastChart" class="chart-container"></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Recent Logs</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-striped table-hover">
                                <thead>
                                    <tr>
                                        <th>Timestamp</th>
                                        <th>Level</th>
                                        <th>Type</th>
                                        <th>Message</th>
                                    </tr>
                                </thead>
                                <tbody id="recentLogsTable">
                                    <!-- Recent logs will be loaded here -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Load dashboard data
        function loadDashboardData() {
            fetch('/api/dashboard-data')
                .then(response => response.json())
                .then(data => {
                    updateLogSummary(data.log_summary);
                    updateErrorDistribution(data.error_distribution);
                    updateForecast(data.forecast);
                    updateRecentLogs(data.recent_logs);
                })
                .catch(error => console.error('Error loading dashboard data:', error));
        }

        // Update log summary chart
        function updateLogSummary(data) {
            const ctx = document.getElementById('logSummaryChart');

            const layout = {
                title: 'Log Levels Distribution',
                height: 300,
                margin: { t: 30, b: 40, l: 60, r: 40 }
            };

            const chartData = [{
                values: data.values,
                labels: data.labels,
                type: 'pie',
                textinfo: 'label+percent',
                hole: 0.4
            }];

            Plotly.newPlot('logSummaryChart', chartData, layout);
        }

        // Update error distribution chart
        function updateErrorDistribution(data) {
            const layout = {
                title: 'Error Types Distribution',
                height: 300,
                margin: { t: 30, b: 60, l: 60, r: 40 }
            };

            const chartData = [{
                x: data.labels,
                y: data.values,
                type: 'bar',
                marker: {
                    color: 'rgba(50, 171, 96, 0.7)'
                }
            }];

            Plotly.newPlot('errorDistributionChart', chartData, layout);
        }

        // Update forecast chart
        function updateForecast(data) {
            const layout = {
                title: 'Error Forecast (Next 30 Minutes)',
                height: 300,
                margin: { t: 30, b: 40, l: 60, r: 40 },
                xaxis: {
                    title: 'Time'
                },
                yaxis: {
                    title: 'Predicted Errors'
                }
            };

            const chartData = [{
                x: data.x,
                y: data.y,
                type: 'scatter',
                mode: 'lines',
                name: 'Predicted',
                line: {
                    color: 'rgb(31, 119, 180)',
                    width: 2
                }
            }, {
                x: data.x,
                y: data.y_upper,
                type: 'scatter',
                mode: 'lines',
                name: 'Upper Bound',
                line: {
                    width: 0
                },
                marker: {
                    color: 'rgba(31, 119, 180, 0)'
                },
                showlegend: false
            }, {
                x: data.x,
                y: data.y_lower,
                type: 'scatter',
                mode: 'lines',
                name: 'Lower Bound',
                fill: 'tonexty',
                fillcolor: 'rgba(31, 119, 180, 0.2)',
                line: {
                    width: 0
                },
                marker: {
                    color: 'rgba(31, 119, 180, 0)'
                },
                showlegend: false
            }];

            Plotly.newPlot('forecastChart', chartData, layout);
        }

        // Update recent logs table
        function updateRecentLogs(logs) {
            const tableBody = document.getElementById('recentLogsTable');
            tableBody.innerHTML = '';

            logs.forEach(log => {
                const row = document.createElement('tr');

                // Determine row color based on log level
                if (log.level === 'critical') {
                    row.classList.add('table-danger');
                } else if (log.level === 'error') {
                    row.classList.add('table-warning');
                }

                const timestamp = document.createElement('td');
                timestamp.textContent = log.timestamp;

                const level = document.createElement('td');
                const levelBadge = document.createElement('span');
                levelBadge.classList.add('badge', 'error-badge');

                if (log.level === 'critical') {
                    levelBadge.classList.add('bg-danger');
                } else if (log.level === 'error') {
                    levelBadge.classList.add('bg-warning');
                    levelBadge.style.color = 'black';
                } else if (log.level === 'warning') {
                    levelBadge.classList.add('bg-info');
                } else {
                    levelBadge.classList.add('bg-success');
                }

                levelBadge.textContent = log.level;
                level.appendChild(levelBadge);

                const type = document.createElement('td');
                type.textContent = log.type || '-';

                const message = document.createElement('td');
                message.textContent = log.message;

                row.appendChild(timestamp);
                row.appendChild(level);
                row.appendChild(type);
                row.appendChild(message);

                tableBody.appendChild(row);
            });
        }

        // Load initial data
        document.addEventListener('DOMContentLoaded', function() {
            loadDashboardData();

            // Refresh data every 30 seconds
            setInterval(loadDashboardData, 30000);
        });
    </script>
</body>
</html>
"""

# Tạo HTML template ban đầu nếu chưa tồn tại
template_path = os.path.join("templates", "dashboard.html")
if not os.path.exists(template_path):
    with open(template_path, "w") as f:
        f.write(DEFAULT_TEMPLATE)


# Routes
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def get_dashboard(request: Request):
    """
    Hiển thị dashboard tổng quan
    """
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/dashboard-data", tags=["API"])
async def get_dashboard_data():
    """
    Lấy dữ liệu cho dashboard
    """
    try:
        # Lấy lịch sử log
        response = requests.get(f"{LOG_ANALYZER_API}/history", params={"limit": 100})
        response.raise_for_status()
        log_history = response.json()

        # Lấy dự đoán
        forecast_response = requests.post(f"{LOG_ANALYZER_API}/forecast", json={"periods": 30})
        forecast_response.raise_for_status()
        forecast = forecast_response.json()

        # Chuẩn bị dữ liệu cho dashboard
        dashboard_data = prepare_dashboard_data(log_history, forecast)

        return dashboard_data

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from log analyzer API: {e}")
        raise HTTPException(status_code=503, detail="Log analyzer service unavailable")
    except Exception as e:
        logger.error(f"Error preparing dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Error preparing dashboard data: {str(e)}")


def prepare_dashboard_data(log_history, forecast):
    """
    Chuẩn bị dữ liệu cho dashboard
    """
    # Dữ liệu log summary
    if 'logs' in log_history and log_history['logs']:
        log_df = pd.DataFrame(log_history['logs'])

        # Log levels distribution
        level_counts = log_df['level'].value_counts()
        log_summary = {
            'labels': level_counts.index.tolist(),
            'values': level_counts.values.tolist()
        }

        # Error types distribution
        if 'type' in log_df.columns:
            error_df = log_df[log_df['level'].isin(['error', 'critical'])]
            type_counts = error_df['type'].value_counts()
            error_distribution = {
                'labels': type_counts.index.tolist(),
                'values': type_counts.values.tolist()
            }
        else:
            error_distribution = {
                'labels': [],
                'values': []
            }

        # Recent logs
        recent_logs = log_df.sort_values('timestamp', ascending=False).head(20).to_dict('records')
    else:
        log_summary = {
            'labels': ['No Data'],
            'values': [1]
        }
        error_distribution = {
            'labels': [],
            'values': []
        }
        recent_logs = []

    # Dữ liệu dự đoán
    if 'success' in forecast and forecast['success'] and 'forecast_data' in forecast:
        forecast_data = forecast['forecast_data']
        forecast_df = pd.DataFrame(forecast_data)

        forecast_chart = {
            'x': forecast_df['ds'].tolist(),
            'y': forecast_df['yhat'].tolist(),
            'y_upper': forecast_df['yhat_upper'].tolist(),
            'y_lower': forecast_df['yhat_lower'].tolist()
        }
    else:
        # Tạo dữ liệu giả nếu không có dự đoán
        now = datetime.now()
        times = [(now + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(30)]

        forecast_chart = {
            'x': times,
            'y': [0] * 30,
            'y_upper': [0] * 30,
            'y_lower': [0] * 30
        }

    return {
        'log_summary': log_summary,
        'error_distribution': error_distribution,
        'forecast': forecast_chart,
        'recent_logs': recent_logs
    }


# Chạy ứng dụng
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dashboard_service:app", host="0.0.0.0", port=8001, reload=True)