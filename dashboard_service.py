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

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_log/dashboard.log"),
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
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
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
        .log-message {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 300px;
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

        <div class="row mt-3">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Manual Labeling</h5>
                    </div>
                    <div class="card-body">
                        <div class="alert alert-info">
                            Select logs to manually label for improving model accuracy
                        </div>
                        <div class="table-responsive">
                            <table class="table table-striped table-hover">
                                <thead>
                                    <tr>
                                        <th>Select</th>
                                        <th>Timestamp</th>
                                        <th>Level</th>
                                        <th>Current Type</th>
                                        <th>Message</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody id="labelingTable">
                                    <!-- Logs for labeling will be loaded here -->
                                </tbody>
                            </table>
                        </div>
                        <div class="row mt-3">
                            <div class="col-md-4">
                                <select id="labelType" class="form-select">
                                    <option value="">Select error type...</option>
                                    <option value="database">Database</option>
                                    <option value="server">Server</option>
                                    <option value="network">Network</option>
                                    <option value="application">Application</option>
                                    <option value="security">Security</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <select id="confidenceLevel" class="form-select">
                                    <option value="1.0">High Confidence (1.0)</option>
                                    <option value="0.8">Medium Confidence (0.8)</option>
                                    <option value="0.6">Low Confidence (0.6)</option>
                                </select>
                            </div>
                            <div class="col-md-4">
                                <button id="applyLabelsBtn" class="btn btn-primary">Apply Labels</button>
                                <button id="exportForTrainingBtn" class="btn btn-success ms-2">Export for Training</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal xác nhận -->
    <div class="modal fade" id="confirmExportModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Confirm Export</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p>Are you sure you want to export <span id="exportCount">0</span> labeled logs for training?</p>
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="includeUnlabeled" checked>
                        <label class="form-check-label" for="includeUnlabeled">
                            Include unlabeled logs (for contrastive learning)
                        </label>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" id="confirmExportBtn">Export</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal kết quả -->
    <div class="modal fade" id="exportResultModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Export Result</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body" id="exportResultBody">
                    <!-- Result message will be shown here -->
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-primary" data-bs-dismiss="modal">OK</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Biến lưu trữ log data
        let allLogsData = [];
        let selectedLogs = [];

        // Hàm cập nhật bảng gán nhãn
        function updateLabelingTable(logs) {
            const tableBody = document.getElementById('labelingTable');
            tableBody.innerHTML = '';

            logs.forEach(log => {
                const row = document.createElement('tr');

                // Ô checkbox
                const selectCell = document.createElement('td');
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.className = 'form-check-input log-checkbox';
                checkbox.dataset.logId = log.timestamp;
                checkbox.addEventListener('change', function() {
                    if (this.checked) {
                        selectedLogs.push(log.timestamp);
                    } else {
                        selectedLogs = selectedLogs.filter(id => id !== log.timestamp);
                    }
                });
                selectCell.appendChild(checkbox);

                // Ô timestamp
                const timestampCell = document.createElement('td');
                timestampCell.textContent = log.timestamp;

                // Ô level
                const levelCell = document.createElement('td');
                const levelBadge = document.createElement('span');
                levelBadge.className = 'badge ' + 
                    (log.level === 'critical' ? 'bg-danger' : 
                     log.level === 'error' ? 'bg-warning' : 
                     log.level === 'warning' ? 'bg-info' : 'bg-success');
                levelBadge.textContent = log.level;
                levelCell.appendChild(levelBadge);

                // Ô current type
                const typeCell = document.createElement('td');
                typeCell.textContent = log.type || 'Not classified';

                // Ô message (rút gọn)
                const messageCell = document.createElement('td');
                messageCell.className = 'log-message';
                messageCell.textContent = log.message;
                messageCell.title = log.message;

                // Ô actions
                const actionsCell = document.createElement('td');
                const viewBtn = document.createElement('button');
                viewBtn.className = 'btn btn-sm btn-outline-primary';
                viewBtn.textContent = 'View';
                viewBtn.addEventListener('click', () => showLogDetails(log));
                actionsCell.appendChild(viewBtn);

                row.appendChild(selectCell);
                row.appendChild(timestampCell);
                row.appendChild(levelCell);
                row.appendChild(typeCell);
                row.appendChild(messageCell);
                row.appendChild(actionsCell);

                tableBody.appendChild(row);
            });
        }

        // Hàm hiển thị chi tiết log
        function showLogDetails(log) {
            alert(`Log Details:\n\nTimestamp: ${log.timestamp}\nLevel: ${log.level}\nType: ${log.type || 'N/A'}\nMessage: ${log.message}`);
        }

        // Hàm áp dụng nhãn
        function applyLabels() {
            const labelType = document.getElementById('labelType').value;
            const confidence = parseFloat(document.getElementById('confidenceLevel').value);

            if (!labelType) {
                alert('Please select an error type');
                return;
            }

            if (selectedLogs.length === 0) {
                alert('Please select at least one log');
                return;
            }

            const logsToLabel = allLogsData.filter(log => selectedLogs.includes(log.timestamp));

            fetch('/api/manual-label', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    logs: logsToLabel.map(log => ({
                        log_id: log.timestamp,
                        type: labelType,
                        confidence: confidence
                    }))
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Labels applied successfully!');
                    loadDashboardData(); // Refresh data
                } else {
                    alert('Error applying labels: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Error applying labels:', error);
                alert('Error applying labels');
            });
        }

        // Hàm xuất dữ liệu để training
        function exportForTraining() {
            const includeUnlabeled = document.getElementById('includeUnlabeled').checked;
            const logsToExport = includeUnlabeled ? allLogsData : 
                allLogsData.filter(log => log.type && log.confidence > 0.6);

            document.getElementById('exportCount').textContent = logsToExport.length;

            // Hiển thị modal xác nhận
            const modal = new bootstrap.Modal(document.getElementById('confirmExportModal'));
            modal.show();
        }

        // Xác nhận xuất dữ liệu
        function confirmExport() {
            const includeUnlabeled = document.getElementById('includeUnlabeled').checked;
            const logsToExport = includeUnlabeled ? allLogsData : 
                allLogsData.filter(log => log.type && log.confidence > 0.6);

            fetch('/api/trigger-fine-tune', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    logs: logsToExport
                })
            })
            .then(response => response.json())
            .then(data => {
                const modal = new bootstrap.Modal(document.getElementById('confirmExportModal'));
                modal.hide();

                const resultModal = new bootstrap.Modal(document.getElementById('exportResultModal'));
                const resultBody = document.getElementById('exportResultBody');

                if (data.success) {
                    resultBody.innerHTML = `
                        <div class="alert alert-success">
                            <strong>Success!</strong> ${data.message}
                        </div>
                        <p>Total logs exported: ${logsToExport.length}</p>
                    `;
                } else {
                    resultBody.innerHTML = `
                        <div class="alert alert-danger">
                            <strong>Error!</strong> ${data.message}
                        </div>
                    `;
                }

                resultModal.show();
            })
            .catch(error => {
                console.error('Error exporting for training:', error);
                alert('Error exporting for training');
            });
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

        // Cập nhật hàm loadDashboardData để lưu trữ allLogsData
        function loadDashboardData() {
            fetch('/api/dashboard-data')
                .then(response => response.json())
                .then(data => {
                    updateLogSummary(data.log_summary);
                    updateErrorDistribution(data.error_distribution);
                    updateForecast(data.forecast);
                    updateRecentLogs(data.recent_logs);

                    // Lưu trữ logs để sử dụng cho gán nhãn
                    allLogsData = data.recent_logs;
                    updateLabelingTable(data.recent_logs);
                })
                .catch(error => console.error('Error loading dashboard data:', error));
        }

        // Thêm event listeners khi DOM loaded
        document.addEventListener('DOMContentLoaded', function() {
            loadDashboardData();
            setInterval(loadDashboardData, 30000);

            // Gán sự kiện cho các nút
            document.getElementById('applyLabelsBtn').addEventListener('click', applyLabels);
            document.getElementById('exportForTrainingBtn').addEventListener('click', exportForTraining);
            document.getElementById('confirmExportBtn').addEventListener('click', confirmExport);
        });
    </script>
</body>
</html>
"""

# Tạo HTML template ban đầu nếu chưa tồn tại
template_path = os.path.join("templates", "dashboard.html")
if not os.path.exists(template_path):
    with open(template_path, "w", encoding="utf-8") as f:
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


@app.post("/api/manual-label", tags=["API"])
async def manual_label_logs(request: Request):
    """
    Gán nhãn thủ công cho nhiều log cùng lúc
    """
    try:
        data = await request.json()
        if not data.get("logs"):
            raise HTTPException(status_code=400, detail="No logs provided")

        # Gửi yêu cầu đến log analyzer service
        response = requests.post(
            f"{LOG_ANALYZER_API}/manual-label",
            json={"logs": data["logs"]}
        )
        response.raise_for_status()

        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending manual label request: {e}")
        raise HTTPException(status_code=503, detail="Log analyzer service unavailable")
    except Exception as e:
        logger.error(f"Error processing manual labels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trigger-fine-tune", tags=["API"])
async def trigger_fine_tune(request: Request):
    """
    Kích hoạt quá trình fine-tune với dữ liệu đã gán nhãn
    """
    try:
        data = await request.json()
        if not data.get("logs"):
            raise HTTPException(status_code=400, detail="No logs provided")

        # Gửi yêu cầu đến log analyzer service
        response = requests.post(
            f"{LOG_ANALYZER_API}/trigger-fine-tune",
            json={"logs": data["logs"]}
        )
        response.raise_for_status()

        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending fine-tune request: {e}")
        raise HTTPException(status_code=503, detail="Log analyzer service unavailable")
    except Exception as e:
        logger.error(f"Error triggering fine-tune: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

    uvicorn.run("dashboard_service:app", host="0.0.0.0", port=8002, reload=True)