# log_analyzer_service.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from log_analyzer_core import LogAnalyzerCore
from fine_tuner import FineTuner
from time_forecaster import TimeSeriesForecaster
from apscheduler.schedulers.background import BackgroundScheduler
import uvicorn
import pandas as pd
import datetime

app = FastAPI()

log_analyzer = LogAnalyzerCore()
fine_tuner = FineTuner()
time_forecaster = TimeSeriesForecaster()

log_buffer = []  # Tạm thời lưu log mới để fine-tune

class LogInput(BaseModel):
    logs: List[str]

@app.post("/analyze")
async def analyze_logs(log_input: LogInput):
    global log_buffer
    raw_logs = log_input.logs
    log_df = log_analyzer.preprocess_logs(raw_logs)
    results = log_analyzer.analyze_errors(log_df)

    log_buffer.extend(log_df.to_dict(orient="records"))

    if len(log_buffer) >= 100:
        df = pd.DataFrame(log_buffer)
        fine_tuner.fine_tune(df)
        log_buffer = []

    # Dự đoán xu hướng lỗi trong tương lai
    forecast = time_forecaster.forecast(df=log_analyzer.get_log_history())
    results["trend_forecast"] = forecast

    return results

# Scheduler cho fine-tune theo batch định kỳ (mỗi 30 phút)
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: fine_tuner.fine_tune(pd.DataFrame(log_buffer)), 'interval', minutes=30)
scheduler.start()

if __name__ == "__main__":
    uvicorn.run("log_analyzer_service:app", host="0.0.0.0", port=8000, reload=True)
