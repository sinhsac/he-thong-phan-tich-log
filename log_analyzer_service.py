from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from log_analyzer_core import LogAnalyzerCore
from fine_tuner import FineTuner
from time_forecaster import TimeSeriesForecaster
from apscheduler.schedulers.background import BackgroundScheduler
import uvicorn
import pandas as pd
import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize components
log_analyzer = LogAnalyzerCore(max_log_history=5000)
fine_tuner = FineTuner()
time_forecaster = TimeSeriesForecaster()

# Log buffer with size limit
LOG_BUFFER_MAX_SIZE = 1000
log_buffer = []


class LogInput(BaseModel):
    logs: List[str]
    priority: Optional[int] = 0


@app.post("/analyze")
async def analyze_logs(log_input: LogInput):
    global log_buffer

    try:
        # Preprocess and analyze logs
        processed_logs = []
        for log in log_input.logs:
            processed = log_analyzer.preprocess_log(log)
            if processed:
                analyzed_log, solutions = log_analyzer.analyze_log(processed)
                if analyzed_log:
                    processed_logs.append({
                        "log": analyzed_log,
                        "solutions": solutions
                    })
                    log_buffer.append(analyzed_log)

        # Check if we should fine-tune
        if len(log_buffer) >= 100 and len(log_buffer) % 100 == 0:
            await schedule_fine_tuning()

        # Get trend forecast
        forecast = {}
        try:
            forecast = time_forecaster.forecast(df=log_analyzer.get_log_history())
        except Exception as e:
            logger.error(f"Error in forecasting: {e}")

        return {
            "results": processed_logs,
            "trend_forecast": forecast,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error processing logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def schedule_fine_tuning():
    global log_buffer
    try:
        if not log_buffer:
            return

        df = pd.DataFrame(log_buffer)
        success = fine_tuner.fine_tune(df)
        if success:
            log_buffer = []  # Clear buffer only if fine-tuning succeeded
            logger.info("Fine-tuning completed successfully")
        else:
            logger.warning("Fine-tuning failed, keeping logs in buffer")
    except Exception as e:
        logger.error(f"Error during scheduled fine-tuning: {e}")


# Configure scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    schedule_fine_tuning,
    'interval',
    minutes=30,
    next_run_time=datetime.datetime.now() + datetime.timedelta(minutes=30)
)
scheduler.start()


@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()


if __name__ == "__main__":
    uvicorn.run(
        "log_analyzer_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )