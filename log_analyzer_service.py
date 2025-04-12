# log_analyzer_service.py

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from log_analyzer_core import LogAnalyzerCore
from fine_tuner import FineTuner
from time_forecaster import TimeSeriesForecaster
from apscheduler.schedulers.background import BackgroundScheduler
import uvicorn
import pandas as pd
import datetime
import logging
import os
import json

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_log/service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LogAnalyzerService")

# Khởi tạo FastAPI
app = FastAPI(
    title="Log Analyzer Service",
    description="API để phân tích log, phát hiện lỗi, gợi ý giải pháp và dự đoán xu hướng",
    version="1.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo các thành phần
log_analyzer = LogAnalyzerCore()
fine_tuner = FineTuner()
time_forecaster = TimeSeriesForecaster()

# Buffer để gom log cho fine-tune định kỳ
log_buffer = []


# Định nghĩa các model cho API
class LogInput(BaseModel):
    logs: List[str] = Field(..., description="Danh sách các log dạng text để phân tích")


class LogQuery(BaseModel):
    query: str = Field(..., description="Truy vấn để tìm kiếm trong lịch sử log")
    limit: int = Field(50, description="Số lượng kết quả tối đa trả về")


class ForecastInput(BaseModel):
    periods: int = Field(30, description="Số phút muốn dự đoán trong tương lai")


# API endpoints
@app.post("/analyze", tags=["Analysis"])
async def analyze_logs(log_input: LogInput, background_tasks: BackgroundTasks):
    """
    Phân tích danh sách log, phát hiện loại lỗi và gợi ý giải pháp
    """
    global log_buffer

    if not log_input.logs:
        raise HTTPException(status_code=400, detail="Không có log nào được cung cấp")

    try:
        # Tiền xử lý log
        log_df = log_analyzer.preprocess_logs(log_input.logs)

        # Phân tích lỗi
        results = log_analyzer.analyze_errors(log_df)

        # Thêm vào buffer để fine-tune sau
        log_records = log_df.to_dict(orient="records")
        log_buffer.extend(log_records)

        # Dự đoán xu hướng lỗi trong tương lai
        trend_prediction = log_analyzer.predict_trend()
        results["trend_prediction"] = trend_prediction

        # Thêm task fine-tune vào background nếu đủ dữ liệu
        if len(log_buffer) >= 100:
            background_tasks.add_task(process_log_buffer)

        return results

    except Exception as e:
        logger.error(f"Error analyzing logs: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi phân tích log: {str(e)}")


@app.post("/forecast", tags=["Forecasting"])
async def forecast_errors(forecast_input: ForecastInput):
    """
    Dự đoán xu hướng lỗi trong tương lai dựa trên dữ liệu lịch sử
    """
    try:
        # Lấy lịch sử log
        log_history = log_analyzer.get_log_history()

        if log_history.empty:
            return {
                "success": False,
                "message": "Không đủ dữ liệu lịch sử để dự đoán"
            }

        # Dự đoán xu hướng
        forecast = time_forecaster.forecast(df=log_history, periods=forecast_input.periods)
        return forecast

    except Exception as e:
        logger.error(f"Error forecasting errors: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán xu hướng: {str(e)}")


@app.get("/history", tags=["History"])
async def get_log_history(limit: int = 100):
    """
    Lấy lịch sử log đã phân tích
    """
    try:
        log_history = log_analyzer.get_log_history()

        if log_history.empty:
            return {"logs": [], "count": 0}

        # Giới hạn số lượng kết quả
        logs = log_history.tail(limit).to_dict(orient="records")

        return {
            "logs": logs,
            "count": len(logs),
            "total": len(log_history)
        }

    except Exception as e:
        logger.error(f"Error retrieving log history: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy lịch sử log: {str(e)}")


@app.post("/search", tags=["History"])
async def search_logs(query: LogQuery):
    """
    Tìm kiếm trong lịch sử log
    """
    try:
        log_history = log_analyzer.get_log_history()

        if log_history.empty:
            return {"logs": [], "count": 0}

        # Tìm kiếm trong cột message
        if 'message' in log_history.columns:
            mask = log_history['message'].str.contains(query.query, case=False, na=False)
            results = log_history[mask].tail(query.limit).to_dict(orient="records")

            return {
                "logs": results,
                "count": len(results),
                "query": query.query
            }
        else:
            return {"logs": [], "count": 0, "error": "Không tìm thấy cột message trong lịch sử log"}

    except Exception as e:
        logger.error(f"Error searching logs: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm log: {str(e)}")


@app.post("/finetune", tags=["ML"])
async def trigger_fine_tuning():
    """
    Kích hoạt quá trình fine-tune mô hình với dữ liệu đã thu thập
    """
    try:
        # Export dữ liệu để fine-tune
        fine_tune_data = log_analyzer.export_for_fine_tuning()

        if fine_tune_data is None or fine_tune_data.empty:
            return {
                "success": False,
                "message": "Không đủ dữ liệu để fine-tune mô hình"
            }

        # Thực hiện fine-tune
        result = fine_tuner.fine_tune(fine_tune_data)

        if result:
            return {
                "success": True,
                "message": f"Fine-tune thành công với {len(fine_tune_data)} log"
            }
        else:
            return {
                "success": False,
                "message": "Fine-tune không thành công"
            }

    except Exception as e:
        logger.error(f"Error triggering fine-tuning: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi fine-tune mô hình: {str(e)}")


@app.get("/status", tags=["System"])
async def get_system_status():
    """
    Lấy trạng thái của hệ thống
    """
    try:
        log_history = log_analyzer.get_log_history()

        return {
            "status": "running",
            "logs_analyzed": len(log_history),
            "logs_in_buffer": len(log_buffer),
            "forecaster_status": "trained" if time_forecaster.is_trained else "not_trained",
            "components": {
                "log_analyzer": "active",
                "fine_tuner": "active",
                "time_forecaster": "active"
            },
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


# Hàm xử lý buffer log (gọi trong background)
def process_log_buffer():
    global log_buffer

    try:
        if len(log_buffer) < 10:
            logger.info(f"Not enough logs in buffer for fine-tuning ({len(log_buffer)} < 10)")
            return

        logger.info(f"Processing {len(log_buffer)} logs for fine-tuning")

        # Tạo DataFrame từ buffer
        df = pd.DataFrame(log_buffer)

        # Lưu vào file để backup
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("log_backups", exist_ok=True)
        df.to_csv(f"log_backups/logs_{timestamp}.csv", index=False)

        # Chuẩn bị dữ liệu cho fine-tune
        if "type" in df.columns and "message" in df.columns:
            fine_tune_data = df[["message", "type"]].rename(columns={"type": "label"})
            fine_tuner.fine_tune(fine_tune_data)
        else:
            logger.warning("Logs in buffer don't have required columns for fine-tuning")

        # Xóa buffer
        log_buffer = []
        logger.info("Fine-tuning completed and buffer cleared")

    except Exception as e:
        logger.error(f"Error processing log buffer: {e}")


# Thiết lập scheduler cho fine-tuning định kỳ
scheduler = BackgroundScheduler()
scheduler.add_job(process_log_buffer, 'interval', minutes=30)
scheduler.start()

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run("log_analyzer_service:app", host="0.0.0.0", port=8000, reload=True)