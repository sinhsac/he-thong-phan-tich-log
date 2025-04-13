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

# Lưu log_history vào tệp sau khi cập nhật
def save_log_history():
    try:
        # Xử lý NaN trước khi lưu vào file
        log_history_clean = [
            {k: (None if pd.isna(v) else v) for k, v in log_item.items()}
            for log_item in log_analyzer.log_history
        ]

        with open("data/log_history.json", "w") as f:
            json.dump(log_history_clean, f)
    except Exception as e:
        logger.error(f"Error saving log history: {e}")

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

        # Lọc log có confidence cao để thêm vào buffer cho fine-tune
        log_records = []
        for record in results["results"]:
            analysis = record["analysis"]
            # Chỉ thêm vào buffer nếu confidence > 0.8 để đảm bảo chất lượng dữ liệu
            if "confidence" in analysis and analysis["confidence"] > 0.8 and "type" in analysis:
                log_records.append(analysis)

        # Thêm vào buffer để fine-tune sau
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

        # Xử lý NaN trước khi chuyển đổi
        log_history_clean = log_history.fillna("")

        # Giới hạn số lượng kết quả
        logs = log_history_clean.tail(limit).to_dict(orient="records")

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


# Thêm trong file log_analyzer_service.py
# Thêm model mới
class ManualLabelBatchInput(BaseModel):
    logs: List[Dict[str, Any]] = Field(..., description="Danh sách log và nhãn mới")


class TriggerFineTuneInput(BaseModel):
    logs: List[Dict[str, Any]] = Field(..., description="Danh sách log để fine-tune")

# Thêm endpoint mới
@app.post("/manual-label", tags=["ML"])
async def manual_label_batch(input_data: ManualLabelBatchInput):
    """Gán nhãn hàng loạt cho log"""
    try:
        global log_buffer

        # Đảm bảo log_history là DataFrame
        if isinstance(log_analyzer.log_history, list):
            # Chuyển đổi list thành DataFrame
            log_df = pd.DataFrame(log_analyzer.log_history)
        else:
            log_df = log_analyzer.log_history

        # Xử lý từng log đầu vào
        for log_input in input_data.logs:
            log_id = log_input.get("log_id")

            # Tìm và cập nhật log
            if isinstance(log_analyzer.log_history, list):
                # Xử lý nếu log_history là list
                for i, log in enumerate(log_analyzer.log_history):
                    if log.get("timestamp") == log_id:
                        log_analyzer.log_history[i]["type"] = log_input["type"]
                        log_analyzer.log_history[i]["confidence"] = float(log_input["confidence"])
                        log_analyzer.log_history[i]["manually_labeled"] = True

                        log_copy = log_analyzer.log_history[i].copy()
                        # Đảm bảo không có giá trị NaN
                        for key, value in log_copy.items():
                            if pd.isna(value):
                                log_copy[key] = None
                        log_buffer.append(log_copy)
                        break
            else:
                # Xử lý nếu log_history là DataFrame
                mask = log_df["timestamp"] == log_id
                if mask.any():
                    log_analyzer.log_history.loc[mask, "type"] = log_input["type"]
                    log_analyzer.log_history.loc[mask, "confidence"] = float(log_input["confidence"])
                    log_analyzer.log_history.loc[mask, "manually_labeled"] = True

                    # Thêm vào buffer
                    log_row = log_analyzer.log_history[mask].to_dict('records')[0]
                    # Đảm bảo không có giá trị NaN
                    for key, value in log_row.items():
                        if pd.isna(value):
                            log_row[key] = None
                    log_buffer.append(log_row)

        # Lưu log_history
        save_log_history()

        return {
            "success": True,
            "message": f"Applied labels to {len(input_data.logs)} logs",
            "total_labeled": len(log_buffer)
        }
    except Exception as e:
        logger.error(f"Error in batch labeling: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trigger-fine-tune", tags=["ML"])
async def trigger_fine_tune_with_data(input_data: TriggerFineTuneInput):
    """Kích hoạt fine-tune với dữ liệu cụ thể"""
    try:
        if not input_data.logs:
            return {
                "success": False,
                "message": "No logs provided"
            }

        # Chuẩn bị dữ liệu
        fine_tune_data = pd.DataFrame([
            {
                "message": log["message"],
                "label": log.get("type", "unknown")
            }
            for log in input_data.logs
            if log.get("type") in log_analyzer.labels  # Chỉ lấy các log có nhãn hợp lệ
        ])

        if fine_tune_data.empty:
            return {
                "success": False,
                "message": "No valid labeled logs found"
            }

        # Thực hiện fine-tune
        result = fine_tuner.fine_tune(fine_tune_data)

        if result:
            return {
                "success": True,
                "message": f"Fine-tuned model with {len(fine_tune_data)} samples"
            }
        else:
            return {
                "success": False,
                "message": "Fine-tuning failed"
            }
    except Exception as e:
        logger.error(f"Error in fine-tuning with provided data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

        # Kiểm tra dữ liệu trong buffer
        if "type" not in df.columns or "message" not in df.columns:
            logger.warning("Logs in buffer don't have required columns for fine-tuning")
            return

        # Kiểm tra phân phối nhãn để đảm bảo chất lượng dữ liệu
        label_counts = df["type"].value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")

        # Chỉ tiếp tục nếu có ít nhất 2 nhãn khác nhau và mỗi nhãn có ít nhất 3 mẫu
        if len(label_counts) < 2 or label_counts.min() < 3:
            logger.warning("Insufficient label distribution for effective fine-tuning")
            return

        # Lưu vào file để backup
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("log_backups", exist_ok=True)
        df.to_csv(f"log_backups/logs_{timestamp}.csv", index=False)

        # Chuẩn bị dữ liệu cho fine-tune
        fine_tune_data = df[["message", "type"]].rename(columns={"type": "label"})

        # Thực hiện fine-tune
        success = fine_tuner.fine_tune(fine_tune_data)

        if success:
            logger.info(f"Fine-tuning completed successfully with {len(fine_tune_data)} samples")
            # Xóa buffer chỉ khi fine-tune thành công
            log_buffer = []
        else:
            logger.warning("Fine-tuning failed, keeping buffer for next attempt")

    except Exception as e:
        logger.error(f"Error processing log buffer: {e}")


# Thiết lập scheduler cho fine-tuning định kỳ
scheduler = BackgroundScheduler()
scheduler.add_job(process_log_buffer, 'interval', minutes=30)
scheduler.start()

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run("log_analyzer_service:app", host="0.0.0.0", port=8000, reload=True)