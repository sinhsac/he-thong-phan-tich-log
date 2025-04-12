# log_analyzer_core.py

import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import re
import json
from peft import LoraConfig, get_peft_model


class LogAnalyzerCore:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_logbert")
            print("Loaded fine-tuned model")
        except:
            print("Fine-tuned model not found, loading base model...")
            base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_lin", "v_lin"],
                lora_dropout=0.1,
                bias="none"
            )
            self.model = get_peft_model(base_model, lora_config)
            print("Loaded base model with LoRA config")

        self.label_encoder = LabelEncoder()
        self.labels = ["database", "server", "network", "application", "security"]
        self.label_encoder.fit(self.labels)

        self.log_history = []  # Dùng cho realtime monitor và fine-tune sau
        self.error_patterns = self._compile_error_patterns()

    def _compile_error_patterns(self):
        """Compile regex patterns để phát hiện các loại lỗi phổ biến"""
        return {
            "database": [
                re.compile(r"database\s+connection\s+(?:failed|error|refused)", re.I),
                re.compile(r"sql\s+(?:error|exception)", re.I),
                re.compile(r"(?:mysql|postgresql|oracle|mongodb)\s+error", re.I)
            ],
            "server": [
                re.compile(r"server\s+(?:error|down|crashed|unavailable)", re.I),
                re.compile(r"out\s+of\s+(?:memory|disk\s+space)", re.I),
                re.compile(r"cpu\s+usage\s+(?:high|excessive)", re.I)
            ],
            "network": [
                re.compile(r"network\s+(?:error|failure|timeout)", re.I),
                re.compile(r"connection\s+(?:refused|timed\s+out|reset)", re.I),
                re.compile(r"dns\s+(?:error|failure|lookup\s+failed)", re.I)
            ],
            "application": [
                re.compile(r"application\s+(?:error|crashed|failed)", re.I),
                re.compile(r"exception\s+(?:occurred|thrown|caught)", re.I),
                re.compile(r"(?:null\s+pointer|index\s+out\s+of\s+bounds)", re.I)
            ],
            "security": [
                re.compile(r"(?:unauthorized|forbidden)\s+access", re.I),
                re.compile(r"(?:sql\s+injection|xss|csrf)", re.I),
                re.compile(r"authentication\s+(?:failed|error)", re.I)
            ]
        }

    def preprocess_log(self, log_text):
        """Xử lý một log dạng text thành dạng có cấu trúc"""
        # Phát hiện mức độ nghiêm trọng
        level = "info"
        if re.search(r'\b(error|fail(ed|ure)?)\b', log_text, re.I):
            level = "error"
        elif re.search(r'\b(critical|fatal|alert)\b', log_text, re.I):
            level = "critical"
        elif re.search(r'\b(warn(ing)?)\b', log_text, re.I):
            level = "warning"

        # Phát hiện service
        service = "unknown"
        if re.search(r'\b(web|http|rest|api)\b', log_text, re.I):
            service = "web"
        elif re.search(r'\b(db|database|sql|query)\b', log_text, re.I):
            service = "database"
        elif re.search(r'\b(auth|login|user)\b', log_text, re.I):
            service = "auth"

        # Phát hiện mã lỗi
        error_code_match = re.search(r'(?:error|code)[\s:]+([A-Z0-9_-]+)', log_text, re.I)
        error_code = error_code_match.group(1) if error_code_match else "UNKNOWN"

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "service": service,
            "message": log_text.strip(),
            "error_code": error_code
        }

    def preprocess_logs(self, logs):
        """Xử lý nhiều log dạng text thành một DataFrame"""
        processed_logs = []

        for log in logs:
            processed_log = self.preprocess_log(log)
            processed_logs.append(processed_log)

        return pd.DataFrame(processed_logs)

    def analyze_log(self, structured_log):
        """Phân tích một log đã được tiền xử lý"""
        message = structured_log["message"]
        inputs = self.tokenizer(message, return_tensors="pt", truncation=True, padding=True)

        # Run model inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get predicted class and probabilities
        logits = outputs.logits[0]
        probs = torch.nn.functional.softmax(logits, dim=0)
        predicted_class = torch.argmax(logits).item()
        confidence = probs[predicted_class].item()

        # Fallback to rule-based if confidence is low
        error_type = self.label_encoder.inverse_transform([predicted_class])[0]
        rule_based_type = self._rule_based_classification(message)

        # If model has low confidence, use rule-based result
        if confidence < 0.6 and rule_based_type != "unknown":
            error_type = rule_based_type

        # Add classification to structured log
        structured_log["type"] = error_type
        structured_log["confidence"] = round(confidence, 4)

        # Add to history
        self.log_history.append(structured_log)

        return structured_log, self.suggest_solutions(error_type, message)

    def _rule_based_classification(self, message):
        """Phân loại lỗi dựa trên các regex mẫu"""
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern.search(message):
                    return error_type
        return "unknown"

    def analyze_errors(self, log_df):
        """Phân tích nhiều log được lưu trong DataFrame"""
        results = []

        for _, row in log_df.iterrows():
            structured_log = row.to_dict()
            analysis, solutions = self.analyze_log(structured_log)
            results.append({
                "analysis": analysis,
                "solutions": solutions
            })

        # Group logs by error type
        error_types = log_df["type"].value_counts().to_dict() if "type" in log_df.columns else {}

        return {
            "results": results,
            "summary": {
                "total_logs": len(log_df),
                "error_types": error_types,
                "critical_errors": len(log_df[log_df["level"] == "critical"]) if "level" in log_df.columns else 0
            }
        }

    def suggest_solutions(self, error_type, message):
        """Gợi ý các phương pháp xử lý theo loại lỗi"""
        solutions = []

        if error_type == "database":
            solutions = [
                "Kiểm tra kết nối database và credentials",
                "Xem lại query đang thực thi",
                "Kiểm tra giới hạn kết nối và tài nguyên database server",
                "Xác minh quyền truy cập của user đến database và bảng"
            ]

            if "timeout" in message.lower():
                solutions.append("Tối ưu hóa query để giảm thời gian thực thi")
            if "duplicate" in message.lower():
                solutions.append("Kiểm tra ràng buộc unique và xử lý duplicate key")

        elif error_type == "server":
            solutions = [
                "Kiểm tra tài nguyên server (CPU, RAM, disk)",
                "Xem lại cấu hình service",
                "Kiểm tra log hệ thống để phát hiện sự cố phần cứng",
                "Xem xét tăng tài nguyên hoặc scale hệ thống"
            ]

            if "restart" in message.lower():
                solutions.append("Xác minh quy trình khởi động lại service đúng cách")

        elif error_type == "network":
            solutions = [
                "Kiểm tra kết nối mạng, DNS và firewall",
                "Xác minh cấu hình network interface",
                "Kiểm tra latency và packet loss",
                "Xem xét vấn đề DNS caching"
            ]

        elif error_type == "application":
            solutions = [
                "Kiểm tra mã nguồn và exception handling",
                "Xem lại các thay đổi gần đây trong code",
                "Kiểm tra version compatibility giữa các dependency",
                "Xem xét rollback về phiên bản ổn định trước đó"
            ]

        elif error_type == "security":
            solutions = [
                "Kiểm tra cấu hình bảo mật và nhật ký truy cập",
                "Xem xét các mẫu tấn công phổ biến",
                "Cập nhật các biện pháp bảo mật và firewall rules",
                "Kiểm tra quyền truy cập file và thư mục"
            ]

        # Thêm gợi ý chung
        solutions.append("Tìm kiếm mã lỗi này trong knowledge base")

        return solutions

    def get_log_history(self):
        """Trả về lịch sử log dưới dạng DataFrame"""
        return pd.DataFrame(self.log_history)

    def predict_trend(self):
        """Dự đoán xu hướng lỗi trong thời gian ngắn hạn"""
        df = self.get_log_history()
        if len(df) < 10:  # Cần ít nhất 10 log để dự đoán
            return {
                "time_window": "15-30 minutes",
                "probability": 0.0,
                "likely_errors": []
            }

        # Lấy 50 log gần nhất để phân tích
        recent_logs = df.tail(50)

        # Tính tỷ lệ lỗi nghiêm trọng
        error_levels = ["error", "critical"]
        error_rate = len(recent_logs[recent_logs.level.isin(error_levels)]) / len(recent_logs)

        # Tìm các loại lỗi phổ biến nhất
        if "type" in recent_logs.columns:
            likely_types = recent_logs["type"].value_counts().head(2).index.tolist()
        else:
            likely_types = []

        # Phát hiện xu hướng tăng
        is_increasing = False
        if len(df) >= 20:
            recent_20 = df.tail(20)
            recent_10 = df.tail(10)
            rate_20 = len(recent_20[recent_20.level.isin(error_levels)]) / 20
            rate_10 = len(recent_10[recent_10.level.isin(error_levels)]) / 10
            is_increasing = rate_10 > rate_20

        return {
            "time_window": "15-30 minutes",
            "probability": min(1.0, error_rate * 2.5),
            "is_increasing": is_increasing,
            "likely_errors": likely_types
        }

    def export_for_fine_tuning(self):
        """Xuất dữ liệu để fine-tune mô hình"""
        df = self.get_log_history()
        if df.empty or "type" not in df.columns:
            return None

        # Chỉ lấy các log đã được phân loại thuộc vào các nhãn đã biết
        valid_logs = df[df["type"].isin(self.labels)]

        if len(valid_logs) < 5:  # Cần ít nhất 5 log để fine-tune
            return None

        # Chuẩn bị dữ liệu để fine-tune
        fine_tune_data = {
            "message": valid_logs["message"].tolist(),
            "label": valid_logs["type"].tolist()
        }

        return pd.DataFrame(fine_tune_data)


class LogDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)