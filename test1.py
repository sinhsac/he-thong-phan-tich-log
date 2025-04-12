import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import torch
from peft import LoraConfig, get_peft_model
import logging
from datetime import datetime, timedelta


class LogAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased"):
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load pre-trained model or fine-tuned model
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_logbert")
            print("Loaded fine-tuned model")
        except:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
            print("Loaded base model")

        # Setup LoRA for fine-tuning
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=0.1,
            bias="none"
        )
        self.model = get_peft_model(self.model, self.lora_config)

        # Label encoder for error types
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["database", "server", "network", "application", "security"])

        # Log severity levels
        self.severity_levels = ["info", "warning", "error", "critical"]

    def preprocess_logs(self, raw_logs):
        """Chuẩn hóa log raw thành định dạng structured"""
        standardized_logs = []
        for log in raw_logs:
            # Đây là phần bạn cần điều chỉnh theo định dạng log cụ thể
            standardized = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "level": "error" if "error" in log.lower() else "warning",
                "service": "web" if "web" in log.lower() else "database",
                "message": log.strip(),
                "error_code": "ERR_001"  # Cần extract từ log thực tế
            }
            standardized_logs.append(standardized)
        return pd.DataFrame(standardized_logs)

    def analyze_errors(self, log_df):
        """Phân tích log và trả lời các câu hỏi"""
        results = {
            "has_error": False,
            "error_details": [],
            "future_prediction": {},
            "solutions": []
        }

        # Phát hiện lỗi
        error_logs = log_df[log_df["level"].isin(["error", "critical"])]
        results["has_error"] = len(error_logs) > 0

        if results["has_error"]:
            # Phân tích từng lỗi
            for _, row in error_logs.iterrows():
                # Phân loại lỗi bằng model
                inputs = self.tokenizer(row["message"], return_tensors="pt", truncation=True, padding=True)
                outputs = self.model(**inputs)
                predicted_class = torch.argmax(outputs.logits).item()
                error_type = self.label_encoder.inverse_transform([predicted_class])[0]

                error_detail = {
                    "message": row["message"],
                    "type": error_type,
                    "severity": row["level"],
                    "service": row["service"],
                    "timestamp": row["timestamp"]
                }
                results["error_details"].append(error_detail)

                # Đề xuất giải pháp dựa trên loại lỗi
                solutions = self._suggest_solutions(error_type, row["message"])
                results["solutions"].extend(solutions)

            # Dự đoán lỗi trong tương lai
            results["future_prediction"] = self._predict_future_errors(log_df)

        return results

    def _suggest_solutions(self, error_type, message):
        """Đề xuất giải pháp dựa trên loại lỗi"""
        solutions = []
        if error_type == "database":
            solutions.append("Kiểm tra kết nối database và credentials")
            solutions.append("Xem lại query đang thực thi")
        elif error_type == "server":
            solutions.append("Kiểm tra tài nguyên server (CPU, RAM, disk)")
            solutions.append("Xem lại cấu hình service")
        # Thêm các loại lỗi khác...

        return solutions

    def _predict_future_errors(self, log_df):
        """Dự đoán lỗi trong 15-30 phút tới"""
        # Phân tích tần suất lỗi
        error_rate = len(log_df[log_df["level"].isin(["error", "critical"])]) / len(log_df)

        # Dự đoán đơn giản - trong thực tế có thể dùng time series forecasting
        prediction = {
            "time_window": "15-30 minutes",
            "probability": min(1.0, error_rate * 2.5),  # Giả sử tỷ lệ lỗi tăng
            "likely_errors": ["database", "server"]  # Từ phân tích hiện tại
        }

        return prediction

    def fine_tune_model(self, training_data):
        """Fine-tune model với dữ liệu mới"""
        # Chuẩn bị dữ liệu training
        # (Cần triển khai chi tiết hơn)

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=8,
            num_train_epochs=3,
            save_steps=1000
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=training_data  # Cần chuẩn bị dataset
        )

        trainer.train()
        self.model.save_pretrained("./fine_tuned_logbert")


# Ví dụ sử dụng
if __name__ == "__main__":
    analyzer = LogAnalyzer()

    # 1. Tạo 10 dòng log chuẩn hóa (ví dụ)
    raw_logs = [
        "ERROR: Database connection failed - timeout after 30s",
        "WARNING: High memory usage detected on server-01",
        "ERROR: API /users returned 500",
        "INFO: User login successful",
        "CRITICAL: Disk space full on /var/log",
        "ERROR: Failed to authenticate with LDAP",
        "WARNING: Unusual network activity detected",
        "INFO: Backup completed successfully",
        "ERROR: Database query timeout",
        "WARNING: CPU temperature exceeding threshold"
    ]

    # 2. Chuẩn hóa log
    standardized_logs = analyzer.preprocess_logs(raw_logs)
    print("Standardized logs:")
    print(standardized_logs)

    # 3. Phân tích lỗi
    analysis = analyzer.analyze_errors(standardized_logs)
    print("\nAnalysis results:")
    print(f"1. Có lỗi hay không? {'Có' if analysis['has_error'] else 'Không'}")

    if analysis["has_error"]:
        print("\n2. Chi tiết lỗi:")
        for error in analysis["error_details"]:
            print(f"- [{error['severity'].upper()}] {error['message']}")
            print(f"  Loại: {error['type']}, Service: {error['service']}")

        print("\n3. Dự đoán lỗi trong 15-30 phút tới:")
        pred = analysis["future_prediction"]
        print(f"- Khả năng xảy ra lỗi: {pred['probability'] * 100:.1f}%")
        print(f"- Loại lỗi có thể xảy ra: {', '.join(pred['likely_errors'])}")

        print("\n4. Hướng xử lý:")
        for i, solution in enumerate(set(analysis["solutions"]), 1):
            print(f"{i}. {solution}")