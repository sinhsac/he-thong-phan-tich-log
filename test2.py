import pandas as pd
import numpy as np
import logging
import torch
from datetime import datetime
from typing import List, Dict, Any
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from sklearn.preprocessing import LabelEncoder
from peft import LoraConfig, get_peft_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LogPreprocessor:
    def __init__(self):
        self.default_error_code = "ERR_001"

    def standardize(self, raw_logs: List[str]) -> pd.DataFrame:
        standardized_logs = []

        for log in raw_logs:
            level = "error" if "error" in log.lower() or "critical" in log.lower() else "warning"
            service = "web" if "web" in log.lower() else "database"

            standardized_logs.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "level": level,
                "service": service,
                "message": log.strip(),
                "error_code": self.default_error_code
            })

        return pd.DataFrame(standardized_logs)


class SolutionAdvisor:
    def suggest(self, error_type: str, message: str) -> List[str]:
        mapping = {
            "database": [
                "Kiểm tra kết nối database và credentials",
                "Xem lại query đang thực thi"
            ],
            "server": [
                "Kiểm tra tài nguyên server (CPU, RAM, disk)",
                "Xem lại cấu hình service"
            ],
            "network": [
                "Kiểm tra kết nối mạng và thiết bị chuyển mạch",
                "Đảm bảo firewall không chặn traffic"
            ],
            "application": [
                "Xem lại log ứng dụng chi tiết",
                "Xác minh cấu hình môi trường (env)"
            ],
            "security": [
                "Kiểm tra quyền truy cập và sự kiện đăng nhập bất thường",
                "Rà soát cấu hình bảo mật"
            ]
        }
        return mapping.get(error_type, ["Không tìm thấy hướng xử lý phù hợp."])


class LogAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_logbert")
            logging.info("Loaded fine-tuned model from disk.")
        except Exception as e:
            logging.warning("Could not load fine-tuned model. Using base model. Error: %s", str(e))
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

        self.model = get_peft_model(self.model, LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=0.1,
            bias="none"
        ))

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["database", "server", "network", "application", "security"])
        self.severity_levels = ["info", "warning", "error", "critical"]
        self.solution_advisor = SolutionAdvisor()

    def analyze_errors(self, log_df: pd.DataFrame) -> Dict[str, Any]:
        results = {
            "has_error": False,
            "error_details": [],
            "future_prediction": {},
            "solutions": []
        }

        error_logs = log_df[log_df["level"].isin(["error", "critical"])]
        results["has_error"] = not error_logs.empty

        for _, row in error_logs.iterrows():
            inputs = self.tokenizer(row["message"], return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            predicted_class = torch.argmax(outputs.logits).item()
            error_type = self.label_encoder.inverse_transform([predicted_class])[0]

            error_info = {
                "message": row["message"],
                "type": error_type,
                "severity": row["level"],
                "service": row["service"],
                "timestamp": row["timestamp"]
            }
            results["error_details"].append(error_info)
            results["solutions"].extend(self.solution_advisor.suggest(error_type, row["message"]))

        if results["has_error"]:
            results["future_prediction"] = self._predict_future_errors(log_df)

        return results

    def _predict_future_errors(self, log_df: pd.DataFrame) -> Dict[str, Any]:
        error_ratio = len(log_df[log_df["level"].isin(["error", "critical"])]) / max(len(log_df), 1)
        return {
            "time_window": "15-30 minutes",
            "probability": round(min(1.0, error_ratio * 2.5), 2),
            "likely_errors": ["database", "server"]  # TODO: phân tích thực tế để chọn động
        }

    def fine_tune_model(self, training_dataset: Any) -> None:
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=8,
            num_train_epochs=3,
            save_steps=1000,
            logging_dir='./logs'
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=training_dataset
        )

        trainer.train()
        self.model.save_pretrained("./fine_tuned_logbert")
        logging.info("Model fine-tuned and saved to ./fine_tuned_logbert")


# =========================
# Ví dụ sử dụng
# =========================
if __name__ == "__main__":
    analyzer = LogAnalyzer()
    preprocessor = LogPreprocessor()

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

    structured_logs = preprocessor.standardize(raw_logs)
    logging.info("Standardized logs:\n%s", structured_logs.to_string(index=False))

    result = analyzer.analyze_errors(structured_logs)

    if result["has_error"]:
        print("\n🛑 Phát hiện lỗi trong log!")
        for err in result["error_details"]:
            print(f"[{err['severity'].upper()}] {err['message']} → {err['type']} ({err['service']})")

        pred = result["future_prediction"]
        print(f"\n🔮 Dự đoán lỗi tiếp theo (trong {pred['time_window']}):")
        print(f"- Xác suất lỗi: {pred['probability']*100:.1f}%")
        print(f"- Loại lỗi có khả năng cao: {', '.join(pred['likely_errors'])}")

        print("\n🛠️  Hướng xử lý gợi ý:")
        for i, solution in enumerate(set(result["solutions"]), 1):
            print(f"{i}. {solution}")
    else:
        print("\n✅ Không phát hiện lỗi nghiêm trọng trong log.")
