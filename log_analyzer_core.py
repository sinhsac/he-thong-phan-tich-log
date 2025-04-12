# log_analyzer_core.py

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from peft import LoraConfig, get_peft_model

class LogAnalyzerCore:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_logbert")
            print("Loaded fine-tuned model")
        except:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
            print("Loaded base model")

        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=0.1,
            bias="none"
        )
        self.model = get_peft_model(self.model, self.lora_config)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["database", "server", "network", "application", "security"])

        self.log_history = []  # Dùng cho realtime monitor và fine-tune sau

    def preprocess_log(self, log_text):
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": "error" if "error" in log_text.lower() else ("critical" if "critical" in log_text.lower() else "warning"),
            "service": "web" if "web" in log_text.lower() else "database",
            "message": log_text.strip(),
            "error_code": "ERR_001"
        }

    def analyze_log(self, structured_log):
        inputs = self.tokenizer(structured_log["message"], return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        predicted_class = torch.argmax(outputs.logits).item()
        error_type = self.label_encoder.inverse_transform([predicted_class])[0]

        structured_log["type"] = error_type

        self.log_history.append(structured_log)

        return structured_log, self.suggest_solutions(error_type, structured_log["message"])

    def suggest_solutions(self, error_type, message):
        solutions = []
        if error_type == "database":
            solutions.append("Kiểm tra kết nối database và credentials")
            solutions.append("Xem lại query đang thực thi")
        elif error_type == "server":
            solutions.append("Kiểm tra tài nguyên server (CPU, RAM, disk)")
            solutions.append("Xem lại cấu hình service")
        elif error_type == "network":
            solutions.append("Kiểm tra kết nối mạng, DNS và firewall")
        elif error_type == "application":
            solutions.append("Kiểm tra mã nguồn và exception handling")
        elif error_type == "security":
            solutions.append("Kiểm tra cấu hình bảo mật và nhật ký truy cập")
        return solutions

    def get_log_history(self):
        return pd.DataFrame(self.log_history)

    def predict_trend(self):
        df = self.get_log_history()
        if df.empty:
            return {
                "time_window": "15-30 minutes",
                "probability": 0.0,
                "likely_errors": []
            }

        recent_logs = df.tail(50)
        error_rate = len(recent_logs[recent_logs.level.isin(["error", "critical"])]) / len(recent_logs)
        likely_types = recent_logs["type"].value_counts().head(2).index.tolist()

        return {
            "time_window": "15-30 minutes",
            "probability": min(1.0, error_rate * 2.5),
            "likely_errors": likely_types
        }

    def prepare_for_fine_tuning(self):
        df = self.get_log_history()
        if df.empty:
            return None, None

        texts = df["message"].tolist()
        labels = self.label_encoder.transform(df["type"].tolist())

        encodings = self.tokenizer(texts, truncation=True, padding=True)
        dataset = LogDataset(encodings, labels)
        return dataset, self.label_encoder

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
