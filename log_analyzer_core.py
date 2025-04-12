from datetime import datetime

import torch
from peft import get_peft_model, LoraConfig
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class LogAnalyzerCore:
    def __init__(self, model_name="distilbert-base-uncased", max_log_history=10000):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_log_history = max_log_history
        self._init_model(model_name)
        self._init_label_encoder()
        self.log_history = []

    def _init_model(self, model_name):
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_logbert")
            print("Loaded fine-tuned model")
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}. Loading base model.")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=5
            )

        # Apply LoRA consistently
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"
        )
        self.model = get_peft_model(self.model, lora_config)

    def _init_label_encoder(self):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["database", "server", "network", "application", "security"])

    def _trim_log_history(self):
        if len(self.log_history) > self.max_log_history:
            self.log_history = self.log_history[-self.max_log_history:]

    def preprocess_log(self, log_text):
        try:
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "level": self._detect_log_level(log_text),
                "service": self._detect_service(log_text),
                "message": log_text.strip(),
                "error_code": self._extract_error_code(log_text)
            }
        except Exception as e:
            print(f"Error preprocessing log: {e}")
            return None

    def _detect_log_level(self, text):
        text_lower = text.lower()
        if "critical" in text_lower:
            return "critical"
        elif "error" in text_lower:
            return "error"
        elif "warning" in text_lower:
            return "warning"
        return "info"

    def _detect_service(self, text):
        text_lower = text.lower()
        if "database" in text_lower or "sql" in text_lower:
            return "database"
        elif "web" in text_lower or "http" in text_lower:
            return "web"
        elif "api" in text_lower:
            return "api"
        return "unknown"

    def _extract_error_code(self, text):
        # Add more sophisticated error code extraction logic
        return "ERR_001"

    def analyze_log(self, structured_log):
        if not structured_log:
            return None, []

        try:
            inputs = self.tokenizer(
                structured_log["message"],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            outputs = self.model(**inputs)
            predicted_class = torch.argmax(outputs.logits).item()
            error_type = self.label_encoder.inverse_transform([predicted_class])[0]

            structured_log["type"] = error_type
            self.log_history.append(structured_log)
            self._trim_log_history()

            return structured_log, self.suggest_solutions(error_type, structured_log["message"])
        except Exception as e:
            print(f"Error analyzing log: {e}")
            return None, []