import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class LogDataset(Dataset):
    def __init__(self, logs, labels, tokenizer, max_length=128):
        self.logs = logs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.logs[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class LogAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_logbert")
            logging.info("Loaded fine-tuned model.")
        except:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
            logging.info("Loaded base model.")

        self.lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=0.1, bias="none"
        )
        self.model = get_peft_model(self.model, self.lora_config)

        self.label_encoder = LabelEncoder()
        self.error_types = ["database", "server", "network", "application", "security"]
        self.label_encoder.fit(self.error_types)

        self.severity_levels = ["info", "warning", "error", "critical"]

        # Cache logs used for future fine-tuning
        self.finetune_log_cache = []

    def preprocess_log(self, log_line):
        level = "error" if "error" in log_line.lower() or "fail" in log_line.lower() else "warning"
        service = "web" if "web" in log_line.lower() else "database"
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "service": service,
            "message": log_line.strip(),
            "error_code": "ERR_UNKNOWN"
        }

    def predict_log(self, log_line):
        log_entry = self.preprocess_log(log_line)
        inputs = self.tokenizer(log_entry["message"], return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        predicted_class = torch.argmax(outputs.logits).item()
        error_type = self.label_encoder.inverse_transform([predicted_class])[0]

        logging.info(f"[{log_entry['timestamp']}] {log_entry['level'].upper()} - {log_entry['message']}")
        logging.info(f"=> Dự đoán loại lỗi: {error_type}")

        # Thêm vào cache để fine-tune sau
        self.finetune_log_cache.append((log_entry["message"], error_type))
        return error_type

    def fine_tune_on_cache(self):
        if not self.finetune_log_cache:
            logging.info("Không có log mới để fine-tune.")
            return

        logs, labels = zip(*self.finetune_log_cache)
        encoded_labels = self.label_encoder.transform(labels)
        train_texts, val_texts, train_labels, val_labels = train_test_split(logs, encoded_labels, test_size=0.2)

        train_dataset = LogDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = LogDataset(val_texts, val_labels, self.tokenizer)

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=4,
            num_train_epochs=2,
            logging_dir="./logs",
            logging_steps=10,
            save_steps=1000,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        logging.info("Fine-tuning model với log mới...")
        trainer.train()
        self.model.save_pretrained("./fine_tuned_logbert")
        logging.info("Đã fine-tune và lưu mô hình mới.")
        self.finetune_log_cache = []

    def interactive_loop(self):
        logging.info("Nhập log mới để dự đoán, nhập 'exit' để thoát.")
        while True:
            log_input = input("\nLog> ")
            if log_input.strip().lower() == "exit":
                break
            self.predict_log(log_input)
            self.fine_tune_on_cache()

if __name__ == "__main__":
    analyzer = LogAnalyzer()
    analyzer.interactive_loop()
