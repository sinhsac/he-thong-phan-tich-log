import pandas as pd
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig
import torch
import os
from datetime import datetime

LABELS = ["database", "server", "network", "application", "security"]


class LogDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class FineTuner:
    def __init__(self, model_path="./fine_tuned_logbert", base_model="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(LABELS)
        self.model_path = model_path
        self.base_model = base_model
        self._init_model()

    def _init_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                print(f"Loaded fine-tuned model from {self.model_path}")
            except Exception as e:
                print(f"Error loading fine-tuned model: {e}. Loading base model instead.")
                self._load_base_model_with_lora()
        else:
            self._load_base_model_with_lora()

    def _load_base_model_with_lora(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=len(LABELS))
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"
        )
        self.model = get_peft_model(model, lora_config)
        print("Loaded base model with LoRA")

    def prepare_dataset(self, df):
        # Filter and validate data
        df = df[df['label'].isin(LABELS)]
        if df.empty:
            raise ValueError("No valid labels found in the dataset")

        df['label'] = self.label_encoder.transform(df['label'])
        texts = df['message'].tolist()
        labels = df['label'].tolist()

        encodings = self.tokenizer(texts, truncation=True, padding='max_length', max_length=512)
        return LogDataset(encodings, labels)

    def fine_tune(self, dataset, epochs=3, batch_size=8):
        if len(dataset) < 10:
            print("Not enough samples for fine-tuning (min 10 required)")
            return False

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            save_strategy="epoch",
            evaluation_strategy="no",
            logging_dir="./logs",
            logging_steps=10,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )

        try:
            trainer.train()
            self.model.save_pretrained(self.model_path)
            print(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            return False