import pandas as pd
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig
import torch
import os

LABELS = ["database", "server", "network", "application", "security"]

class FineTuner:
    def __init__(self, model_path="./fine_tuned_logbert", base_model="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(LABELS)

        if os.path.exists(model_path):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            print("Loaded fine-tuned model")
        else:
            model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=len(LABELS))
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_lin", "v_lin"],
                lora_dropout=0.1,
                bias="none"
            )
            self.model = get_peft_model(model, lora_config)
            print("Loaded base model with LoRA")

    def prepare_dataset(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df[df['label'].isin(LABELS)]
        df['label'] = self.label_encoder.transform(df['label'])

        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda e: self.tokenizer(e['message'], truncation=True, padding='max_length'), batched=True)
        return dataset

    def fine_tune(self, dataset):
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=8,
            num_train_epochs=3,
            save_strategy="no",
            logging_dir="./logs",
            logging_steps=10
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )

        trainer.train()
        self.model.save_pretrained("./fine_tuned_logbert")

if __name__ == "__main__":
    tuner = FineTuner()
    dataset = tuner.prepare_dataset("logs_to_finetune.csv")
    tuner.fine_tune(dataset)
