import pandas as pd
import os
import torch
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fine_tuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FineTuner")

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
        self.model_path = model_path
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(LABELS)

        self._load_model()
        logger.info(f"FineTuner initialized with model at {model_path}")

    def _load_model(self):
        """Load the model, either from fine-tuned path or initialize a new one with LoRA"""
        if os.path.exists(self.model_path):
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                logger.info("Loaded fine-tuned model")
            except Exception as e:
                logger.error(f"Error loading fine-tuned model: {e}")
                self._initialize_base_model()
        else:
            self._initialize_base_model()

    def _initialize_base_model(self):
        """Initialize a base model with LoRA configuration"""
        try:
            model = AutoModelForSequenceClassification.from_pretrained(self.base_model, num_labels=len(LABELS))
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_lin", "v_lin"],
                lora_dropout=0.1,
                bias="none"
            )
            self.model = get_peft_model(model, lora_config)
            logger.info("Initialized base model with LoRA configuration")
        except Exception as e:
            logger.error(f"Error initializing base model: {e}")
            raise

    def prepare_dataset_from_csv(self, csv_path):
        """Prepare dataset from a CSV file"""
        try:
            df = pd.read_csv(csv_path)
            return self.prepare_dataset_from_dataframe(df)
        except Exception as e:
            logger.error(f"Error preparing dataset from CSV {csv_path}: {e}")
            return None

    def prepare_dataset_from_dataframe(self, df):
        """Prepare dataset from a DataFrame"""
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for fine-tuning")
            return None

        # Make sure the DataFrame has the required columns
        required_columns = ["message", "label"]
        if not all(col in df.columns for col in required_columns):
            logger.error(f"DataFrame missing required columns. Has columns: {df.columns.tolist()}")
            return None

        # Filter records with valid labels
        df = df[df['label'].isin(LABELS)]

        if df.empty:
            logger.warning("No valid labels found in DataFrame after filtering")
            return None

        logger.info(f"Preparing dataset with {len(df)} records")

        # Encode labels
        df['encoded_label'] = self.label_encoder.transform(df['label'])

        # Tokenize texts
        texts = df['message'].tolist()
        labels = df['encoded_label'].tolist()

        encodings = self.tokenizer(texts, truncation=True, padding='max_length', return_tensors="pt")
        dataset = LogDataset(encodings, labels)

        return dataset

    def fine_tune(self, dataset_or_df, epochs=3, batch_size=8):
        """Fine-tune the model with the provided dataset or DataFrame"""
        # Handle DataFrame input
        if isinstance(dataset_or_df, pd.DataFrame):
            dataset = self.prepare_dataset_from_dataframe(dataset_or_df)
            if dataset is None:
                logger.error("Could not prepare dataset from DataFrame")
                return False
        else:
            dataset = dataset_or_df

        if dataset is None or len(dataset) == 0:
            logger.warning("Empty dataset provided for fine-tuning")
            return False

        logger.info(f"Starting fine-tuning with {len(dataset)} samples, {epochs} epochs, batch size {batch_size}")

        try:
            training_args = TrainingArguments(
                output_dir="./results",
                per_device_train_batch_size=batch_size,
                num_train_epochs=epochs,
                save_strategy="no",
                logging_dir="./logs",
                logging_steps=10,
                learning_rate=2e-5,
                weight_decay=0.01,
                report_to="none"  # Disable reporting to avoid unnecessary logging
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset
            )

            trainer.train()

            # Save the fine-tuned model
            self.model.save_pretrained(self.model_path)
            logger.info(f"Model fine-tuned and saved to {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            return False

    def export_model(self, output_path=None):
        """Export the model to a specified path"""
        if output_path is None:
            output_path = self.model_path

        try:
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            logger.info(f"Model exported to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            return False


if __name__ == "__main__":
    tuner = FineTuner()
    dataset = tuner.prepare_dataset_from_csv("logs_to_finetune.csv")
    if dataset:
        tuner.fine_tune(dataset)