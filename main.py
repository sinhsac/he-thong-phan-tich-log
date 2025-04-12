# Create a basic BERT-based log analysis framework

import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xcore

# 1. Load and prepare your log data
# Example: reading from a CSV file (adjust based on your data format)
# !wget https://your-log-data-source.com/logs.csv  # Uncomment to download your data
# logs_df = pd.read_csv('logs.csv')

# For demonstration, creating sample log data
logs = [
    "Connection established successfully",
    "Error: Failed to connect to database",
    "User authentication successful",
    "Warning: CPU usage above 90%",
    "System shutdown initiated"
]
labels = [0, 1, 0, 1, 0]  # 0: normal, 1: anomaly/error


# 3. Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 4. Create train/test datasets
train_texts, test_texts, train_labels, test_labels = train_test_split(logs, labels, test_size=0.2)
train_dataset = xcore.LogDataset(train_texts, train_labels, tokenizer)
test_dataset = xcore.LogDataset(test_texts, test_labels, tokenizer)

# 5. Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# 6. Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


# 7. Training loop (simple example)
def train_epoch():
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")


# Run training for 1 epoch as an example
train_epoch()

# 8. Function for analyzing new logs
def analyze_log(log_text):
    model.eval()
    encoding = tokenizer(
        log_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.softmax(outputs.logits, dim=1)

    return {
        'normal_probability': predictions[0][0].item(),
        'anomaly_probability': predictions[0][1].item(),
        'prediction': 'normal' if predictions[0][0] > predictions[0][1] else 'anomaly'
    }

log_to_analyze = "Error: Memory allocation failed"
result = analyze_log(log_to_analyze)
print(f"Log: {log_to_analyze}")
print(f"Analysis: {result}")