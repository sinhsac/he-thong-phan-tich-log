# Create a basic BERT-based log analysis framework

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer

import xcore

# 1. Load and prepare your log data
# Example: reading from a CSV file (adjust based on your data format)
# !wget https://your-log-data-source.com/logs.csv  # Uncomment to download your data
# logs_df = pd.read_csv('logs.csv')

# For demonstration, creating sample log data
# Ví dụ data với 5 cấp độ
# 3. Initialize tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)

print(f"Vocabulary size: {len(tokenizer)}")

# Giả sử data đã được chuẩn bị với 5 labels
logs = []
labels = []  # 0: normal, 1: notice, 2: warning, 3: error, 4: critical

# Ví dụ với nhiều mẫu hơn cho mỗi lớp
# Thêm 10 mẫu cho mỗi lớp
for i in range(20):
    logs.append(f"Normal log {i}: Connection established successfully")
    labels.append(0)
    logs.append(f"Notice log {i}: System resource usage normal")
    labels.append(1)
    logs.append(f"Warning log {i}: CPU usage above 90%")
    labels.append(2)
    logs.append(f"Error log {i}: Failed to connect to database")
    labels.append(3)
    logs.append(f"Critical log {i}: System crash detected")
    labels.append(4)

# 3. Kiểm tra labels có chuẩn không
print(f"Labels: {set(labels)}")  # Nên là {0, 1, 2, 3, 4} cho 5 classes

# 4. Kiểm tra các token IDs được tạo ra
sample_encoding = tokenizer(logs[0], return_tensors='pt')
print(f"Max token ID: {sample_encoding['input_ids'].max().item()}")

# Chia dữ liệu thành train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(logs, labels, test_size=0.2, stratify=labels)

# Tạo dataset và dataloader
train_dataset = xcore.LogDataset(train_texts, train_labels, tokenizer)
test_dataset = xcore.LogDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# Khởi tạo optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


# Hàm training
def train(epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

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
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Đánh giá sau mỗi epoch
        evaluate()

    # Lưu model sau khi train xong
    model.save_pretrained('./log_classifier_5_labels')
    tokenizer.save_pretrained('./log_classifier_5_labels')


# Hàm đánh giá
def evaluate():
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    # Tính accuracy
    accuracy = sum(1 for p, a in zip(predictions, actual_labels) if p == a) / len(actual_labels)
    print(f"Accuracy: {accuracy:.4f}")

    # Thêm confusion matrix để phân tích chi tiết
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(actual_labels, predictions)
    print("Confusion Matrix:")
    print(cm)

    # Classification report cung cấp precision, recall, f1-score cho mỗi class
    class_names = ['normal', 'notice', 'warning', 'error', 'critical']
    report = classification_report(actual_labels, predictions, target_names=class_names)
    print("Classification Report:")
    print(report)

# Run training for 1 epoch as an example
train(epochs=5)

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

    # Lấy các xác suất cho từng cấp độ
    probabilities = {
        'normal': predictions[0][0].item(),
        'notice': predictions[0][1].item(),
        'warning': predictions[0][2].item(),
        'error': predictions[0][3].item(),
        'critical': predictions[0][4].item()
    }

    # Lấy cấp độ có xác suất cao nhất
    max_prob_label = max(probabilities.items(), key=lambda x: x[1])

    return {
        'probabilities': probabilities,
        'prediction': max_prob_label[0]
    }

log_to_analyze = "Error: Memory allocation failed"
result = analyze_log(log_to_analyze)
print(f"Log: {log_to_analyze}")
print(f"Analysis: {result}")