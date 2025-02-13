import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_scheduler
)
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
import joblib
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from tqdm import tqdm


# Load the dataset
df = pd.read_csv('dataset.csv')

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')

# Tokenize and encode the symptoms
tokenized_data = [tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True) 
                 for text in df['Symptoms']]
max_length = max(len(seq) for seq in tokenized_data)
input_ids = torch.tensor([seq + [0] * (max_length - len(seq)) for seq in tokenized_data])

# Encode labels
label_encoder = LabelEncoder()
labels = torch.tensor(label_encoder.fit_transform(df['Condition']), dtype=torch.long)

# Create attention masks
attention_masks = torch.tensor([[1 if token > 0 else 0 for token in seq] for seq in input_ids])

# Create tensor dataset
dataset = TensorDataset(input_ids, attention_masks, labels)

# Split dataset
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloaders
batch_size = 16
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
validation_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    'dmis-lab/biobert-v1.1', 
    num_labels=len(label_encoder.classes_)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 10
early_stopping_limit = 3
best_val_loss = float('inf')
no_improve_epochs = 0

num_training_steps = epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training Loop
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:
        b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_train_loss += loss.item()
        progress_bar.set_postfix({"Loss": loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}: Average Training Loss: {avg_train_loss}")

    # Validation
    model.eval()
    total_val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in validation_dataloader:
            b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            total_val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(validation_dataloader)
    print(f"Epoch {epoch+1}: Validation Loss: {avg_val_loss}")
    print("Validation Accuracy:", accuracy_score(all_labels, all_preds))

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_epochs = 0
        model.save_pretrained("model")
        tokenizer.save_pretrained("tokenizer")
        joblib.dump(label_encoder, "label_encoder.pkl")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stopping_limit:
            print("Early stopping triggered.")
            break

# Final Evaluation
print("\nFinal Classification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
