# train_single_jieba.py

import os
import torch
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, EvalPrediction, AutoTokenizer
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import EarlyStoppingCallback


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.ce_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

class SingleLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(list(tqdm(texts, desc="Tokenizing")), truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# ===== 載入資料 =====
train_texts = pd.read_csv("train_texts_single_jieba.csv")['text']
val_texts = pd.read_csv("val_texts_single_jieba.csv")['text']
train_labels = pd.read_csv("train_labels_single_jieba.csv").values.flatten()
val_labels = pd.read_csv("val_labels_single_jieba.csv").values.flatten()
label_encoder = joblib.load("label_encoder.pkl")

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", use_fast=True)
train_dataset = SingleLabelDataset(train_texts, train_labels, tokenizer)
val_dataset = SingleLabelDataset(val_texts, val_labels, tokenizer)

# ===== 模型與訓練參數 =====
model = BertForSequenceClassification.from_pretrained(
    "hfl/chinese-roberta-wwm-ext",
    num_labels=len(label_encoder.classes_),
    problem_type="single_label_classification"
)

def compute_metrics(pred: EvalPrediction):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_f1": f1_score(labels, preds, average="micro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro")
    }

class TQDMProgressBar(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_bar = tqdm(total=state.max_steps, desc="Training")
    def on_step_end(self, args, state, control, **kwargs):
        self.train_bar.update(1)
    def on_train_end(self, args, state, control, **kwargs):
        self.train_bar.close()

training_args = TrainingArguments(
    output_dir="./model_single_jieba",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=250,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[TQDMProgressBar(), EarlyStoppingCallback(early_stopping_patience=30)]
)

model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

print("🚀 開始訓練...")
trainer.train()
trainer.save_model("./model_single_jieba")
tokenizer.save_pretrained("./model_single_jieba")
print("✅ 訓練完成，模型已儲存。")

# ===== Loss 曲線 =====
loss_log = trainer.state.log_history
loss_values = [entry['loss'] for entry in loss_log if 'loss' in entry]
plt.plot(range(1, len(loss_values)+1), loss_values)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_curve_single_jieba.png")
