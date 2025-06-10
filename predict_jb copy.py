# predict_test_single_jieba.py

import torch
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from transformers import BertForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from evaluation_matrix import top_k_accuracy, thresholded_multilabel_accuracy, ratio_based_multilabel_accuracy

# ===== 載入資料 =====
texts = pd.read_csv("test_texts_single_jieba.csv")['text']
labels = pd.read_csv("test_labels_single_jieba.csv").values.flatten()
label_encoder = joblib.load("label_encoder.pkl")

texts_df = pd.DataFrame({'text': texts})
grouped_multi = pd.read_csv("test_labels_multilabel_jieba.csv")
mlb = joblib.load("multilabel_binarizer.pkl")
assert grouped_multi.shape[0] == texts_df.shape[0]

tokenizer = AutoTokenizer.from_pretrained("./model_single_jieba")
model = BertForSequenceClassification.from_pretrained("./model_single_jieba")
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

# ===== 預測 =====
all_preds = []
all_probs = []
with torch.no_grad():
    for i in tqdm(range(0, len(texts), 32), desc="Predicting Test"):
        batch_texts = texts[i:i+32].tolist()
        encodings = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encodings = {k: v.to(model.device) for k, v in encodings.items()}
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_probs.append(probs)
        all_preds.extend(preds)

all_probs = np.vstack(all_probs)

# ===== 多標籤評估指標 =====
pred_labels = label_encoder.inverse_transform(all_preds)
pred_multi = mlb.transform(pred_labels.reshape(-1, 1))
subset_acc = accuracy_score(grouped_multi, pred_multi)

# ===== 自定義多標籤評估指標 =====
y_test_multi = pd.read_csv("test_labels_multilabel_jieba.csv").values
assert y_test_multi.shape == all_probs.shape, \
    f"Shape mismatch: y_true={y_test_multi.shape}, probs={all_probs.shape}"

subset_acc_thresh = thresholded_multilabel_accuracy(y_test_multi, all_probs, threshold=0.3)
# ===== 自定義多標籤評估指標 =====
ratio_acc = ratio_based_multilabel_accuracy(y_test_multi, all_probs, ratio_threshold=1.4)

for ratio in [1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0, 4.0]:
    ratio_acc = ratio_based_multilabel_accuracy(y_test_multi, all_probs, ratio_threshold=ratio)
    print(f"Subset Accuracy @ratio={ratio:.1f}: {ratio_acc:.4f}")

# ===== 評估指標 =====
acc = accuracy_score(labels, all_preds)
top3_acc = top_k_accuracy(labels, all_probs, k=3)
macro_f1 = f1_score(labels, all_preds, average="macro")
micro_f1 = f1_score(labels, all_preds, average="micro")
precision = precision_score(labels, all_preds, average="macro")
recall = recall_score(labels, all_preds, average="macro")

print("\n=== 測試集評估結果 ===")
print(f"Accuracy   : {acc:.4f}")
print(f"Ratio-based Subset Accuracy @1.4: {ratio_acc:.4f}")
print(f"Subset Accuracy @threshold=0.3: {subset_acc_thresh:.4f}")
print(f"Subset Accuracy (multi-label 完全匹配): {subset_acc:.4f}")
print(f"Top-3 Accuracy: {top3_acc:.6f}")
print(f"Macro F1   : {macro_f1:.4f}")
print(f"Micro F1   : {micro_f1:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print("=" * 35)

# ===== 輸出結果 CSV =====
df_result = pd.DataFrame({
    "text": [t[:200] + "..." if len(t) > 200 else t for t in texts],
    "true_label": label_encoder.inverse_transform(labels),
    "predicted_label": label_encoder.inverse_transform(all_preds)
})
probs_df = pd.DataFrame(all_probs, columns=label_encoder.classes_)
df_result = pd.concat([df_result, probs_df], axis=1)
df_result.to_csv("test_predictions_single_jieba.csv", index=False, encoding="utf-8-sig")
print("📄 預測結果儲存為 test_predictions_single_jieba.csv")
