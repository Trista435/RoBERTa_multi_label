import numpy as np
def top_k_accuracy(y_true, y_probs, k=3):
    """
    y_true: numpy array, shape (n_samples,)
    y_probs: numpy array, shape (n_samples, n_classes)
    k: top-k 的值（預設為 3）
    """
    topk_preds = np.argsort(y_probs, axis=1)[:, -k:]  # 取每列 top-k 預測索引
    correct = [y_true[i] in topk_preds[i] for i in range(len(y_true))]
    return np.mean(correct)