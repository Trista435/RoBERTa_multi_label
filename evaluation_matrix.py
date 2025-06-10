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


from sklearn.metrics import accuracy_score

def thresholded_multilabel_accuracy(y_true, y_probs, threshold=0.3):
    """
    y_true: np.ndarray of shape (n_samples, n_classes), one-hot multi-label
    y_probs: np.ndarray of shape (n_samples, n_classes), softmax output
    threshold: float, probability threshold to include label in prediction
    """
    y_pred = (y_probs >= threshold).astype(int)

    # Edge case: if no prob > threshold, fallback to argmax (top-1 prediction)
    empty_rows = np.where(y_pred.sum(axis=1) == 0)[0]
    y_pred[empty_rows, np.argmax(y_probs[empty_rows], axis=1)] = 1

    return accuracy_score(y_true, y_pred)

def ratio_based_multilabel_accuracy(y_true, y_probs, ratio_threshold=1.3):
    """
    使用 softmax 輸出與機率比值，決定多標籤預測。

    y_true: (n_samples, n_classes) 多標籤 ground truth (one-hot)
    y_probs: (n_samples, n_classes) softmax 機率輸出
    ratio_threshold: float, 當前一機率 / 下一個機率 >= threshold 時停止擴充預測
    """
    n_samples, n_classes = y_probs.shape
    y_pred = np.zeros_like(y_probs, dtype=int)

    for i in range(n_samples):
        prob_row = y_probs[i]
        sorted_indices = np.argsort(prob_row)[::-1]  # 機率排序：高 → 低

        selected = [sorted_indices[0]]  # 一定選第一高
        for j in range(1, n_classes):
            prev = prob_row[sorted_indices[j - 1]]
            curr = prob_row[sorted_indices[j]]
            if prev / curr < ratio_threshold:
                selected.append(sorted_indices[j])
            else:
                break
        y_pred[i, selected] = 1

    return accuracy_score(y_true, y_pred)

