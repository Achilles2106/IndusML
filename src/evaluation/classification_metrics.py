from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def compute_classification_metrics(y_true, y_pred, y_scores):
    """
    y_true: list of 0/1 (0 = good, 1 = defective)
    y_pred: list of 0/1 predicted labels (based on threshold)
    y_scores: list of continuous anomaly scores (for AUROC)
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = None  # happens if y_true has only one class present

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc
    }