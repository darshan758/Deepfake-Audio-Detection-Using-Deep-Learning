import numpy as np
from sklearn.metrics import roc_curve

def alt_compute_eer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).
    y_true: ground truth labels (0 or 1)
    y_pred: predicted labels/probabilities
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1 - tpr
    # Find point where FPR == FNR
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer
