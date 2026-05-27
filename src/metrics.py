"""
Operational & Point-Wise Evaluation Metrics for Supply Chain Stockout Forecasting
=================================================================================
Includes:
1. Rigid Point-Wise Metrics:
   - Precision, Recall, F1, Accuracy, Exact Match 24h
2. Supply Chain Operational Metrics:
   - Window-Tolerant F1 (+- 1h and +- 2h slack)
   - Stockout Onset Timing MAE (How early/late was the first alert?)
   - Alert Timing Accuracy within +- 1 hour (%)
   - Episode Interval IoU (Curve overlap %)
   - Daily Volume MAE (Absolute error in total stockout hours)
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def window_tolerant_eval(y_true, y_pred, slack=1):
    """
    Evaluates window-tolerant precision, recall, and F1 given a temporal slack window (+- slack hours).

    Overcomes the 'Double Penalty' flaw where an alert 1 hour early/late is counted
    as both a False Positive and a False Negative, even though it provides immense operational utility.
    """
    N, H = y_true.shape
    tp_slack, fp_slack, fn_slack = 0, 0, 0

    for i in range(N):
        yt = y_true[i]
        yp = y_pred[i]

        yt_dilated = np.zeros(H, dtype=int)
        for h in range(H):
            if yt[h] == 1:
                yt_dilated[max(0, h - slack):min(H, h + slack + 1)] = 1

        yp_dilated = np.zeros(H, dtype=int)
        for h in range(H):
            if yp[h] == 1:
                yp_dilated[max(0, h - slack):min(H, h + slack + 1)] = 1

        for h in range(H):
            if yp[h] == 1:
                if yt_dilated[h] == 1:
                    tp_slack += 1
                else:
                    fp_slack += 1

        for h in range(H):
            if yt[h] == 1:
                if yp_dilated[h] == 0:
                    fn_slack += 1

    total_true = int(np.sum(y_true))
    rec_tp = total_true - fn_slack
    prec = tp_slack / max(1, (tp_slack + fp_slack))
    rec = rec_tp / max(1, total_true)
    f1 = 2 * (prec * rec) / max(1e-8, (prec + rec))
    return float(prec), float(rec), float(f1)


def compute_comprehensive_metrics(targets_arr, binary_preds, display_name="Model", params=0):
    """
    Computes a comprehensive suite of rigid point-wise and realistic operational metrics.

    Parameters:
    -----------
    targets_arr : np.ndarray of shape (N, 24)
        Ground truth binary stockout indicators (0 = In-Stock, 1 = Stockout).
    binary_preds : np.ndarray of shape (N, 24)
        Predicted binary stockout indicators (0 = In-Stock, 1 = Stockout).
    display_name : str
        Name of the model being evaluated.
    params : int
        Number of trainable parameters in the model.

    Returns:
    --------
    dict containing all computed metrics.
    """
    targets_arr = np.asarray(targets_arr, dtype=int)
    binary_preds = np.asarray(binary_preds, dtype=int)
    N, H = targets_arr.shape

    # 1. Rigid Point-Wise Metrics
    f1_rigid = float(f1_score(targets_arr.ravel(), binary_preds.ravel(), zero_division=0))
    rec_rigid = float(recall_score(targets_arr.ravel(), binary_preds.ravel(), zero_division=0))
    prec_rigid = float(precision_score(targets_arr.ravel(), binary_preds.ravel(), zero_division=0))
    acc_rigid = float(accuracy_score(targets_arr.ravel(), binary_preds.ravel()))
    exact_match = float(np.mean(np.all(binary_preds == targets_arr, axis=1))) * 100.0
    volume_mae = float(np.mean(np.abs(binary_preds.sum(axis=1) - targets_arr.sum(axis=1))))

    # 2. Window-Tolerant Metrics (+- 1h and +- 2h)
    prec_w1, rec_w1, f1_w1 = window_tolerant_eval(targets_arr, binary_preds, slack=1)
    prec_w2, rec_w2, f1_w2 = window_tolerant_eval(targets_arr, binary_preds, slack=2)

    # 3. Episode Interval IoU (Curve Overlap)
    ious = []
    for i in range(N):
        yt = targets_arr[i]
        yp = binary_preds[i]
        inter = np.sum((yt == 1) & (yp == 1))
        union = np.sum((yt == 1) | (yp == 1))
        ious.append(1.0 if union == 0 else inter / union)
    mean_iou = float(np.mean(ious)) * 100.0

    # 4. Stockout Onset Timing MAE & Alert Accuracy
    timing_errors = []
    for i in range(N):
        t_hours = np.where(targets_arr[i] == 1)[0]
        p_hours = np.where(binary_preds[i] == 1)[0]
        if len(t_hours) > 0 and len(p_hours) > 0:
            timing_errors.append(abs(t_hours[0] - p_hours[0]))

    mean_onset_mae = float(np.mean(timing_errors)) if len(timing_errors) > 0 else 0.0
    pct_within_1h = float(np.mean(np.array(timing_errors) <= 1) * 100.0) if len(timing_errors) > 0 else 0.0

    return {
        "Model": display_name,
        "Parameters": params,
        "Rigid F1": round(f1_rigid, 4),
        "Rigid Precision": round(prec_rigid, 4),
        "Rigid Recall": round(rec_rigid, 4),
        "Accuracy": round(acc_rigid, 4),
        "Window F1 (+-1h)": round(f1_w1, 4),
        "Window Prec (+-1h)": round(prec_w1, 4),
        "Window Rec (+-1h)": round(rec_w1, 4),
        "Window F1 (+-2h)": round(f1_w2, 4),
        "Onset Timing MAE (h)": round(mean_onset_mae, 2),
        "Timing <= +-1h (%)": round(pct_within_1h, 1),
        "Interval IoU (%)": round(mean_iou, 1),
        "Volume MAE (h)": round(volume_mae, 2),
        "Exact Match 24h (%)": round(exact_match, 2)
    }
