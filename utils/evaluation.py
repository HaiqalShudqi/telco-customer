"""Metrik dan kurva evaluasi dari prediksi test set yang diekspor notebook (tanpa melatih apa pun)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def metrics_at(y: np.ndarray, p: np.ndarray, thr: float, beta: float = 2.0) -> dict:
    yh = (np.asarray(p) >= thr).astype(int)
    y = np.asarray(y).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); fp = int(((yh == 1) & (y == 0)).sum())
    fn = int(((yh == 0) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fb = (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec) if (prec + rec) else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec, "f1": f1, "f_beta": fb,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def roc_pr_summary(y: np.ndarray, p: np.ndarray) -> dict:
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}


def roc_points(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return fpr, tpr


def pr_points(y, p):
    prec, rec, _ = precision_recall_curve(y, p)
    return rec, prec


def threshold_table(y, p, beta: float = 2.0, step: float = 0.01) -> pd.DataFrame:
    grid = np.round(np.arange(0.05, 0.96, step), 2)
    rows = [{"threshold": t, **metrics_at(y, p, t, beta)} for t in grid]
    return pd.DataFrame(rows)
