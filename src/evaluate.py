from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def fairness_by_group(df, y_true, y_pred, group_col: str):
    results = []
    for group_name, group_df in df.groupby(group_col):
        idx = group_df.index
        yt = np.array(y_true)[idx]
        yp = np.array(y_pred)[idx]
        if len(yt) == 0:
            continue
        approval_rate = float(np.mean(yp))
        tpr = float(((yp == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1))
        fpr = float(((yp == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1))
        results.append(
            {
                "group": group_name,
                "records": int(len(yt)),
                "approval_rate": approval_rate,
                "tpr": tpr,
                "fpr": fpr,
            }
        )
    return results
