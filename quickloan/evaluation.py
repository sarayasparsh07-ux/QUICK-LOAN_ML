"""Classification metrics, curves and segment diagnostics. Positive class is always default = 1."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from . import finance


def classification_metrics(y_true, y_pred, scores=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if scores is not None:
        out["roc_auc"] = roc_auc_score(y_true, scores)
        out["pr_auc"] = average_precision_score(y_true, scores)
        out["brier"] = brier_score_loss(y_true, scores)
    return out


def expected_calibration_error(y_true, scores, n_bins: int = 10) -> float:
    """Weighted mean |observed default rate - mean score| over equal-count (decile) bins."""
    y, s = np.asarray(y_true, dtype=float), np.asarray(scores, dtype=float)
    order = np.argsort(s, kind="mergesort")
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) * abs(y[b].mean() - s[b].mean()) for b in bins if len(b)) / len(y))


def full_row(name, family, y_true, y_pred, scores, loan, interest) -> dict:
    row = {"model": name, "family": family}
    row.update(classification_metrics(y_true, y_pred, scores))
    row.update(finance.breakdown(y_true, y_pred, loan, interest))
    row["money_saved"] = row.pop("net_money_saved")
    return row


def curves(y_true, scores) -> dict:
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    frac_pos, mean_pred = calibration_curve(y_true, scores, n_bins=10, strategy="quantile")
    # thin the curves so the artifact stays small but the shape is exact enough to plot
    def thin(a, b, n=400):
        if len(a) <= n:
            return a.tolist(), b.tolist()
        idx = np.unique(np.linspace(0, len(a) - 1, n).astype(int))
        return a[idx].tolist(), b[idx].tolist()
    fpr, tpr = thin(fpr, tpr)
    rec, prec = thin(rec, prec)
    return {"roc": {"fpr": fpr, "tpr": tpr}, "pr": {"recall": rec, "precision": prec},
            "calibration": {"mean_pred": mean_pred.tolist(), "frac_pos": frac_pos.tolist()}}


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[20, 29, 39, 49, 64], labels=["21-29", "30-39", "40-49", "50-64"])


def segment_table(X_test: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    """Diagnostic error rates by age band and home ownership. Not a fairness certification."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred),
                      "age_band": age_band(X_test["age"]).astype(str).to_numpy(),
                      "home_ownership": X_test["home_ownership"].to_numpy()})
    rows = []
    for col in ["age_band", "home_ownership"]:
        for g, s in d.groupby(col, observed=True):
            pos, neg = s["y"] == 1, s["y"] == 0
            rows.append({
                "dimension": "Age band" if col == "age_band" else "Home ownership", "group": g,
                "n": len(s), "defaulters": int(pos.sum()), "default_rate": s["y"].mean(),
                "rejection_rate": s["p"].mean(),
                "recall": (s.loc[pos, "p"] == 1).mean() if pos.any() else np.nan,
                "false_rejection_rate": (s.loc[neg, "p"] == 1).mean() if neg.any() else np.nan,
            })
    return pd.DataFrame(rows)
