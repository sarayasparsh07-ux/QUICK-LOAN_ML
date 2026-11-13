"""The official money-saved metric and everything derived from it.

`money_saved` is the assignment's function, logic unchanged. Everything else in this module is a
breakdown of the same quantity or a separate risk measure that is never subtracted from it.

Payoff per decision (interest = foregone interest as a share of principal, default 20 %):

                     approve (0)            reject (1)
    repaid  (0)   +interest * loan       0  (interest lost)
    default (1)   -1.00 * loan           0  (principal saved)

Money saved = model payoff - approve-everyone payoff
            = principal of correctly rejected defaulters - interest * principal of wrongly rejected repayers.
"""
from __future__ import annotations

import numpy as np


def money_saved(y_true, y_pred, loan_amounts, interest=0.20):
    """Official assignment metric (logic unchanged)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    loan = np.asarray(loan_amounts)

    caught_defaulter = (y_pred == 1) & (y_true == 1)
    wrongly_rejected = (y_pred == 1) & (y_true == 0)

    return (
        loan[caught_defaulter].sum()
        - interest * loan[wrongly_rejected].sum()
    )


def breakdown(y_true, y_pred, loan_amounts, interest=0.20) -> dict:
    """Components of money saved, plus missed-default exposure as a *separate* risk measure."""
    y_true, y_pred, loan = np.asarray(y_true), np.asarray(y_pred), np.asarray(loan_amounts, dtype=float)
    tp = (y_pred == 1) & (y_true == 1)
    fp = (y_pred == 1) & (y_true == 0)
    fn = (y_pred == 0) & (y_true == 1)
    tn = (y_pred == 0) & (y_true == 0)
    loans_saved = float(loan[tp].sum())
    interest_lost = float(interest * loan[fp].sum())
    net = float(money_saved(y_true, y_pred, loan, interest))
    assert np.isclose(net, loans_saved - interest_lost)          # identity, never double counted
    return {
        "n_tp": int(tp.sum()), "n_fp": int(fp.sum()), "n_fn": int(fn.sum()), "n_tn": int(tn.sum()),
        "loans_saved": loans_saved,
        "interest_lost": interest_lost,
        "net_money_saved": net,
        "missed_default_principal": float(loan[fn].sum()),     # exposure only - NOT subtracted again
        "rejected_good_principal": float(loan[fp].sum()),
    }


def baselines(y_true, loan_amounts, interest=0.20) -> dict:
    y_true, loan = np.asarray(y_true), np.asarray(loan_amounts, dtype=float)
    return {
        "approve_all": float(money_saved(y_true, np.zeros_like(y_true), loan, interest)),   # always 0
        "reject_all": float(money_saved(y_true, np.ones_like(y_true), loan, interest)),
        "total_default_principal": float(loan[y_true == 1].sum()),
        "total_good_principal": float(loan[y_true == 0].sum()),
    }


def portfolio_pnl(y_true, y_pred, loan_amounts, interest=0.20) -> float:
    """Absolute simplified P&L of a decision policy (interest earned on repayers minus lost principal)."""
    y_true, y_pred, loan = np.asarray(y_true), np.asarray(y_pred), np.asarray(loan_amounts, dtype=float)
    approved = y_pred == 0
    return float(interest * loan[approved & (y_true == 0)].sum() - loan[approved & (y_true == 1)].sum())


THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def best_threshold(y_true, scores, loan_amounts, interest=0.20, grid=None) -> tuple[float, float]:
    """Threshold maximising money saved. Only ever called on training-set out-of-fold scores."""
    grid = THRESHOLD_GRID if grid is None else grid
    scores = np.asarray(scores)
    values = [money_saved(y_true, (scores >= t).astype(int), loan_amounts, interest) for t in grid]
    i = int(np.argmax(values))
    return float(grid[i]), float(values[i])


# ------------------------------------------------------------------------- interest-dependent thresholds
def threshold_curve(y_true, scores, loan_amounts, grid=None) -> dict:
    """For every candidate threshold t: principal of caught defaulters and of rejected repayers.

    money_saved(t, i) = caught(t) - i * wrong(t), computed with exactly the same masked sums as the official
    function, so `threshold_at` returns the same threshold as `best_threshold` at any interest - but for the
    whole 5-50 % slider range in microseconds. Built from training out-of-fold scores only."""
    grid = THRESHOLD_GRID if grid is None else np.asarray(grid)
    y, s, loan = np.asarray(y_true), np.asarray(scores), np.asarray(loan_amounts, dtype=float)
    caught = np.array([loan[(s >= t) & (y == 1)].sum() for t in grid])
    wrong = np.array([loan[(s >= t) & (y == 0)].sum() for t in grid])
    return {"grid": np.asarray(grid, dtype=float), "caught": caught, "wrong": wrong}


def threshold_at(curve: dict, interest: float) -> float:
    values = curve["caught"] - interest * curve["wrong"]
    return float(curve["grid"][int(np.argmax(values))])


def break_even_threshold(interest: float) -> float:
    """Reject when P(default) x loan > P(repay) x interest x loan  <=>  P(default) > i / (1 + i).
    Exact only for calibrated probabilities; shown as a reference line, never used for decisions."""
    return interest / (1 + interest)


# ------------------------------------------------------------------------- bootstrap uncertainty
BOOT_B, BOOT_SEED = 2000, 1


def bootstrap_counts(n: int, B: int = BOOT_B, seed: int = BOOT_SEED) -> np.ndarray:
    """B x n matrix: how often each test loan appears in each bootstrap resample (paired across models)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    counts = np.zeros((B, n), dtype=np.float64)
    for b in range(B):
        counts[b] = np.bincount(idx[b], minlength=n)
    return counts


def bootstrap_money(counts, y_true, preds: dict, loan_amounts, interest=0.20) -> dict[str, np.ndarray]:
    """Money saved of every model on every resample. Each resample sum equals `money_saved` on the
    resampled loans; the identity is unit-tested."""
    y, loan = np.asarray(y_true), np.asarray(loan_amounts, dtype=float)
    out = {}
    for name, p in preds.items():
        p = np.asarray(p)
        caught = loan * ((p == 1) & (y == 1))
        wrong = loan * ((p == 1) & (y == 0))
        out[name] = counts @ caught - interest * (counts @ wrong)
    return out


def bootstrap_summary(boot: dict[str, np.ndarray], champion: str, level: float = 0.95) -> dict[str, dict]:
    """Percentile CI per model, probability of being best, and the paired gap to the champion."""
    names = list(boot)
    M = np.column_stack([boot[n] for n in names])                       # B x models
    best = np.argmax(M, axis=1)
    a = (1 - level) / 2
    out = {}
    for j, n in enumerate(names):
        gap = boot[champion] - boot[n]
        lo, hi = np.quantile(gap, [a, 1 - a])
        out[n] = {"ci_low": float(np.quantile(M[:, j], a)), "ci_high": float(np.quantile(M[:, j], 1 - a)),
                  "p_best": float(np.mean(best == j)),
                  "gap_ci_low": float(lo), "gap_ci_high": float(hi),
                  "distinguishable": bool(n != champion and lo > 0)}
    return out
