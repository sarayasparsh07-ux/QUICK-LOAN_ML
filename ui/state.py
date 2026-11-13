"""Artifact loading (cached) and the scenario calculations every page shares.

Money figures are always recomputed from stored holdout predictions with the official
`quickloan.finance.money_saved` function - nothing financial is hard-coded or cached per interest.

Business thresholds are re-optimised for the current interest assumption from *training* out-of-fold
threshold curves (never from the test set); the 0.50 benchmark is fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from quickloan import artifacts as A, config as C, evaluation as E, finance as F

POLICIES = {"benchmark": "0.50 benchmark", "business": "CV business thresholds"}


@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    return A.load_model_artifact(), A.load_eval_artifact()


@st.cache_resource(show_spinner=False)
def boot_counts(n: int, B: int, seed: int) -> np.ndarray:
    return F.bootstrap_counts(n, B, seed)


def thresholds(ev: dict, policy: str, interest: float) -> dict[str, float]:
    if policy == "benchmark":
        return {n: C.BENCHMARK_THRESHOLD for n in ev["models"]}
    return {n: F.threshold_at(ev["threshold_curves"][n], interest) for n in ev["models"]}


def predictions(ev: dict, policy: str, interest: float) -> dict[str, np.ndarray]:
    t = thresholds(ev, policy, interest)
    return {n: (r["scores"] >= t[n]).astype(int) for n, r in ev["models"].items()}


@st.cache_data(show_spinner=False)
def _static_metrics(ev_id: str, policy: str, interest_pct: int) -> pd.DataFrame:
    """Threshold-dependent classification metrics (interest only matters through business thresholds)."""
    _, ev = load_artifacts()
    i = interest_pct / 100
    t, p = thresholds(ev, policy, i), predictions(ev, policy, i)
    rows = []
    for name, r in ev["models"].items():
        m = E.classification_metrics(ev["y_test"], p[name], r["scores"])
        m.update(model=name, family=r["family"], threshold=t[name])
        rows.append(m)
    return pd.DataFrame(rows)


@dataclass
class Board:
    df: pd.DataFrame            # one row per model, sorted by money saved (desc)
    champion: str
    interest: float
    policy: str
    baselines: dict
    preds: dict = field(default_factory=dict)
    boot: dict = field(default_factory=dict)          # per-model bootstrap summary for this scenario
    baseline_model: dict = field(default_factory=dict)

    def row(self, name: str) -> pd.Series:
        return self.df.set_index("model").loc[name]


def board(ev: dict, interest: float, policy: str) -> Board:
    ev_id = f"{ev['deployed_model']}|{len(ev['y_test'])}|{ev['schema_version']}"
    pct = int(round(interest * 100))
    base = _static_metrics(ev_id, policy, pct if policy == "business" else 0)
    preds = predictions(ev, policy, interest)
    y, loan = ev["y_test"], ev["loan_test"]
    fin = []
    for name in base["model"]:
        b = F.breakdown(y, preds[name], loan, interest)
        b["money_saved"] = F.money_saved(y, preds[name], loan, interest)        # official function
        b["model"] = name
        fin.append(b)
    df = base.merge(pd.DataFrame(fin), on="model")
    df = df.sort_values(["money_saved", "recall", "model"], ascending=[False, False, True]).reset_index(drop=True)
    champion = df.iloc[0]["model"]

    bs = ev["bootstrap"]
    counts = boot_counts(len(y), bs["B"], bs["seed"])
    summary = F.bootstrap_summary(F.bootstrap_money(counts, y, preds, loan, interest), champion)
    for col in ("ci_low", "ci_high", "p_best", "gap_ci_low", "gap_ci_high", "distinguishable"):
        df[col] = df["model"].map(lambda m: summary[m][col])

    (bname, bl), = ev["baselines"].items()
    bm = E.classification_metrics(y, bl["pred_050"], bl["scores"])
    bm.update(F.breakdown(y, bl["pred_050"], loan, interest), name=bname,
              money_saved=F.money_saved(y, bl["pred_050"], loan, interest))
    return Board(df=df, champion=champion, interest=interest, policy=policy,
                 baselines=F.baselines(y, loan, interest), preds=preds, boot=summary, baseline_model=bm)


def sensitivity(ev: dict, policy: str, grid=None) -> pd.DataFrame:
    grid = np.round(np.arange(C.INTEREST_MIN, C.INTEREST_MAX + 1e-9, C.INTEREST_STEP), 2) if grid is None else grid
    y, loan = ev["y_test"], ev["loan_test"]
    rows = []
    for i in grid:
        p = predictions(ev, policy, float(i))
        rows += [{"interest": i, "model": n, "money_saved": F.money_saved(y, p[n], loan, i)} for n in ev["models"]]
    return pd.DataFrame(rows)


def focus_model(b: Board) -> str:
    choice = st.session_state.get("focus", "auto")
    return b.champion if choice in (None, "auto") else choice


@st.cache_resource(show_spinner="Preparing exact explanations…")
def explainer(name: str):
    """Exact reason-code explainer for supported models (None otherwise)."""
    from quickloan import explain as X
    _, ev = load_artifacts()
    if not X.supported(name):
        return None
    return X.ReasonExplainer(name, ev["fitted_models"][name], ev["X_train"])
