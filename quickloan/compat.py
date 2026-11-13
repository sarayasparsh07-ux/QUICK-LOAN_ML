"""Compatibility with the instructor's starter dashboard (app.py supplied with the brief).

The starter app loads ``M = joblib.load("loan_model.joblib")`` and uses ``M["model"]`` and ``M["columns"]``:

    encode(df) = pd.get_dummies(df, columns=["home_ownership"]).reindex(columns=COLUMNS, fill_value=0)
    model.predict(encode(df)) / model.predict_proba(encode(df))

Our pipelines take the raw ten columns (home_ownership as text). ``StarterCompatible`` wraps a fitted pipeline
so that it accepts *either* format: raw columns, or the starter's one-hot ``home_ownership_<level>`` columns,
which it folds back into the single text column before calling the pipeline. Predictions are identical to
calling the pipeline on raw inputs (unit-tested).

This module deliberately imports nothing from the rest of the package: model artifacts are pickled
"by value" (see artifacts.save_portable), so they load where this project's code is not installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

HOME_LEVELS = ("mortgage", "own", "rent")                  # pd.get_dummies orders levels alphabetically
NUMERIC = ("age", "annual_income", "loan_amount", "loan_term_months", "employment_years",
           "credit_history_months", "num_existing_loans", "past_defaults", "debt_to_income")
RAW_ORDER = NUMERIC + ("home_ownership",)
STARTER_COLUMNS = list(NUMERIC) + [f"home_ownership_{h}" for h in HOME_LEVELS]


class StarterCompatible(BaseEstimator, ClassifierMixin):
    """Fitted-pipeline wrapper that accepts raw or starter-app (one-hot) inputs. ``predict`` uses the
    standard 0.50 cut-off, exactly like any scikit-learn classifier; the business threshold lives in the
    artifact's ``thresholds`` entry."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    # The wrapped pipeline is already fitted; fit() refits it only if someone explicitly asks.
    def fit(self, X, y):
        self.pipeline.fit(self.to_raw(X), y)
        return self

    @property
    def classes_(self):
        return self.pipeline.classes_

    @staticmethod
    def to_raw(X) -> pd.DataFrame:
        df = pd.DataFrame(X).copy()
        if "home_ownership" in df.columns and df["home_ownership"].map(lambda v: isinstance(v, str)).all():
            pass                                                              # already raw text
        else:
            dummies = [f"home_ownership_{h}" for h in HOME_LEVELS]
            present = [c for c in dummies if c in df.columns]
            if present:
                M = np.column_stack([df[c].astype(float).to_numpy() if c in df.columns else np.zeros(len(df))
                                     for c in dummies])
                level = np.array(HOME_LEVELS, dtype=object)[M.argmax(axis=1)]
                df["home_ownership"] = np.where(M.max(axis=1) > 0, level, "unknown")   # unknown -> all-zero one-hot
            elif "home_ownership" not in df.columns:
                df["home_ownership"] = "unknown"
            else:
                df["home_ownership"] = df["home_ownership"].astype(str)
        missing = [c for c in NUMERIC if c not in df.columns]
        if missing:
            raise ValueError(f"missing input columns: {missing}")
        return df[list(RAW_ORDER)]

    def predict_proba(self, X):
        return self.pipeline.predict_proba(self.to_raw(X))

    def predict(self, X):
        return self.pipeline.predict(self.to_raw(X))
