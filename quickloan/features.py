"""Feature engineering.

Every engineered feature is a deterministic, row-wise function of the applicant's own inputs, so it
cannot leak target information and needs no fitting. It lives inside the model pipeline, which means
the saved artifact and the dashboard compute it identically.

None of these is a true instalment (EMI): the dataset has no interest rate or amortisation schedule,
so principal-only burden proxies are the honest choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# name -> (formula, financial meaning)
CANDIDATES: dict[str, tuple[str, str]] = {
    "monthly_principal_burden": (
        "loan_amount / loan_term_months",
        "Principal the borrower must retire each month (AED). A principal-only proxy, not an EMI."),
    "annual_principal_burden_ratio": (
        "(loan_amount / loan_term_months * 12) / annual_income",
        "Share of yearly income consumed by principal repayment. Combines size, term and income."),
    "credit_history_years": (
        "credit_history_months / 12",
        "Same information in years. A linear rescaling, so it should add nothing - kept as a control."),
    "log_annual_income": (
        "ln(annual_income)",
        "Diminishing marginal effect of income; tames the right-skewed income distribution."),
    "has_past_default": (
        "1 if past_defaults > 0 else 0",
        "Any prior default at all - credit officers treat the first default as the key signal."),
    "employment_share_of_adult_life": (
        "employment_years / max(age - 18, 1)",
        "Career stability relative to age: 5 years means more at 24 than at 60."),
}


def add_features(df: pd.DataFrame, names: list[str] | tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for n in names:
        if n == "monthly_principal_burden":
            out[n] = out["loan_amount"] / out["loan_term_months"]
        elif n == "annual_principal_burden_ratio":
            out[n] = out["loan_amount"] / out["loan_term_months"] * 12 / out["annual_income"]
        elif n == "credit_history_years":
            out[n] = out["credit_history_months"] / 12
        elif n == "log_annual_income":
            out[n] = np.log(out["annual_income"])
        elif n == "has_past_default":
            out[n] = (out["past_defaults"] > 0).astype(float)
        elif n == "employment_share_of_adult_life":
            out[n] = out["employment_years"] / np.maximum(out["age"] - 18, 1)
        else:
            raise KeyError(f"Unknown engineered feature: {n}")
    return out


AGE_DERIVED = ("employment_share_of_adult_life",)      # engineered features that use age


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Pipeline step that appends the chosen engineered features and optionally drops raw inputs
    (used by the no-age fairness variant). Stateless apart from remembering input column names."""

    def __init__(self, features: tuple[str, ...] = (), drop: tuple[str, ...] = ()):
        self.features = features
        self.drop = drop

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(pd.DataFrame(X).columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        out = add_features(pd.DataFrame(X), list(self.features))
        drop = [c for c in getattr(self, "drop", ()) if c in out.columns]
        return out.drop(columns=drop) if drop else out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])
        names = list(input_features) + list(self.features)
        return np.array([n for n in names if n not in getattr(self, "drop", ())], dtype=object)
