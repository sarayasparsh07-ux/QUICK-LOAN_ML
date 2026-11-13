"""Applicant-level reason codes.

Only exact, additive explanations are offered:
  * tree models (Decision Tree, Random Forest, Gradient Boosting, XGBoost): Tree SHAP (path-dependent),
  * Logistic Regression: coefficient x (feature - training mean), exact in log-odds.
For every explanation, base value + sum of contributions reproduces the model's own output - this is
checked on every call. Other model types (KNN, Voting, Bagging, AdaBoost, Stacking) would need sampling-based
approximations that are slow and unstable, so they are reported as unsupported rather than approximated.
Contributions describe what moved *this model's* score; they are not causal effects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KIND = {"Decision Tree": "tree_proba", "Random Forest": "tree_proba",
        "Gradient Boosting": "tree_logodds", "XGBoost": "tree_logodds",
        "Logistic Regression": "linear_logodds"}
UNITS = {"tree_proba": "risk-score points", "tree_logodds": "log-odds", "linear_logodds": "log-odds"}

LABELS = {
    "age": "Age", "annual_income": "Annual income", "loan_amount": "Loan amount", "loan_term_months": "Loan term",
    "employment_years": "Employment years", "credit_history_months": "Credit history",
    "num_existing_loans": "Existing loans", "past_defaults": "Past defaults",
    "debt_to_income": "Loan-to-income", "home_ownership": "Home ownership",
    "has_past_default": "Any past default (derived)",
    "employment_share_of_adult_life": "Career stability: employment ÷ adult years (derived)",
}


def supported(name: str) -> bool:
    return name in KIND


@dataclass
class Explanation:
    table: pd.DataFrame          # feature, label, value, contribution (sorted by |contribution|)
    base: float
    output: float                # model output in `units` space (proba for tree_proba, log-odds otherwise)
    units: str
    max_additivity_error: float


def _group(names: list[str]) -> list[str]:
    """Collapse one-hot columns (home_ownership_rent ...) onto their raw input."""
    return ["home_ownership" if n.startswith("home_ownership_") else n for n in names]


def _fmt(feature: str, row: pd.Series) -> str:
    """Human-readable value from the raw inputs (derived features recomputed unscaled)."""
    if feature == "home_ownership":
        return str(row["home_ownership"])
    if feature in row.index:
        v = row[feature]
    else:
        from .features import add_features
        v = float(add_features(row.to_frame().T.infer_objects(), [feature])[feature].iloc[0])
    if feature in ("annual_income", "loan_amount"):
        return f"AED {v:,.0f}"
    if feature == "loan_term_months":
        return f"{v:.0f} months"
    if feature == "credit_history_months":
        return f"{v:.0f} months"
    if feature == "has_past_default":
        return "yes" if v else "no"
    if feature in ("debt_to_income", "employment_share_of_adult_life"):
        return f"{v:.2f}"
    return f"{v:g}"


class ReasonExplainer:
    def __init__(self, name: str, pipeline, X_train: pd.DataFrame):
        if not supported(name):
            raise ValueError(f"No exact explainer for {name}")
        self.name, self.kind = name, KIND[name]
        self.prep, self.clf = pipeline[:-1], pipeline[-1]
        self.names = list(self.prep.get_feature_names_out())
        if self.kind.startswith("tree"):
            import shap  # local import: only needed when reason codes are requested
            self._shap = shap.TreeExplainer(self.clf)          # tree_path_dependent, exact
        else:
            Xt = np.asarray(self.prep.transform(X_train), dtype=float)
            self.mean = Xt.mean(axis=0)
            self.coef = self.clf.coef_.ravel()
            self.intercept = float(self.clf.intercept_[0])

    def _model_output(self, Xt) -> np.ndarray:
        if self.kind == "tree_proba":
            return self.clf.predict_proba(Xt)[:, 1]
        if self.name == "XGBoost":
            return self.clf.predict(Xt, output_margin=True)
        return self.clf.decision_function(Xt)

    def explain(self, X: pd.DataFrame) -> Explanation:
        Xt = np.asarray(self.prep.transform(X), dtype=float)
        if self.kind == "linear_logodds":
            contrib = (Xt - self.mean) * self.coef
            base = self.intercept + float(self.mean @ self.coef)
        else:
            sv = np.asarray(self._shap.shap_values(Xt))
            ev = np.asarray(self._shap.expected_value, dtype=float).ravel()
            if sv.ndim == 3:                     # (n, features, classes) -> class 1
                sv, ev = sv[:, :, 1], ev[1:2]
            contrib, base = sv, float(ev[-1])
        out = self._model_output(Xt)
        err = float(np.max(np.abs(base + contrib.sum(axis=1) - out)))
        if err > 1e-4:
            raise AssertionError(f"explanation not additive for {self.name}: error {err:.2e}")
        row = X.iloc[0]
        d = pd.DataFrame({"feature": _group(self.names), "contribution": contrib[0]})
        d = d.groupby("feature", sort=False, as_index=False)["contribution"].sum()
        d["label"] = d["feature"].map(LABELS).fillna(d["feature"])
        d["value"] = [_fmt(f, row) for f in d["feature"]]
        d = d.reindex(d["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        return Explanation(d[["feature", "label", "value", "contribution"]], base, float(out[0]),
                           UNITS[self.kind], err)


def reason_codes(exp: Explanation, reject: bool, k: int = 3) -> list[str]:
    """Plain-language top drivers in the direction of the decision."""
    t = exp.table
    side = t[t["contribution"] > 0] if reject else t[t["contribution"] < 0]
    verb = "raised" if reject else "lowered"
    return [f"{r.label} = {r.value} {verb} the risk score" for r in side.head(k).itertuples()]
