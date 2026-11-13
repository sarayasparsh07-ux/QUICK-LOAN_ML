"""Data loading, validation, the data-quality audit and the one common train/test split."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


class DataValidationError(ValueError):
    """Raised when the dataset breaks a hard requirement (schema, labels, positivity)."""


def resolve_data_path(path: str | Path | None = None) -> Path:
    """Find loans.csv. Accepts an explicit path or the names the brief uses (loans.csv / loans(1).csv)."""
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
        return p
    for name in C.DATA_CANDIDATES:
        for base in (Path.cwd(), C.ROOT):
            p = base / name
            if p.exists():
                return p
    raise FileNotFoundError("loans.csv not found. Place it next to train_models.py / app.py.")


def load_loans(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(resolve_data_path(path))
    validate(df)
    df["home_ownership"] = df["home_ownership"].astype(str).str.strip().str.lower()
    return df


def validate(df: pd.DataFrame) -> None:
    """Hard checks. Anything that fails here would invalidate the whole analysis."""
    expected = set(C.RAW_FEATURES + [C.TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns: {sorted(missing)}")
    if df[list(expected)].isna().any().any():
        raise DataValidationError("Dataset contains missing values.")
    if not set(df[C.TARGET].unique()) <= {0, 1}:
        raise DataValidationError("Target must be 0/1.")
    if (df["loan_amount"] <= 0).any() or (df["annual_income"] <= 0).any():
        raise DataValidationError("loan_amount and annual_income must be positive.")


def quality_audit(df: pd.DataFrame) -> dict:
    """Data-quality summary used by the notebook and the Methodology page."""
    num = df[C.NUMERIC]
    desc = num.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    outliers = ((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).sum()

    dti_calc = (df["loan_amount"] / df["annual_income"]).round(2)
    mismatch = df.loc[~np.isclose(dti_calc, df["debt_to_income"])].copy()
    mismatch["raw_ratio"] = (mismatch["loan_amount"] / mismatch["annual_income"]).round(4)
    capped = mismatch["debt_to_income"].eq(C.DTI_CAP) & mismatch["raw_ratio"].gt(C.DTI_CAP)
    mismatch["reason"] = np.where(capped, "capped at 3.00", "half-way rounding")

    # Logical plausibility: years employed cannot exceed years since age 16 (minimum working age)
    implausible = df["employment_years"] > (df["age"] - 16)

    sanity = {
        "loan_amount_positive": bool((df["loan_amount"] > 0).all()),
        "annual_income_positive": bool((df["annual_income"] > 0).all()),
        "target_binary": bool(set(df[C.TARGET].unique()) <= {0, 1}),
        "home_ownership_known": bool(set(df["home_ownership"].unique()) <= set(C.HOME_CATEGORIES)),
        "employment_le_age_minus_16": bool((~implausible).all()),
        "loan_term_in_allowed_set": bool(df["loan_term_months"].isin(C.LOAN_TERMS).all()),
        "non_negative_counts": bool((df[["num_existing_loans", "past_defaults",
                                         "credit_history_months", "employment_years"]] >= 0).all().all()),
    }
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "categories": df["home_ownership"].value_counts().to_dict(),
        "target_counts": df[C.TARGET].value_counts().sort_index().to_dict(),
        "default_rate": float(df[C.TARGET].mean()),
        "describe": desc,
        "iqr_outliers": outliers.to_dict(),
        "dti_mismatch": mismatch[["loan_amount", "annual_income", "debt_to_income", "raw_ratio", "reason"]],
        "sanity": sanity,
        "implausible_employment": {
            "n": int(implausible.sum()), "share": float(implausible.mean()),
            "default_rate": float(df.loc[implausible, "default"].mean()) if implausible.any() else float("nan"),
            "max_age": int(df.loc[implausible, "age"].max()) if implausible.any() else None,
            "decision": "kept - flagged, not removed (no evidence of which field is wrong; removal would "
                        "change the assignment's reference split)"},
    }


def observed_ranges(df: pd.DataFrame) -> dict:
    """Min / max of each numeric input - used by the simulator to warn on extrapolation."""
    return {c: (float(df[c].min()), float(df[c].max())) for c in C.NUMERIC}


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def loan_test(self) -> np.ndarray:
        # Taken from X_test itself, so it is aligned row-by-row with every prediction by construction.
        return self.X_test["loan_amount"].to_numpy()

    @property
    def loan_train(self) -> np.ndarray:
        return self.X_train["loan_amount"].to_numpy()


def make_split(df: pd.DataFrame) -> Split:
    """The single, assignment-mandated split used by every model (25 % test, random_state=1, stratified)."""
    X, y = df[C.RAW_FEATURES], df[C.TARGET]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=y)
    return Split(Xtr, Xte, ytr, yte)


def reconcile_split(split: Split) -> dict:
    """Compare the split with the instructor's reference values."""
    yte, loan = split.y_test.to_numpy(), split.loan_test
    actual = {
        "n_test": int(len(yte)), "n_def": int(yte.sum()), "n_good": int((yte == 0).sum()),
        "total_def_loans": float(loan[yte == 1].sum()), "total_good_loans": float(loan[yte == 0].sum()),
    }
    return {k: {"reference": C.REFERENCE_SPLIT[k], "actual": v,
                "match": bool(np.isclose(C.REFERENCE_SPLIT[k], v))} for k, v in actual.items()}
