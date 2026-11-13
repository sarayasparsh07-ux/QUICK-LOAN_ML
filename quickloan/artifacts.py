"""Artifact contract.

loan_model.joblib  (deployed model, consumed by the simulator)
    schema_version        "quickloan-artifact/1.0"
    model_name / family   deployed model, chosen by training-set cross-validation only
    pipeline              fitted sklearn object: raw 10-column DataFrame -> predict_proba
    model                 StarterCompatible(pipeline): accepts raw columns OR the instructor starter app's
                          one-hot home_ownership_<level> columns; identical predictions (quickloan/compat.py)
    raw_features          ordered input schema (the dashboard builds exactly this DataFrame)
    columns               the starter app's encoded schema: 9 numeric + home_ownership_{mortgage,own,rent}
    categorical_levels    {"home_ownership": [...]} known levels; unknown -> all-zero one-hot + UI warning
    engineered_features   names appended inside the pipeline
    observed_ranges       {feature: (min, max)} from the full dataset, for extrapolation warnings
    thresholds            {"benchmark": 0.50, "business": t* at 20 %, "business_interest": 0.20,
                           "business_by_interest": {5: t, ..., 50: t}}  (all from training OOF scores)
    interest_assumption   0.20 (foregone interest as share of principal)
    score_type            decisions use the uncalibrated score vs threshold; calibration evidence attached
    calibrated_pipeline   CalibratedClassifierCV around the same pipeline, or None if calibration did not help
    calibration           {"method", "justified", "probability_source" ("raw" or the calibrator),
                           "probability_justified", oof/holdout Brier + ECE for raw and calibrated}
    selected_on           "training_cv" (loan_model.joblib) or "holdout" (leaderboard-champion artifact)
    selection_methodology {"procedure", "assessment" (nested CV), "final"}
    metrics               {"nested_cv": summary of the 5x5 nested CV of the selection procedure,
                           "threshold_stability": near-optimal band + per-fold cut-offs (deployed model),
                           "cv": ..., "test_benchmark": ..., "test_business": ...}
    training              split settings, row counts, library versions (incl. pyarrow), dataset sha256, timestamp

loan_model_leaderboard_champion.joblib  (same contract) - the holdout leaderboard winner at 0.50 / 20 %,
    shipped because the brief says "save your best model". Chosen *on* the holdout, so its holdout
    figures are optimistic; loan_model.joblib remains the leakage-free deployment choice.

Both model artifacts are written with `save_portable`: the project's own classes are pickled *by value*
(cloudpickle, a joblib dependency), so `joblib.load("loan_model.joblib")` works in a folder that contains only
the instructor's app.py and loans.csv - exactly the layout the starter app describes. The library versions in
requirements.txt are still required, as for any scikit-learn pickle.

artifacts/evaluation_results.joblib  (everything the dashboard plots; see training.build_eval_results)
"""
from __future__ import annotations

from pathlib import Path

import joblib

from . import config as C

REQUIRED_MODEL_KEYS = {"schema_version", "model_name", "family", "pipeline", "raw_features",
                       "categorical_levels", "thresholds", "interest_assumption", "observed_ranges",
                       "metrics", "training", "score_type", "selected_on", "calibration"}
REQUIRED_EVAL_KEYS = {"schema_version", "models", "y_test", "loan_test", "deployed_model",
                      "split_check", "audit", "studies", "reference_results", "oof", "threshold_curves",
                      "baselines", "fairness", "bootstrap", "X_train", "y_train", "loan_train",
                      "nested_cv", "threshold_diagnostics"}


class ArtifactError(RuntimeError):
    """Missing or incompatible artifact."""


def save_portable(obj: dict, path: Path) -> None:
    """Self-contained model artifact: package classes pickled by value, gzip-compressed, readable by plain
    joblib.load() without this project's code on the path."""
    import gzip

    import cloudpickle

    import quickloan
    path.parent.mkdir(parents=True, exist_ok=True)
    cloudpickle.register_pickle_by_value(quickloan)
    try:
        with gzip.open(path, "wb", compresslevel=6) as fh:
            cloudpickle.dump(obj, fh, protocol=5)
    finally:
        cloudpickle.unregister_pickle_by_value(quickloan)


def version_mismatches(saved: dict, packages=("scikit-learn", "xgboost", "numpy", "pandas", "pyarrow")) -> dict:
    """{package: (saved_version, installed_version)} for libraries whose versions differ from training."""
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for pkg in packages:
        try:
            now = version(pkg)
        except PackageNotFoundError:
            now = "not installed"
        if pkg in saved and saved[pkg] != now:
            out[pkg] = (saved[pkg], now)
    return out


def save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)


def _load(path: Path, required: set, version: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"{path.name} not found. Run `python train_models.py` first.")
    try:
        obj = joblib.load(path)
    except Exception as e:  # corrupt file or library mismatch
        raise ArtifactError(f"{path.name} could not be loaded ({type(e).__name__}: {e}). "
                            "Re-run `python train_models.py` in this environment.") from e
    if not isinstance(obj, dict):
        raise ArtifactError(f"{path.name} is not a QuickLoan artifact (got {type(obj).__name__}).")
    missing = required - set(obj)
    if missing:
        raise ArtifactError(f"{path.name} is missing keys {sorted(missing)} - it was produced by an "
                            "older or different script. Re-run `python train_models.py`.")
    if obj["schema_version"] != version:
        raise ArtifactError(f"{path.name} has schema {obj['schema_version']!r}; expected {version!r}.")
    return obj


def load_model_artifact(path: Path | None = None) -> dict:
    obj = _load(path or C.model_artifact_path(), REQUIRED_MODEL_KEYS, C.ARTIFACT_SCHEMA_VERSION)
    if list(obj["raw_features"]) != C.RAW_FEATURES:
        raise ArtifactError("Artifact input schema does not match the dashboard schema.")
    return obj


def load_eval_artifact(path: Path | None = None) -> dict:
    return _load(path or C.eval_artifact_path(), REQUIRED_EVAL_KEYS, C.EVAL_SCHEMA_VERSION)
