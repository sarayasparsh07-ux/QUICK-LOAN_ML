"""Project-wide constants. Everything that more than one module needs lives here."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
import os  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_CANDIDATES = ["loans.csv", "loans(1).csv"]          # brief ships it as loans(1).csv sometimes
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_ARTIFACT = ROOT / "loan_model.joblib"               # deployed model, chosen on training CV (contract below)
CHAMPION_ARTIFACT = ROOT / "loan_model_leaderboard_champion.joblib"   # holdout leaderboard winner (brief's "best model")
EVAL_ARTIFACT = ARTIFACT_DIR / "evaluation_results.joblib"  # everything the dashboard plots
LEADERBOARD_CSV = ROOT / "model_leaderboard.csv"


def model_artifact_path() -> Path:
    """Overridable with QUICKLOAN_MODEL_PATH (used by the automated dashboard tests)."""
    return Path(os.environ.get("QUICKLOAN_MODEL_PATH", MODEL_ARTIFACT))


def eval_artifact_path() -> Path:
    return Path(os.environ.get("QUICKLOAN_EVAL_PATH", EVAL_ARTIFACT))

ARTIFACT_SCHEMA_VERSION = "quickloan-artifact/1.2"
EVAL_SCHEMA_VERSION = "quickloan-eval/1.2"

# ---------------------------------------------------------------- data schema
TARGET = "default"                        # 1 = defaulted (positive class), 0 = repaid
CATEGORICAL = ["home_ownership"]
NUMERIC = [
    "age", "annual_income", "loan_amount", "loan_term_months", "employment_years",
    "credit_history_months", "num_existing_loans", "past_defaults", "debt_to_income",
]
RAW_FEATURES = ["age", "annual_income", "loan_amount", "loan_term_months", "employment_years",
                "credit_history_months", "num_existing_loans", "past_defaults", "debt_to_income",
                "home_ownership"]
HOME_CATEGORIES = ["rent", "own", "mortgage"]
LOAN_TERMS = [12, 24, 36, 48, 60]
DTI_CAP = 3.00                            # observed maximum of debt_to_income in the dataset

# ---------------------------------------------------------------- split (assignment-mandated)
TEST_SIZE = 0.25
RANDOM_STATE = 1
CV_FOLDS = 5

# Instructor reference values for the common split (reproducibility checks only)
REFERENCE_SPLIT = {
    "n_test": 1500, "n_def": 188, "n_good": 1312,
    "total_def_loans": 12_271_900.0, "total_good_loans": 62_945_000.0,
}
# Instructor dashboard results at 20 % (benchmarks to compare against, never to reproduce)
REFERENCE_RESULTS = {
    "Decision Tree": {"acc": 0.8947, "recall": 0.2766, "money": 4225200 - 0.2 * 1495200},
    "Logistic Regression": {"acc": 0.8880, "recall": 0.2181, "money": 3648800 - 0.2 * 1707500},
    "KNN": {"acc": 0.8833, "recall": 0.1383, "money": 2447000 - 0.2 * 1129900},
    "Voting": {"acc": 0.8913, "recall": 0.2181, "money": 3746200 - 0.2 * 1372600},
    "Bagging": {"acc": 0.8967, "recall": 0.3138, "money": 4956600 - 0.2 * 2119300},
    "Random Forest": {"acc": 0.8960, "recall": 0.3138, "money": 5033400 - 0.2 * 2179200},
    "AdaBoost": {"acc": 0.8907, "recall": 0.1968, "money": 3207100 - 0.2 * 1139900},
    "Gradient Boosting": {"acc": 0.8927, "recall": 0.3298, "money": 5241600 - 0.2 * 2780100},
    "XGBoost": {"acc": 0.8947, "recall": 0.3351, "money": 5339700 - 0.2 * 2563800},
    "Stacking": {"acc": 0.8940, "recall": 0.3085, "money": 5022100 - 0.2 * 2321400},
}

# ---------------------------------------------------------------- finance
DEFAULT_INTEREST = 0.20                   # foregone interest as a share of principal (not an APR)
INTEREST_MIN, INTEREST_MAX, INTEREST_STEP = 0.05, 0.50, 0.01
BENCHMARK_THRESHOLD = 0.50

# ---------------------------------------------------------------- models
MODEL_ORDER = ["Decision Tree", "Logistic Regression", "KNN", "Voting", "Bagging",
               "Random Forest", "AdaBoost", "Gradient Boosting", "XGBoost", "Stacking"]
FAMILY = {
    "Decision Tree": "Single", "Logistic Regression": "Single", "KNN": "Single",
    "Voting": "Voting", "Bagging": "Bagging", "Random Forest": "Bagging",
    "AdaBoost": "Boosting", "Gradient Boosting": "Boosting", "XGBoost": "Boosting",
    "Stacking": "Stacking",
}
# Qualitative interpretability for the recommendation narrative (documented judgement, not a metric)
INTERPRETABILITY = {
    "Decision Tree": ("High", "a readable set of if-then rules"),
    "Logistic Regression": ("High", "signed coefficients per feature"),
    "KNN": ("Medium", "explained by similar past applicants, but no global rules"),
    "Voting": ("Low", "an average of three different models"),
    "Bagging": ("Low", "hundreds of bootstrapped trees"),
    "Random Forest": ("Low", "hundreds of de-correlated trees; needs post-hoc explanation"),
    "AdaBoost": ("Medium-Low", "a weighted sequence of shallow trees"),
    "Gradient Boosting": ("Low", "a long sequence of residual-fitting trees"),
    "XGBoost": ("Low", "a regularised boosted ensemble; needs post-hoc explanation"),
    "Stacking": ("Very low", "a meta-model over several ensembles"),
}

# ---------------------------------------------------------------- design tokens (shared by EDA + dashboard)
PALETTE = {
    "bg0": "#080F1C", "bg1": "#0B1424", "bg2": "#101D31", "bg3": "#16243A",
    "teal": "#29D3C2", "emerald": "#35D6A0", "cyan": "#64D6F5",
    "blue": "#638CFF", "violet": "#A78BFA", "amber": "#F4BD64",
    "pos": "#35D6A0", "neg": "#FF727F", "warn": "#F3B75D", "neutral": "#8B9BB2",
    "t1": "#F4F7FC", "t2": "#AAB8CD", "t3": "#72829A",
}
FAMILY_COLOR = {
    "Single": PALETTE["neutral"], "Voting": PALETTE["violet"], "Bagging": PALETTE["blue"],
    "Boosting": PALETTE["cyan"], "Stacking": PALETTE["amber"],
}
