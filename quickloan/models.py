"""The ten models as raw-DataFrame-in, probability-out pipelines, plus their (deliberately small) grids.

Every model object accepts the 10 raw input columns, so preprocessing is always inside the artifact:
    FeatureEngineer -> ColumnTransformer(OneHotEncoder(handle_unknown="ignore") + scaler/passthrough) -> estimator
Scaling is applied for Logistic Regression and KNN; trees get the numbers untouched.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .features import FeatureEngineer

try:  # XGBoost is preferred; the brief explicitly allows HistGradientBoosting as a substitute
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False

RS = C.RANDOM_STATE
SCALED = {"Logistic Regression", "KNN"}


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=RS)


def preprocessor(scale: bool) -> ColumnTransformer:
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", categories=[C.HOME_CATEGORIES]), C.CATEGORICAL)],
        remainder=StandardScaler() if scale else "passthrough",
        verbose_feature_names_out=False,
    )


def _estimator(name: str):
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=RS)
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000)
    if name == "KNN":
        return KNeighborsClassifier()
    if name == "Bagging":
        return BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RS), n_estimators=200,
                                 bootstrap=True, random_state=RS, n_jobs=1)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=1)
    if name == "AdaBoost":
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=RS), random_state=RS)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(subsample=0.8, random_state=RS)
    if name == "XGBoost":
        if HAS_XGB:
            return XGBClassifier(eval_metric="logloss", tree_method="hist", random_state=RS, n_jobs=1,
                                 subsample=0.8, colsample_bytree=0.8, min_child_weight=1)
        return HistGradientBoostingClassifier(random_state=RS)  # pragma: no cover
    raise KeyError(name)


def make_pipeline(name: str, features: tuple[str, ...], params: dict | None = None) -> Pipeline:
    pipe = Pipeline([
        ("features", FeatureEngineer(features=tuple(features))),
        ("prep", preprocessor(scale=name in SCALED)),
        ("clf", _estimator(name)),
    ])
    if params:
        pipe.set_params(**params)
    return pipe


# Small, transparent grids. Scored with 5-fold CV average precision on the TRAINING set only.
GRIDS: dict[str, dict] = {
    "Decision Tree": {"clf__max_depth": [3, 4, 5, 6, 8], "clf__min_samples_leaf": [10, 25, 50],
                      "clf__min_samples_split": [20, 50]},
    "Logistic Regression": {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0]},
    "KNN": {"clf__n_neighbors": [31, 51, 75, 101, 151, 201], "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2]},
    "Bagging": {"clf__max_samples": [0.6, 1.0], "clf__estimator__min_samples_leaf": [1, 5, 10]},
    "Random Forest": {"clf__max_depth": [None, 10], "clf__min_samples_leaf": [1, 5, 10],
                      "clf__max_features": ["sqrt", 0.5], "clf__min_samples_split": [2]},
    "AdaBoost": {"clf__estimator__max_depth": [1, 2], "clf__n_estimators": [100, 300],
                 "clf__learning_rate": [0.1, 0.5]},
    "Gradient Boosting": {"clf__learning_rate": [0.03, 0.1], "clf__n_estimators": [100, 300],
                          "clf__max_depth": [2, 3]},
    "XGBoost": ({"clf__learning_rate": [0.03, 0.1], "clf__n_estimators": [200, 400],
                 "clf__max_depth": [3, 4], "clf__reg_lambda": [1.0, 5.0]} if HAS_XGB else
                {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [3, None]}),
}
TUNED_DIRECTLY = list(GRIDS)


def make_voting(singles: dict) -> VotingClassifier:
    """Soft vote of the three tuned single models (DT, LR, KNN all expose predict_proba)."""
    return VotingClassifier(
        estimators=[("dt", singles["Decision Tree"]), ("lr", singles["Logistic Regression"]),
                    ("knn", singles["KNN"])],
        voting="soft", n_jobs=1)


def make_stacking(tuned: dict) -> StackingClassifier:
    """Strong, diverse bases + a logistic 'manager'. cv=5 means the manager only ever sees
    out-of-fold base predictions, so it never learns from a base model's training-set fit."""
    return StackingClassifier(
        estimators=[("rf", tuned["Random Forest"]), ("xgb", tuned["XGBoost"]),
                    ("lr", tuned["Logistic Regression"])],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=cv_splitter(), stack_method="predict_proba", passthrough=False, n_jobs=1)


XGB_LABEL = "XGBoost" if HAS_XGB else "HistGradientBoosting (XGBoost substitute)"
