"""End-to-end training workflow, shared by train_models.py and the notebook so both produce identical results.

Rule enforced throughout: the test set is touched exactly once per model, for final evaluation.
Feature engineering, class-weighting, hyperparameters, thresholds, deployment choice and feature
importance are all decided on the training set (5-fold stratified CV or an internal validation fold).
"""
from __future__ import annotations

import hashlib
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate, train_test_split

from . import config as C
from . import data as D
from . import evaluation as E
from . import finance as F
from . import models as M
from .compat import STARTER_COLUMNS, StarterCompatible
from .features import CANDIDATES


def _log(msg, verbose=True):
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- feature engineering study
def feature_engineering_study(split: D.Split, verbose=True) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add each candidate feature on its own to a linear and a boosted reference model; keep a feature
    only if it raises mean CV average precision by more than 0.002. Then confirm the combined set."""
    refs = {"Logistic Regression": {"clf__C": 1.0},
            "Gradient Boosting": {"clf__n_estimators": 100, "clf__max_depth": 3, "clf__learning_rate": 0.1}}
    cv = M.cv_splitter()

    def score(feats):
        out = {}
        for name, p in refs.items():
            r = cross_validate(M.make_pipeline(name, feats, p), split.X_train, split.y_train, cv=cv,
                               scoring={"ap": "average_precision", "auc": "roc_auc"}, n_jobs=2)
            out[name] = (r["test_ap"].mean(), r["test_auc"].mean())
        return out

    base = score(())
    rows = [{"feature_set": "original 10 features", "formula": "-", "LR_pr_auc": base["Logistic Regression"][0],
             "GB_pr_auc": base["Gradient Boosting"][0], "LR_roc_auc": base["Logistic Regression"][1],
             "GB_roc_auc": base["Gradient Boosting"][1], "delta_mean_pr_auc": 0.0, "decision": "baseline"}]
    keep = []
    for feat, (formula, _) in CANDIDATES.items():
        s = score((feat,))
        delta = np.mean([s[m][0] - base[m][0] for m in refs])
        ok = delta > 0.002
        keep += [feat] if ok else []
        rows.append({"feature_set": f"+ {feat}", "formula": formula,
                     "LR_pr_auc": s["Logistic Regression"][0], "GB_pr_auc": s["Gradient Boosting"][0],
                     "LR_roc_auc": s["Logistic Regression"][1], "GB_roc_auc": s["Gradient Boosting"][1],
                     "delta_mean_pr_auc": delta, "decision": "candidate" if ok else "rejected"})
        _log(f"  FE {feat:32s} dPR-AUC {delta:+.4f}", verbose)
    selected: tuple[str, ...] = ()
    if keep:
        s = score(tuple(keep))
        delta = np.mean([s[m][0] - base[m][0] for m in refs])
        accepted = delta > 0.002
        rows.append({"feature_set": "+ " + " + ".join(keep), "formula": "combined",
                     "LR_pr_auc": s["Logistic Regression"][0], "GB_pr_auc": s["Gradient Boosting"][0],
                     "LR_roc_auc": s["Logistic Regression"][1], "GB_roc_auc": s["Gradient Boosting"][1],
                     "delta_mean_pr_auc": delta, "decision": "RETAINED" if accepted else "rejected (combined)"})
        if accepted:
            selected = tuple(keep)
            for r in rows:
                if r["feature_set"].lstrip("+ ") in keep:
                    r["decision"] = "RETAINED"
    return pd.DataFrame(rows), selected


# --------------------------------------------------------------------------- tuning
def tune_one(name: str, split: D.Split, features, verbose=True) -> tuple[object, dict, dict]:
    """GridSearchCV (5-fold stratified, average precision) for one model. Returns the unfitted best
    pipeline, its parameters and a summary row (including the full grid results)."""
    t0 = time.time()
    gs = GridSearchCV(M.make_pipeline(name, features), M.GRIDS[name], scoring="average_precision",
                      cv=M.cv_splitter(), n_jobs=2, refit=False)
    gs.fit(split.X_train, split.y_train)
    row = {"model": name, "best_params": gs.best_params_, "cv_pr_auc": gs.best_score_,
           "n_candidates": len(gs.cv_results_["params"]), "seconds": round(time.time() - t0, 1)}
    _log(f"  tuned {name:20s} PR-AUC {gs.best_score_:.4f}  {gs.best_params_}", verbose)
    grid = pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    row["_grid"] = grid.sort_values("rank_test_score").reset_index(drop=True)
    return M.make_pipeline(name, features, gs.best_params_), gs.best_params_, row


ENSEMBLE_PARAMS = {
    "Voting": {"voting": "soft", "members": "tuned Decision Tree + Logistic Regression + KNN"},
    "Stacking": {"bases": "tuned Random Forest + XGBoost + Logistic Regression",
                 "meta": "LogisticRegression", "cv": "5-fold out-of-fold predict_proba"},
}


VOTING_MEMBERS = ("Decision Tree", "Logistic Regression", "KNN")
STACKING_MEMBERS = ("Random Forest", "XGBoost", "Logistic Regression")


def tune(split: D.Split, features, verbose=True, names=None) -> tuple[dict, dict, pd.DataFrame]:
    """Tune the eight directly tuned models, then build Voting and Stacking from the tuned components.
    Returns unfitted best estimators in the canonical model order. `names` restricts the candidate set
    (only the fast leakage tests use it; the project always tunes all ten)."""
    wanted = set(C.MODEL_ORDER if names is None else names)
    best, params, rows = {}, {}, []
    for name in M.TUNED_DIRECTLY:
        if name not in wanted:
            continue
        best[name], params[name], row = tune_one(name, split, features, verbose)
        row.pop("_grid")
        rows.append(row)
    if "Voting" in wanted and set(VOTING_MEMBERS) <= set(best):
        best["Voting"] = M.make_voting(best)
        params["Voting"] = ENSEMBLE_PARAMS["Voting"]
    if "Stacking" in wanted and set(STACKING_MEMBERS) <= set(best):
        best["Stacking"] = M.make_stacking(best)
        params["Stacking"] = ENSEMBLE_PARAMS["Stacking"]
    ordered = {n: best[n] for n in C.MODEL_ORDER if n in best}
    return ordered, params, pd.DataFrame(rows)


# --------------------------------------------------------------------------- class weighting study
def class_weight_study(split: D.Split, tuned: dict, verbose=True) -> pd.DataFrame:
    """Effect of re-weighting the minority class at the 0.50 threshold (training CV only)."""
    pos_w = float((split.y_train == 0).sum() / (split.y_train == 1).sum())
    variants = {
        "Decision Tree": {"clf__class_weight": "balanced"},
        "Logistic Regression": {"clf__class_weight": "balanced"},
        "Random Forest": {"clf__class_weight": "balanced"},
        "XGBoost": {"clf__scale_pos_weight": pos_w} if M.HAS_XGB else None,
    }
    rows = []
    for name, p in variants.items():
        if p is None:
            continue
        for label, est in [("none", tuned[name]), ("balanced", clone(tuned[name]).set_params(**p))]:
            s = cross_val_predict(est, split.X_train, split.y_train, cv=M.cv_splitter(),
                                  method="predict_proba", n_jobs=2)[:, 1]
            pred = (s >= 0.5).astype(int)
            m = E.classification_metrics(split.y_train, pred, s)
            rows.append({"model": name, "class_weight": label, "cv_accuracy": m["accuracy"],
                         "cv_recall": m["recall"], "cv_precision": m["precision"], "cv_pr_auc": m["pr_auc"],
                         "cv_money_saved": F.money_saved(split.y_train, pred, split.loan_train, 0.20)})
        _log(f"  class-weight study {name}", verbose)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- OOF scores & thresholds
def oof_scores(split: D.Split, tuned: dict, verbose=True) -> dict[str, np.ndarray]:
    out = {}
    for name, est in tuned.items():
        out[name] = cross_val_predict(est, split.X_train, split.y_train, cv=M.cv_splitter(),
                                      method="predict_proba", n_jobs=2)[:, 1]
        _log(f"  OOF scores {name}", verbose)
    return out


def cv_summary(split: D.Split, oof: dict) -> pd.DataFrame:
    rows = []
    for name, s in oof.items():
        pred = (s >= C.BENCHMARK_THRESHOLD).astype(int)
        m = E.classification_metrics(split.y_train, pred, s)
        t, money_t = F.best_threshold(split.y_train, s, split.loan_train, C.DEFAULT_INTEREST)
        rows.append({"model": name, "family": C.FAMILY[name], "cv_accuracy": m["accuracy"],
                     "cv_recall": m["recall"], "cv_roc_auc": m["roc_auc"], "cv_pr_auc": m["pr_auc"],
                     "cv_brier": m["brier"],
                     "cv_money_saved_050": F.money_saved(split.y_train, pred, split.loan_train, C.DEFAULT_INTEREST),
                     "business_threshold": t, "cv_money_saved_business": money_t})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- the model-selection procedure
SELECTION_PROCEDURE = (
    "1) engineered features chosen by 5-fold CV PR-AUC; 2) each model's hyperparameters tuned by 5-fold "
    "GridSearchCV (average precision); 3) 5-fold out-of-fold default scores for every tuned model; 4) each "
    "model's business threshold = the cut-off maximising out-of-fold money saved (official formula, 20 %); "
    "5) deployment model = highest out-of-fold money saved at its own business threshold. Uses only the rows it "
    "is given.")


def choose_deployment(cvs: pd.DataFrame) -> str:
    """Step 5 of the procedure: highest out-of-fold money saved at the model's own business threshold
    (ties broken by model name so the choice is deterministic)."""
    ranked = cvs.assign(_neg=-cvs["cv_money_saved_business"]).sort_values(["_neg", "model"], kind="mergesort")
    return str(ranked.iloc[0]["model"])


def select_deployment(split: D.Split, verbose=True, names=None) -> dict:
    """The complete training-only selection procedure. Touches split.X_train / y_train only - never
    split.X_test - so it can be nested inside an outer cross-validation loop."""
    fe_table, features = feature_engineering_study(split, verbose)
    _log(f"Engineered features retained: {features or 'none'}", verbose)
    tuned, params, tuning_table = tune(split, features, verbose, names=names)
    oof = oof_scores(split, tuned, verbose)
    cvs = cv_summary(split, oof)
    deployed = choose_deployment(cvs)
    threshold = float(cvs.set_index("model").loc[deployed, "business_threshold"])
    return {"fe_table": fe_table, "features": features, "tuned": tuned, "params": params,
            "tuning_table": tuning_table, "oof": oof, "cv_summary": cvs, "deployed": deployed,
            "threshold": threshold}


# --------------------------------------------------------------------------- nested cross-validation
NESTED_OUTER_FOLDS = 5
NESTED_OUTER_SEED = 2     # outer split seed; differs from the inner seed (1) so outer and inner folds are unrelated


def _fold_metrics(y, pred, loan, interest=C.DEFAULT_INTEREST) -> dict:
    m = E.classification_metrics(y, pred)
    b = F.breakdown(y, pred, loan, interest)
    n = len(y)
    return {"money_saved": F.money_saved(y, pred, loan, interest), "money_per_1000": F.money_saved(y, pred, loan, interest) / n * 1000,
            "accuracy": m["accuracy"], "recall": m["recall"], "precision": m["precision"], "f1": m["f1"],
            "false_rejection_rate": m["fp"] / max(m["fp"] + m["tn"], 1),
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
            "missed_default_principal": b["missed_default_principal"]}


def nested_cv(split: D.Split, verbose=True, names=None, outer_folds=NESTED_OUTER_FOLDS, keep_models=False) -> dict:
    """Assess the *whole* selection procedure without optimism.

    The 4,500 training loans are split into outer folds (the official holdout is never touched). For each
    outer fold, `select_deployment` runs on the outer-training rows only - feature selection, tuning, inner
    out-of-fold scores, thresholds and the model choice all happen inside. The chosen pipeline is refitted on
    the outer-training rows and scored once on the untouched outer-validation rows at the chosen threshold.
    Every inner-tuned model is also scored there at its own inner threshold, giving an honest per-model view."""
    from sklearn.model_selection import StratifiedKFold
    X, y = split.X_train, split.y_train
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=NESTED_OUTER_SEED)
    folds, per_model, indices, models = [], [], [], []
    t0 = time.time()
    for k, (tr, va) in enumerate(outer.split(X, y), 1):
        inner = D.Split(X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va])
        sel = select_deployment(inner, verbose=False, names=names)
        cvs = sel["cv_summary"].set_index("model")
        yv, lv = inner.y_test.to_numpy(), inner.loan_test
        fitted_k = {}
        for name, est in sel["tuned"].items():
            mdl = clone(est).fit(inner.X_train, inner.y_train)
            s = mdl.predict_proba(inner.X_test)[:, 1]
            t = float(cvs.loc[name, "business_threshold"])
            r = _fold_metrics(yv, (s >= t).astype(int), lv)
            r.update(fold=k, model=name, threshold=t, selected=(name == sel["deployed"]),
                     inner_cv_money_per_1000=float(cvs.loc[name, "cv_money_saved_business"]) / len(tr) * 1000,
                     money_at_050=F.money_saved(yv, (s >= C.BENCHMARK_THRESHOLD).astype(int), lv))
            per_model.append(r)
            fitted_k[name] = mdl
        ch = next(r for r in per_model if r["fold"] == k and r["selected"])
        ranking = cvs["cv_money_saved_business"].sort_values(ascending=False)
        folds.append({**ch, "features": ", ".join(sel["features"]) or "none",
                      "runner_up": ranking.index[1] if len(ranking) > 1 else None,
                      "inner_margin_per_1000": float(ranking.iloc[0] - ranking.iloc[1]) / len(tr) * 1000 if len(ranking) > 1 else np.nan,
                      "n_outer_train": len(tr), "n_outer_val": len(va), "n_val_defaulters": int(yv.sum())})
        indices.append({"fold": k, "outer_train": X.index[tr].to_numpy(), "outer_val": X.index[va].to_numpy()})
        if keep_models:
            models.append({"fold": k, "selection": sel, "fitted": fitted_k})
        _log(f"  nested fold {k}/{outer_folds}: selected {sel['deployed']} @ {sel['threshold']:.2f} -> outer-val "
             f"AED {ch['money_saved']:,.0f} ({ch['money_per_1000']:,.0f} per 1,000 loans)  [{time.time() - t0:.0f}s]", verbose)
    folds = pd.DataFrame(folds)
    per_model = pd.DataFrame(per_model)
    pm = (per_model.groupby("model")
          .agg(mean_money_per_1000=("money_per_1000", "mean"), sd_money_per_1000=("money_per_1000", "std"),
               mean_inner_cv_per_1000=("inner_cv_money_per_1000", "mean"), mean_recall=("recall", "mean"),
               mean_precision=("precision", "mean"), mean_accuracy=("accuracy", "mean"),
               mean_threshold=("threshold", "mean"), sd_threshold=("threshold", "std"),
               times_selected=("selected", "sum"))
          .sort_values("mean_money_per_1000", ascending=False))
    pm["times_selected"] = pm["times_selected"].astype(int)
    counts = folds["model"].value_counts()
    summary = {
        "outer_folds": outer_folds, "inner_folds": C.CV_FOLDS, "outer_seed": NESTED_OUTER_SEED,
        "inner_seed": C.RANDOM_STATE, "n_training_loans": int(len(y)),
        "selection_counts": counts.to_dict(), "modal_model": str(counts.index[0]),
        "modal_share": float(counts.iloc[0] / outer_folds),
        "procedure_mean_money_per_1000": float(folds["money_per_1000"].mean()),
        "procedure_sd_money_per_1000": float(folds["money_per_1000"].std()),
        "procedure_total_money": float(folds["money_saved"].sum()),
        "procedure_mean_inner_cv_per_1000": float(folds["inner_cv_money_per_1000"].mean()),
        "optimism_per_1000": float((folds["inner_cv_money_per_1000"] - folds["money_per_1000"]).mean()),
        "procedure_mean_recall": float(folds["recall"].mean()), "procedure_mean_precision": float(folds["precision"].mean()),
        "procedure_mean_accuracy": float(folds["accuracy"].mean()), "procedure_mean_f1": float(folds["f1"].mean()),
        "procedure_mean_false_rejection": float(folds["false_rejection_rate"].mean()),
        "threshold_mean": float(folds["threshold"].mean()), "threshold_sd": float(folds["threshold"].std()),
        "threshold_min": float(folds["threshold"].min()), "threshold_max": float(folds["threshold"].max()),
        "procedure_money_at_050_total": float(folds["money_at_050"].sum()),
        "best_model_by_outer_money": str(pm.index[0]),
        "seconds": round(time.time() - t0, 1),
        "procedure": SELECTION_PROCEDURE,
    }
    out = {"folds": folds, "per_model": per_model, "per_model_summary": pm, "summary": summary, "indices": indices}
    if keep_models:
        out["models"] = models
    return out


# --------------------------------------------------------------------------- threshold stability (final model)
def threshold_diagnostics(split: D.Split, oof: np.ndarray, threshold: float, interest=C.DEFAULT_INTEREST) -> dict:
    """How stable is the deployed model's business threshold? Uses training out-of-fold scores only:
    per-inner-fold optimal cut-offs, the near-optimal band, and the trade-offs around the chosen cut-off."""
    y, loan = split.y_train.to_numpy(), split.loan_train
    grid = F.THRESHOLD_GRID
    money = np.array([F.money_saved(y, (oof >= t).astype(int), loan, interest) for t in grid])
    best = money.max()
    per_fold = []
    for k, (_, va) in enumerate(M.cv_splitter().split(split.X_train, split.y_train), 1):
        t_k, m_k = F.best_threshold(y[va], oof[va], loan[va], interest)
        per_fold.append({"fold": k, "best_threshold": t_k, "money_at_best": m_k,
                         "money_at_chosen": F.money_saved(y[va], (oof[va] >= threshold).astype(int), loan[va], interest)})
    rows = []
    for t in grid:
        p = (oof >= t).astype(int)
        m = E.classification_metrics(y, p)
        rows.append({"threshold": float(t), "money_saved": F.money_saved(y, p, loan, interest), "recall": m["recall"],
                     "precision": m["precision"], "false_rejection_rate": m["fp"] / max(m["fp"] + m["tn"], 1),
                     "rejection_rate": float(p.mean())})
    curve = pd.DataFrame(rows)
    band99 = curve.loc[curve["money_saved"] >= 0.99 * best, "threshold"]
    band98 = curve.loc[curve["money_saved"] >= 0.98 * best, "threshold"]
    at = curve.set_index("threshold").loc[round(threshold, 2)]
    pf = pd.DataFrame(per_fold)
    return {"curve": curve, "per_fold": pf, "chosen": float(threshold), "max_money": float(best),
            "band_99": (float(band99.min()), float(band99.max())), "band_98": (float(band98.min()), float(band98.max())),
            "per_fold_threshold_mean": float(pf["best_threshold"].mean()), "per_fold_threshold_sd": float(pf["best_threshold"].std()),
            "at_chosen": at.to_dict(), "interest": interest}


# --------------------------------------------------------------------------- final fit + holdout evaluation
def fit_all(split: D.Split, tuned: dict, verbose=True) -> dict:
    fitted = {}
    for name, est in tuned.items():
        t0 = time.time()
        fitted[name] = clone(est).fit(split.X_train, split.y_train)
        _log(f"  fitted {name:20s} ({time.time() - t0:.1f}s)", verbose)
    return fitted


def holdout_results(split: D.Split, fitted: dict, cvs: pd.DataFrame) -> dict:
    """Score every fitted model on the one common test set at 0.50 and at its CV-chosen threshold."""
    res = {}
    thr = cvs.set_index("model")["business_threshold"].to_dict()
    for name, model in fitted.items():
        s = model.predict_proba(split.X_test)[:, 1]
        p050 = (s >= C.BENCHMARK_THRESHOLD).astype(int)
        # sanity: predict() agrees with the 0.50 threshold (equal-probability ties aside)
        agree = float((model.predict(split.X_test) == p050).mean())
        pbiz = (s >= thr[name]).astype(int)
        res[name] = {"family": C.FAMILY[name], "scores": s, "pred_050": p050, "pred_business": pbiz,
                     "business_threshold": thr[name], "predict_agreement": agree,
                     "curves": E.curves(split.y_test, s),
                     "segments_050": E.segment_table(split.X_test, split.y_test, p050),
                     "segments_business": E.segment_table(split.X_test, split.y_test, pbiz)}
    return res


def leaderboard(split: D.Split, res: dict, interest=C.DEFAULT_INTEREST, which="pred_050") -> pd.DataFrame:
    rows = [E.full_row(n, r["family"], split.y_test, r[which], r["scores"], split.loan_test, interest)
            for n, r in res.items()]
    lb = pd.DataFrame(rows).sort_values("money_saved", ascending=False).reset_index(drop=True)
    lb.insert(0, "rank", np.arange(1, len(lb) + 1))
    return lb


# --------------------------------------------------------------------------- importance (training data only)
def permutation_study(split: D.Split, tuned: dict, verbose=True) -> dict[str, pd.DataFrame]:
    """Refit each model on 80 % of the training set, permute raw inputs on the other 20 %."""
    Xa, Xb, ya, yb = train_test_split(split.X_train, split.y_train, test_size=0.2,
                                      random_state=C.RANDOM_STATE, stratify=split.y_train)
    out = {}
    for name, est in tuned.items():
        m = clone(est).fit(Xa, ya)
        r = permutation_importance(m, Xb, yb, scoring="roc_auc", n_repeats=8,
                                   random_state=C.RANDOM_STATE, n_jobs=2)
        out[name] = (pd.DataFrame({"feature": Xb.columns, "importance_mean": r.importances_mean,
                                   "importance_std": r.importances_std})
                     .sort_values("importance_mean", ascending=False).reset_index(drop=True))
        _log(f"  permutation importance {name}", verbose)
    return out


# --------------------------------------------------------------------------- weak-learner baseline
BASELINE_NAME = "Shallow tree (depth 3, untuned)"


def shallow_tree_baseline(split: D.Split, features) -> dict:
    """The brief's 'feel the pain' weak learner: a depth-3 tree with no tuning. Reported beside - not inside -
    the ten-model leaderboard so the tuned comparison stays intact."""
    pipe = M.make_pipeline("Decision Tree", features, {"clf__max_depth": 3}).fit(split.X_train, split.y_train)
    s = pipe.predict_proba(split.X_test)[:, 1]
    return {BASELINE_NAME: {"family": "Baseline", "scores": s, "pred_050": (s >= C.BENCHMARK_THRESHOLD).astype(int),
                            "params": {"max_depth": 3, "tuned": False}, "pipeline": pipe}}


# --------------------------------------------------------------------------- uncertainty
def bootstrap_study(split: D.Split, res: dict, champion: str, interest=C.DEFAULT_INTEREST) -> dict:
    """Paired bootstrap of the holdout (same resampled loans for every model). Models are not refitted:
    this measures how much the *evaluation sample* moves the money figures."""
    counts = F.bootstrap_counts(len(split.y_test))
    boot = F.bootstrap_money(counts, split.y_test, {n: r["pred_050"] for n, r in res.items()},
                             split.loan_test, interest)
    return {"B": F.BOOT_B, "seed": F.BOOT_SEED, "interest": interest, "policy": "benchmark",
            "summary": F.bootstrap_summary(boot, champion)}


def add_bootstrap_columns(lb: pd.DataFrame, summary: dict) -> pd.DataFrame:
    lb = lb.copy()
    for col in ["ci_low", "ci_high", "p_best", "gap_ci_low", "gap_ci_high", "distinguishable"]:
        lb[f"money_{col}" if col.startswith("ci") else col] = lb["model"].map(lambda m: summary[m][col])
    return lb


# --------------------------------------------------------------------------- interest-dependent thresholds
def threshold_curves(split: D.Split, oof: dict, cvs: pd.DataFrame) -> dict:
    curves = {n: F.threshold_curve(split.y_train, s, split.loan_train) for n, s in oof.items()}
    stored = cvs.set_index("model")["business_threshold"]
    for n, c in curves.items():   # the fast path must reproduce the official per-threshold search exactly
        assert np.isclose(F.threshold_at(c, C.DEFAULT_INTEREST), stored[n]), n
    return curves


def thresholds_by_interest(curve: dict) -> dict[int, float]:
    return {p: F.threshold_at(curve, p / 100) for p in range(int(C.INTEREST_MIN * 100), int(C.INTEREST_MAX * 100) + 1)}


# --------------------------------------------------------------------------- fairness: no-age variant
def without_age(est):
    """Clone any of the ten models with age (and the age-derived engineered feature) removed everywhere,
    including inside Voting/Stacking members."""
    from .features import AGE_DERIVED
    e = clone(est)
    upd = {}
    for k, v in e.get_params(deep=True).items():
        if k == "features__drop" or k.endswith("__features__drop"):
            upd[k] = ("age",)
        elif k == "features__features" or k.endswith("__features__features"):
            upd[k] = tuple(f for f in v if f not in AGE_DERIVED)
    return e.set_params(**upd)


def fairness_study(split: D.Split, tuned: dict, res: dict, names: list[str], verbose=True) -> dict:
    out = {}
    for name in names:
        est = without_age(tuned[name])
        oof_na = cross_val_predict(est, split.X_train, split.y_train, cv=M.cv_splitter(),
                                   method="predict_proba", n_jobs=2)[:, 1]
        t_na, _ = F.best_threshold(split.y_train, oof_na, split.loan_train, C.DEFAULT_INTEREST)
        fitted = clone(est).fit(split.X_train, split.y_train)
        s_na = fitted.predict_proba(split.X_test)[:, 1]
        variants = {"with_age": (res[name]["scores"], res[name]["business_threshold"]),
                    "without_age": (s_na, t_na)}
        rows, segs = [], {}
        for label, (s, t) in variants.items():
            for policy, thr in [("0.50 benchmark", C.BENCHMARK_THRESHOLD), ("CV business", t)]:
                p = (s >= thr).astype(int)
                r = E.full_row(name, C.FAMILY[name], split.y_test, p, s, split.loan_test, C.DEFAULT_INTEREST)
                r.update(variant=label, policy=policy, threshold=thr)
                rows.append(r)
                seg = E.segment_table(split.X_test, split.y_test, p)
                segs[(label, policy)] = seg[seg["dimension"] == "Age band"].reset_index(drop=True)
        out[name] = {"table": pd.DataFrame(rows), "segments": segs, "threshold_without_age": t_na,
                     "features_without_age": [c for c in fitted[:-1].get_feature_names_out()]
                     if hasattr(fitted, "__getitem__") else None}
        _log(f"  no-age variant {name}: threshold {t_na:.2f}", verbose)
    return out


# --------------------------------------------------------------------------- calibration
ECE_TOLERANCE = 0.02      # 2 percentage points of mean |observed - predicted| default rate


def calibration_study(split: D.Split, est, raw_oof: np.ndarray, raw_test: np.ndarray, verbose=True) -> dict:
    """Wrap the deployed pipeline in CalibratedClassifierCV. Method and whether calibration is justified are
    decided on training out-of-fold predictions; the holdout only confirms."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss
    ytr, yte = split.y_train.to_numpy(), split.y_test.to_numpy()
    cand = {"raw": {"oof_brier": brier_score_loss(ytr, raw_oof), "oof_ece": E.expected_calibration_error(ytr, raw_oof)}}
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(estimator=clone(est), method=method, cv=M.cv_splitter())
        s = cross_val_predict(cal, split.X_train, split.y_train, cv=M.cv_splitter(), method="predict_proba", n_jobs=2)[:, 1]
        cand[method] = {"oof_brier": brier_score_loss(ytr, s), "oof_ece": E.expected_calibration_error(ytr, s)}
        _log(f"  calibration {method}: OOF Brier {cand[method]['oof_brier']:.4f}  ECE {cand[method]['oof_ece']:.4f}", verbose)
    best = min(("sigmoid", "isotonic"), key=lambda m: cand[m]["oof_brier"])
    justified = bool(cand[best]["oof_brier"] < cand["raw"]["oof_brier"] and cand[best]["oof_ece"] < cand["raw"]["oof_ece"])
    source = best if justified else "raw"
    # a probability is displayed only if the representation actually used is well calibrated on training OOF data
    prob_ok = bool(cand[source]["oof_ece"] <= ECE_TOLERANCE)
    fitted = CalibratedClassifierCV(estimator=clone(est), method=best, cv=M.cv_splitter()).fit(split.X_train, split.y_train)
    s_cal = fitted.predict_proba(split.X_test)[:, 1]
    return {"method": best, "justified": justified, "candidates": cand, "pipeline": fitted if justified else None,
            "probability_source": source, "probability_justified": prob_ok, "ece_tolerance": ECE_TOLERANCE,
            "holdout": {"raw_brier": brier_score_loss(yte, raw_test), "cal_brier": brier_score_loss(yte, s_cal),
                        "raw_ece": E.expected_calibration_error(yte, raw_test),
                        "cal_ece": E.expected_calibration_error(yte, s_cal)},
            "curve_raw": E.curves(yte, raw_test)["calibration"], "curve_cal": E.curves(yte, s_cal)["calibration"],
            "test_scores_cal": s_cal,
            "rule": ("calibrate only if training-OOF Brier AND ECE both improve (method = lower OOF Brier); otherwise keep "
                     f"the raw score. Show it as a probability only if its training-OOF ECE <= {ECE_TOLERANCE:.0%}.")}


# --------------------------------------------------------------------------- the whole thing
def run(data_path=None, verbose=True) -> dict:
    t0 = time.time()
    df = D.load_loans(data_path)
    audit = D.quality_audit(df)
    split = D.make_split(df)
    check = D.reconcile_split(split)
    if not all(v["match"] for v in check.values()):
        _log("WARNING: split does not reconcile with the instructor reference - investigate.", verbose)
    _log(f"Data {df.shape}, test {len(split.y_test)} rows, split reconciles: "
         f"{all(v['match'] for v in check.values())}", verbose)

    _log("Training-only selection procedure on the full training set: features, tuning, OOF scores, thresholds", verbose)
    sel = select_deployment(split, verbose)
    features, fe_table = sel["features"], sel["fe_table"]
    tuned, params, tuning_table = sel["tuned"], sel["params"], sel["tuning_table"]
    oof, cvs = sel["oof"], sel["cv_summary"]

    _log("Class-weighting study (training CV)", verbose)
    cw_table = class_weight_study(split, tuned, verbose)

    _log("Final fit on the full training set", verbose)
    fitted = fit_all(split, tuned, verbose)

    _log("Holdout evaluation (test set used once)", verbose)
    res = holdout_results(split, fitted, cvs)
    lb = leaderboard(split, res)
    lb_biz = leaderboard(split, res, which="pred_business")

    _log("Permutation importance (training-internal validation fold)", verbose)
    perm = permutation_study(split, tuned, verbose)

    deployed = sel["deployed"]
    champion = lb.iloc[0]["model"]
    _log(f"Deployed model (highest CV money saved at its CV threshold): {deployed} | holdout champion: {champion}", verbose)

    out = {"df": df, "audit": audit, "split": split, "split_check": check, "features": features,
           "fe_table": fe_table, "tuned": tuned, "params": params, "tuning_table": tuning_table,
           "cw_table": cw_table, "oof": oof, "cv_summary": cvs, "fitted": fitted, "results": res,
           "leaderboard": lb, "leaderboard_business": lb_biz, "permutation": perm, "deployed": deployed,
           "champion": champion, "data_path": str(D.resolve_data_path(data_path))}
    extend(out, verbose)
    _log(f"Nested cross-validation of the selection procedure ({NESTED_OUTER_FOLDS} outer x {C.CV_FOLDS} inner folds, "
         "training set only)", verbose)
    out["nested_cv"] = nested_cv(split, verbose)
    _log(f"Done in {time.time() - t0:.0f}s", verbose)
    return out


def extend(out: dict, verbose=True) -> dict:
    """Stages added after the core bake-off (shared by the script and the notebook)."""
    split, res = out["split"], out["results"]
    _log("Shallow-tree baseline, bootstrap, threshold curves", verbose)
    out["baselines"] = shallow_tree_baseline(split, out["features"])
    out["bootstrap"] = bootstrap_study(split, res, out["champion"])
    out["leaderboard"] = add_bootstrap_columns(out["leaderboard"], out["bootstrap"]["summary"])
    out["threshold_curves"] = threshold_curves(split, out["oof"], out["cv_summary"])
    thr = float(out["cv_summary"].set_index("model").loc[out["deployed"], "business_threshold"])
    out["threshold_diagnostics"] = threshold_diagnostics(split, out["oof"][out["deployed"]], thr)
    _log("Fairness study: no-age variants (training CV + holdout)", verbose)
    names = list(dict.fromkeys([out["deployed"], out["champion"]]))
    out["fairness"] = fairness_study(split, out["tuned"], res, names, verbose)
    _log(f"Calibration study for the deployed model ({out['deployed']})", verbose)
    out["calibration"] = calibration_study(split, out["tuned"][out["deployed"]], out["oof"][out["deployed"]],
                                           res[out["deployed"]]["scores"], verbose)
    return out


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    v = {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
         "pandas": pd.__version__, "numpy": np.__version__}
    for pkg in ("xgboost", "pyarrow", "scipy", "joblib", "cloudpickle", "shap", "streamlit"):
        try:
            v[pkg] = version(pkg)
        except PackageNotFoundError:
            v[pkg] = "not installed"
    return v


SELECTION_RULES = {
    "training_cv": "Highest 5-fold out-of-fold money saved on the training set at the model's own "
                   "CV-optimised threshold (20 % interest). The test set played no part.",
    "holdout": "Highest holdout money saved at the 0.50 benchmark and 20 % interest - the assignment's "
               "leaderboard winner ('save your best model'). Because the holdout chose it, its holdout "
               "figures are optimistically biased; use loan_model.joblib for a leakage-free deployment.",
}


def build_model_artifact(run_out: dict, which: str = "deployed") -> dict:
    """which = 'deployed' -> loan_model.joblib ; 'champion' -> loan_model_leaderboard_champion.joblib"""
    name = run_out["deployed"] if which == "deployed" else run_out["champion"]
    selected_on = "training_cv" if which == "deployed" else "holdout"
    split, res, cvs = run_out["split"], run_out["results"], run_out["cv_summary"].set_index("model")
    r = res[name]
    with open(run_out["data_path"], "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    cal_err = E.expected_calibration_error(split.y_test, r["scores"])
    cal = run_out.get("calibration") if which == "deployed" else None
    calibration = ({k: v for k, v in cal.items() if k not in ("pipeline", "test_scores_cal")} if cal else
                   {"method": None, "justified": False, "note": "calibration is only fitted for the deployed model"})
    curve = run_out["threshold_curves"][name]
    return {
        "schema_version": C.ARTIFACT_SCHEMA_VERSION,
        "model_name": name, "family": C.FAMILY[name],
        "display_name": M.XGB_LABEL if name == "XGBoost" else name,
        "selected_on": selected_on,
        "pipeline": run_out["fitted"][name],
        "model": StarterCompatible(run_out["fitted"][name]),          # works with the instructor's starter app
        "starter_compatible": True,
        "calibrated_pipeline": cal["pipeline"] if cal else None,
        "calibration": calibration,
        "raw_features": list(C.RAW_FEATURES), "columns": list(STARTER_COLUMNS),
        "categorical_levels": {"home_ownership": list(C.HOME_CATEGORIES)},
        "engineered_features": list(run_out["features"]),
        "observed_ranges": D.observed_ranges(run_out["df"]),
        "thresholds": {"benchmark": C.BENCHMARK_THRESHOLD, "business": float(r["business_threshold"]),
                       "business_interest": C.DEFAULT_INTEREST,
                       "business_by_interest": thresholds_by_interest(curve),
                       "business_rule": "maximises training out-of-fold money saved at the given interest"},
        "interest_assumption": C.DEFAULT_INTEREST,
        "score_type": {"type": "uncalibrated_risk_score", "decision_uses": "raw score vs threshold",
                       "holdout_ece_raw": cal_err,
                       "probability_available": bool(cal and cal["probability_justified"]),
                       "probability_source": cal["probability_source"] if cal else None,
                       "note": "Decisions compare the raw model score with the threshold. A calibrated "
                               "probability of default is attached only where calibration was justified."},
        "selection_rule": SELECTION_RULES[selected_on],
        "selection_methodology": {
            "procedure": SELECTION_PROCEDURE,
            "assessment": (f"Nested cross-validation: {NESTED_OUTER_FOLDS} outer folds (seed {NESTED_OUTER_SEED}) over the "
                           f"4,500 training loans; the full procedure re-run inside each outer-training part with "
                           f"{C.CV_FOLDS} inner folds (seed {C.RANDOM_STATE}); scored once on each untouched outer fold. "
                           "The official 1,500-loan holdout is used only for final evaluation."),
            "final": ("The procedure re-run on all 4,500 training loans selected this model and threshold."
                      if which == "deployed" else
                      "Not chosen by this procedure: this is the 0.50 holdout leaderboard champion (evaluation A). The "
                      f"procedure above selected {run_out['deployed']} (loan_model.joblib)."),
        },
        "metrics": {
            "nested_cv": {k: v for k, v in run_out["nested_cv"]["summary"].items() if k != "procedure"},
            "threshold_stability": ({k: v for k, v in run_out["threshold_diagnostics"].items()
                                     if k not in ("curve", "per_fold")} if which == "deployed" else None),
            "cv": cvs.loc[name].to_dict(),
            "test_benchmark": E.full_row(name, C.FAMILY[name], split.y_test, r["pred_050"], r["scores"],
                                         split.loan_test, C.DEFAULT_INTEREST),
            "test_business": E.full_row(name, C.FAMILY[name], split.y_test, r["pred_business"], r["scores"],
                                        split.loan_test, C.DEFAULT_INTEREST),
        },
        "training": {"n_rows": int(len(run_out["df"])), "n_train": int(len(split.y_train)),
                     "n_test": int(len(split.y_test)), "test_size": C.TEST_SIZE,
                     "random_state": C.RANDOM_STATE, "stratified": True, "cv_folds": C.CV_FOLDS,
                     "dataset_sha256": sha, "versions": _versions(),
                     "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "best_params": run_out["params"][name]},
    }


def build_eval_results(run_out: dict) -> dict:
    split, res = run_out["split"], run_out["results"]
    audit = dict(run_out["audit"])
    return {
        "schema_version": C.EVAL_SCHEMA_VERSION,
        "y_test": split.y_test.to_numpy(), "loan_test": split.loan_test,
        "X_test": split.X_test.reset_index(drop=True),
        "models": {n: {k: v for k, v in r.items()} for n, r in res.items()},
        "fitted_models": run_out["fitted"],
        "params": run_out["params"],
        "deployed_model": run_out["deployed"],
        "features": list(run_out["features"]),
        "split_check": run_out["split_check"],
        "audit": audit,
        "observed_ranges": D.observed_ranges(run_out["df"]),
        "studies": {"feature_engineering": run_out["fe_table"], "tuning": run_out["tuning_table"],
                    "class_weight": run_out["cw_table"], "cv_summary": run_out["cv_summary"]},
        "permutation": run_out["permutation"],
        "leaderboard_050": run_out["leaderboard"],
        "leaderboard_business": run_out["leaderboard_business"],
        "reference_results": C.REFERENCE_RESULTS,
        "eda": eda_summary(run_out["df"]),
        "xgb_label": M.XGB_LABEL,
        "versions": _versions(),
        # ---- uncertainty, thresholds, baseline, fairness, calibration, reason-code inputs
        "champion_050_20": run_out["champion"],
        "X_train": split.X_train.reset_index(drop=True), "y_train": split.y_train.to_numpy(),
        "loan_train": split.loan_train,
        "oof": run_out["oof"],
        "threshold_curves": run_out["threshold_curves"],
        "baselines": {n: {k: v for k, v in b.items() if k != "pipeline"} for n, b in run_out["baselines"].items()},
        "bootstrap": run_out["bootstrap"],
        "fairness": run_out["fairness"],
        "calibration": {k: v for k, v in run_out["calibration"].items() if k != "pipeline"},
        "calibrated_pipeline": run_out["calibration"]["pipeline"],
        "nested_cv": {k: v for k, v in run_out["nested_cv"].items() if k != "models"},
        "threshold_diagnostics": run_out["threshold_diagnostics"],
    }


def eda_summary(df: pd.DataFrame) -> dict:
    """Small aggregates the Methodology page charts (no row-level data needed)."""
    d = df.copy()
    # bands stay ordered categoricals so groupby returns them in numeric, not alphabetical, order
    d["age_band"] = E.age_band(d["age"])
    d["income_band"] = pd.qcut(d["annual_income"], 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
    d["dti_band"] = pd.cut(d["debt_to_income"], [0, .25, .5, .75, 1.0, 1.5, 3.01],
                           labels=["≤0.25", "0.25–0.5", "0.5–0.75", "0.75–1.0", "1.0–1.5", ">1.5"])
    d["past_defaults_band"] = pd.Categorical(d["past_defaults"].clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"}),
                                             categories=["0", "1", "2", "3+"], ordered=True)
    out = {}
    for col in ["home_ownership", "age_band", "income_band", "dti_band", "past_defaults_band", "loan_term_months"]:
        g = d.groupby(col, observed=True)["default"].agg(["mean", "size"]).reset_index()
        g.columns = ["group", "default_rate", "n"]
        g["group"] = g["group"].astype(str)
        out[col] = g
    out["corr"] = df.drop(columns="home_ownership").corr()
    return out
