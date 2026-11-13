"""Leakage and correctness tests for the shared selection procedure and nested cross-validation.

The full nested run (5 outer x 5 inner folds, all ten models) takes about 20 minutes, so these tests run the *same
code* on a small stratified sample with two models and tiny grids. What is checked does not depend on size:
which rows each stage sees, where thresholds come from, reproducibility and aggregation. The integration
tests at the end check the stored results of the real run.
"""
import numpy as np
import pandas as pd
import pytest

from quickloan import config as C, data as D, finance as F, models as M, training as T

NAMES = ["Decision Tree", "Logistic Regression"]
TINY = {"Decision Tree": {"clf__max_depth": [3, 5], "clf__min_samples_leaf": [25]},
        "Logistic Regression": {"clf__C": [0.03, 1.0]}}


@pytest.fixture(scope="module")
def small_split():
    """A stratified 1,500-row sample of the official training set plus the official test set."""
    full = D.make_split(D.load_loans())
    Xs = full.X_train.groupby(full.y_train, group_keys=False).sample(frac=1 / 3, random_state=0)
    return D.Split(Xs, full.X_test, full.y_train.loc[Xs.index], full.y_test), full


@pytest.fixture(scope="module")
def run_small(small_split):
    split, _ = small_split
    seen = []
    orig_select, orig_grids = T.select_deployment, dict(M.GRIDS)

    def spy(sp, verbose=True, names=None):
        seen.append({"train": sp.X_train.index.to_numpy(), "other": sp.X_test.index.to_numpy()})
        return orig_select(sp, verbose=verbose, names=names)

    M.GRIDS.update(TINY)
    T.select_deployment = spy
    try:
        res1 = T.nested_cv(split, verbose=False, names=NAMES, outer_folds=3, keep_models=True)
        res2 = T.nested_cv(split, verbose=False, names=NAMES, outer_folds=3)
    finally:
        T.select_deployment = orig_select
        M.GRIDS.clear(); M.GRIDS.update(orig_grids)
    return res1, res2, seen


def test_outer_folds_partition_training_rows(run_small, small_split):
    res, _, _ = run_small
    split, _ = small_split
    allv = np.concatenate([i["outer_val"] for i in res["indices"]])
    assert len(allv) == len(set(allv)) == len(split.X_train)          # every row validated exactly once
    for i in res["indices"]:
        assert not set(i["outer_train"]) & set(i["outer_val"])        # disjoint
        assert set(i["outer_train"]) | set(i["outer_val"]) == set(split.X_train.index)


def test_selection_never_sees_outer_validation_or_test_rows(run_small, small_split):
    res, _, seen = run_small
    split, full = small_split
    test_idx = set(full.X_test.index)
    sel_calls = seen[:3]                                               # first nested run, 3 outer folds
    for call, i in zip(sel_calls, res["indices"]):
        assert set(call["train"]) == set(i["outer_train"])             # selection got exactly the outer-train rows
        assert not set(call["train"]) & set(i["outer_val"])
        assert not set(call["train"]) & test_idx                       # official holdout never used


def test_inner_folds_stay_inside_outer_training(run_small):
    res, _, _ = run_small
    for m, i in zip(res["models"], res["indices"]):
        n = len(i["outer_train"])
        # out-of-fold scores exist for every outer-train row and only for them
        for scores in m["selection"]["oof"].values():
            assert len(scores) == n


def test_threshold_comes_from_inner_oof_only(run_small, small_split):
    res, _, _ = run_small
    split, _ = small_split
    for m, i, (_, row) in zip(res["models"], res["indices"], res["folds"].iterrows()):
        sel = m["selection"]
        tr = split.X_train.loc[i["outer_train"]]
        y_tr, loan_tr = split.y_train.loc[i["outer_train"]].to_numpy(), tr["loan_amount"].to_numpy()
        t, _ = F.best_threshold(y_tr, sel["oof"][sel["deployed"]], loan_tr, C.DEFAULT_INTEREST)
        assert row["threshold"] == t == sel["threshold"]


def test_preprocessing_fitted_on_outer_training_only(run_small, small_split):
    res, _, _ = run_small
    split, _ = small_split
    for m, i in zip(res["models"], res["indices"]):
        pipe = m["fitted"]["Logistic Regression"]
        Xt = pipe.named_steps["features"].transform(split.X_train.loc[i["outer_train"]])
        num = Xt.drop(columns="home_ownership")
        scaler = pipe.named_steps["prep"].named_transformers_["remainder"]
        assert np.allclose(scaler.mean_, num.to_numpy(dtype=float).mean(axis=0))


def test_reproducible(run_small):
    a, b, _ = run_small
    pd.testing.assert_frame_equal(a["folds"].reset_index(drop=True), b["folds"].reset_index(drop=True))
    pd.testing.assert_frame_equal(a["per_model"].reset_index(drop=True), b["per_model"].reset_index(drop=True))


def test_aggregation(run_small):
    res, _, _ = run_small
    f, s = res["folds"], res["summary"]
    assert s["procedure_mean_money_per_1000"] == pytest.approx(f["money_per_1000"].mean())
    assert s["procedure_total_money"] == pytest.approx(f["money_saved"].sum())
    assert sum(s["selection_counts"].values()) == len(f) == s["outer_folds"]
    assert f["selected"].all()
    pm = res["per_model_summary"]
    assert pm["times_selected"].sum() == len(f)
    for name, g in res["per_model"].groupby("model"):
        assert pm.loc[name, "mean_money_per_1000"] == pytest.approx(g["money_per_1000"].mean())


def test_outer_money_uses_official_formula(run_small, small_split):
    res, _, _ = run_small
    split, _ = small_split
    for m, i, (_, row) in zip(res["models"], res["indices"], res["folds"].iterrows()):
        Xv = split.X_train.loc[i["outer_val"]]
        yv = split.y_train.loc[i["outer_val"]].to_numpy()
        s = m["fitted"][row["model"]].predict_proba(Xv)[:, 1]
        assert row["money_saved"] == pytest.approx(F.money_saved(yv, (s >= row["threshold"]).astype(int),
                                                                 Xv["loan_amount"].to_numpy()))


def test_choose_deployment_rule():
    cvs = pd.DataFrame({"model": ["B", "A", "C"], "cv_money_saved_business": [10.0, 10.0, 5.0]})
    assert T.choose_deployment(cvs) == "A"                              # tie -> alphabetical, deterministic
    cvs.loc[2, "cv_money_saved_business"] = 11.0
    assert T.choose_deployment(cvs) == "C"


# ------------------------------------------------------------------ the real run (stored in the artifacts)
HAS = C.EVAL_ARTIFACT.exists()


@pytest.mark.skipif(not HAS, reason="artifacts not built yet")
def test_stored_nested_cv_is_complete_and_consistent():
    from quickloan.artifacts import load_eval_artifact, load_model_artifact
    ev, art = load_eval_artifact(), load_model_artifact()
    n = ev["nested_cv"]
    s, f = n["summary"], n["folds"]
    assert s["outer_folds"] == len(f) == 5 and s["inner_folds"] == 5
    assert set(n["per_model"]["model"]) == set(C.MODEL_ORDER)
    official = D.make_split(D.load_loans())
    seen = set()
    for i in n["indices"]:                                             # original row labels of the 6,000-row dataset
        assert set(i["outer_train"]) | set(i["outer_val"]) == set(official.X_train.index)
        assert not (set(i["outer_train"]) | set(i["outer_val"])) & set(official.X_test.index)
        seen |= set(i["outer_val"])
    assert len(seen) == len(ev["y_train"]) == s["n_training_loans"]
    assert s["procedure_mean_money_per_1000"] == pytest.approx(f["money_per_1000"].mean())
    assert art["metrics"]["nested_cv"]["procedure_mean_money_per_1000"] == pytest.approx(s["procedure_mean_money_per_1000"])


@pytest.mark.skipif(not HAS, reason="artifacts not built yet")
def test_final_selection_matches_rule_and_threshold_diagnostics():
    from quickloan.artifacts import load_eval_artifact, load_model_artifact
    ev, art = load_eval_artifact(), load_model_artifact()
    cvs = ev["studies"]["cv_summary"]
    assert art["model_name"] == ev["deployed_model"] == T.choose_deployment(cvs)
    td = ev["threshold_diagnostics"]
    assert td["chosen"] == art["thresholds"]["business"]
    assert td["band_99"][0] <= td["chosen"] <= td["band_99"][1]
    assert td["curve"]["money_saved"].max() == pytest.approx(td["max_money"])
