"""Unit + integration tests for the enhancements: interest-dependent thresholds, bootstrap uncertainty,
calibration metric, no-age variant, reason codes and the two-artifact contract."""
import numpy as np
import pandas as pd
import pytest

from quickloan import artifacts as A, config as C, evaluation as E, finance as F, models as M, training as T
from quickloan.features import FeatureEngineer

RNG = np.random.default_rng(7)
Y = RNG.integers(0, 2, 400)
S = np.clip(Y * 0.3 + RNG.random(400) * 0.7, 0, 1)
L = RNG.integers(5_000, 150_000, 400).astype(float)
HAS_ART = C.MODEL_ARTIFACT.exists() and C.EVAL_ARTIFACT.exists() and C.CHAMPION_ARTIFACT.exists()


# ------------------------------------------------------------------ thresholds
@pytest.mark.parametrize("interest", np.round(np.arange(0.05, 0.501, 0.01), 2))
def test_fast_threshold_equals_official_search(interest):
    curve = F.threshold_curve(Y, S, L)
    assert F.threshold_at(curve, interest) == F.best_threshold(Y, S, L, interest)[0]


def test_curve_reproduces_money_saved():
    c = F.threshold_curve(Y, S, L)
    for t, a, b in zip(c["grid"], c["caught"], c["wrong"]):
        assert F.money_saved(Y, (S >= t).astype(int), L, 0.2) == pytest.approx(a - 0.2 * b)


def test_break_even():
    assert F.break_even_threshold(0.20) == pytest.approx(1 / 6)


# ------------------------------------------------------------------ bootstrap
def test_bootstrap_sums_equal_official_money_on_resamples():
    counts = F.bootstrap_counts(len(Y), B=25, seed=3)
    rng = np.random.default_rng(3)
    idx = rng.integers(0, len(Y), size=(25, len(Y)))
    p = (S >= 0.5).astype(int)
    boot = F.bootstrap_money(counts, Y, {"m": p}, L, 0.2)["m"]
    assert np.allclose(boot, [F.money_saved(Y[i], p[i], L[i], 0.2) for i in idx])


def test_bootstrap_summary_properties():
    counts = F.bootstrap_counts(len(Y), B=300, seed=1)
    preds = {"a": (S >= 0.5).astype(int), "b": (S >= 0.7).astype(int), "c": np.zeros_like(Y)}
    boot = F.bootstrap_money(counts, Y, preds, L)
    summ = F.bootstrap_summary(boot, "a")
    assert sum(v["p_best"] for v in summ.values()) == pytest.approx(1.0)
    assert summ["a"]["gap_ci_low"] == summ["a"]["gap_ci_high"] == 0
    assert not summ["a"]["distinguishable"]
    point = F.money_saved(Y, preds["a"], L)
    assert summ["a"]["ci_low"] <= point <= summ["a"]["ci_high"]


# ------------------------------------------------------------------ calibration metric
def test_ece_zero_for_perfect_and_positive_for_biased():
    y = np.array([0] * 90 + [1] * 10)
    assert E.expected_calibration_error(y, np.full(100, 0.10), n_bins=1) == pytest.approx(0.0)
    assert E.expected_calibration_error(y, np.full(100, 0.50), n_bins=1) == pytest.approx(0.40)


# ------------------------------------------------------------------ no-age variant
def _raw(n=30):
    return pd.DataFrame({"age": RNG.integers(21, 64, n), "annual_income": 90000.0, "loan_amount": 40000.0,
                         "loan_term_months": 36, "employment_years": 4.0, "credit_history_months": 60,
                         "num_existing_loans": 1, "past_defaults": RNG.integers(0, 3, n), "debt_to_income": 0.44,
                         "home_ownership": "rent"})[C.RAW_FEATURES]


def test_feature_engineer_drop():
    fe = FeatureEngineer(features=("has_past_default",), drop=("age",)).fit(_raw())
    out = fe.transform(_raw())
    assert "age" not in out.columns and "has_past_default" in out.columns
    assert "age" not in fe.get_feature_names_out()


def test_without_age_strips_age_from_every_member():
    feats = ("has_past_default", "employment_share_of_adult_life")
    tuned = {n: M.make_pipeline(n, feats) for n in M.TUNED_DIRECTLY}
    for est in [T.without_age(tuned["Gradient Boosting"]), T.without_age(M.make_stacking(tuned)),
                T.without_age(M.make_voting(tuned))]:
        p = est.get_params(deep=True)
        drops = [v for k, v in p.items() if k.endswith("features__drop") or k == "features__drop"]
        engs = [v for k, v in p.items() if k.endswith("features__features") or k == "features__features"]
        assert drops and all(d == ("age",) for d in drops)
        assert all("employment_share_of_adult_life" not in e for e in engs)


# ------------------------------------------------------------------ integration with built artifacts
art_only = pytest.mark.skipif(not HAS_ART, reason="artifacts not built yet")


@pytest.fixture(scope="module")
def ev():
    return A.load_eval_artifact()


@art_only
def test_two_artifacts_and_their_selection(ev):
    dep, champ = A.load_model_artifact(C.MODEL_ARTIFACT), A.load_model_artifact(C.CHAMPION_ARTIFACT)
    assert dep["selected_on"] == "training_cv" and champ["selected_on"] == "holdout"
    assert dep["model_name"] == ev["deployed_model"] and champ["model_name"] == ev["champion_050_20"]
    lb = pd.read_csv(C.LEADERBOARD_CSV)
    assert lb.iloc[0]["model"] == champ["model_name"]


@art_only
def test_business_thresholds_by_interest_match_curves(ev):
    dep = A.load_model_artifact()
    by = dep["thresholds"]["business_by_interest"]
    assert sorted(by) == list(range(5, 51))
    curve = ev["threshold_curves"][dep["model_name"]]
    for pct, t in by.items():
        assert t == F.threshold_at(curve, pct / 100)
    assert by[20] == dep["thresholds"]["business"]


@art_only
def test_leaderboard_csv_has_consistent_intervals():
    lb = pd.read_csv(C.LEADERBOARD_CSV)
    assert len(lb) == 10 and T.BASELINE_NAME not in set(lb["model"])
    assert (lb["money_ci_low"] <= lb["money_saved"]).all() and (lb["money_saved"] <= lb["money_ci_high"]).all()
    assert lb["p_best"].sum() == pytest.approx(1.0)


@art_only
def test_baseline_is_weaker_than_tuned_tree(ev):
    (name, b), = ev["baselines"].items()
    y, loan = ev["y_test"], ev["loan_test"]
    assert F.money_saved(y, b["pred_050"], loan) < F.money_saved(y, ev["models"]["Decision Tree"]["pred_050"], loan)


@art_only
def test_calibration_contract(ev):
    dep = A.load_model_artifact()
    cal = dep["calibration"]
    assert cal["method"] in ("sigmoid", "isotonic")
    c = cal["candidates"]
    if cal["justified"]:
        p = dep["calibrated_pipeline"].predict_proba(ev["X_test"].head(20))[:, 1]
        assert ((0 <= p) & (p <= 1)).all()
        assert c[cal["method"]]["oof_ece"] < c["raw"]["oof_ece"] and c[cal["method"]]["oof_brier"] < c["raw"]["oof_brier"]
        assert cal["probability_source"] == cal["method"]
    else:
        assert dep["calibrated_pipeline"] is None and cal["probability_source"] == "raw"
        assert not (c[cal["method"]]["oof_ece"] < c["raw"]["oof_ece"] and c[cal["method"]]["oof_brier"] < c["raw"]["oof_brier"])
    src = cal["probability_source"]
    assert cal["probability_justified"] == (c[src]["oof_ece"] <= cal["ece_tolerance"])
    assert dep["score_type"]["probability_available"] == cal["probability_justified"]


@art_only
def test_fairness_variants_present(ev):
    for name, f in ev["fairness"].items():
        assert set(f["table"]["variant"]) == {"with_age", "without_age"}
        if f["features_without_age"]:
            assert "age" not in f["features_without_age"]


@art_only
@pytest.mark.parametrize("name", ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "Logistic Regression"])
def test_reason_codes_are_exact_and_additive(ev, name):
    from quickloan import explain as X
    ex = X.ReasonExplainer(name, ev["fitted_models"][name], ev["X_train"])
    for i in range(0, 1500, 150):
        e = ex.explain(ev["X_test"].iloc[[i]])
        assert e.max_additivity_error < 1e-4
        assert set(e.table["feature"]) <= set(C.RAW_FEATURES) | {"has_past_default", "employment_share_of_adult_life"}


@art_only
def test_unsupported_models_raise():
    from quickloan import explain as X
    for name in ["KNN", "Voting", "Bagging", "AdaBoost", "Stacking"]:
        assert not X.supported(name)
