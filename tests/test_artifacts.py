"""Integration tests on the trained artifacts. Run `python train_models.py` first."""
import joblib
import numpy as np
import pandas as pd
import pytest

from quickloan import artifacts as A, config as C, data as D, finance as F

pytestmark = pytest.mark.skipif(not C.MODEL_ARTIFACT.exists() or not C.EVAL_ARTIFACT.exists(),
                                reason="artifacts not built yet")


@pytest.fixture(scope="module")
def art():
    return A.load_model_artifact()


@pytest.fixture(scope="module")
def ev():
    return A.load_eval_artifact()


@pytest.fixture(scope="module")
def split():
    return D.make_split(D.load_loans())


def test_split_reconciles_with_reference(split):
    assert all(v["match"] for v in D.reconcile_split(split).values())


def test_all_ten_models_present(ev):
    assert list(ev["models"]) == C.MODEL_ORDER


def test_same_test_set_and_alignment(ev, split):
    assert np.array_equal(ev["y_test"], split.y_test.to_numpy())
    assert np.array_equal(ev["loan_test"], split.X_test["loan_amount"].to_numpy())
    for r in ev["models"].values():
        assert len(r["pred_050"]) == len(ev["y_test"]) == len(r["scores"])


def test_leaderboard_csv_matches_official_formula(ev):
    lb = pd.read_csv(C.LEADERBOARD_CSV)
    assert len(lb) == 10
    for _, row in lb.iterrows():
        m = ev["models"][row["model"]]
        assert row["money_saved"] == pytest.approx(F.money_saved(ev["y_test"], m["pred_050"], ev["loan_test"], 0.20))


def test_artifact_contract(art):
    assert A.REQUIRED_MODEL_KEYS <= set(art)
    assert art["raw_features"] == C.RAW_FEATURES
    assert 0 < art["thresholds"]["business"] < 1
    assert art["model_name"] in C.MODEL_ORDER


def test_independent_load_and_predict(art):
    raw = joblib.load(C.MODEL_ARTIFACT)          # plain joblib, no helper
    row = pd.DataFrame([dict(age=35, annual_income=90000.0, loan_amount=40000.0, loan_term_months=36,
                             employment_years=4.0, credit_history_months=60, num_existing_loans=1,
                             past_defaults=0, debt_to_income=0.44, home_ownership="rent")])[C.RAW_FEATURES]
    p = raw["pipeline"].predict_proba(row)[0, 1]
    assert 0 <= p <= 1


def test_unknown_category_does_not_crash(art):
    row = pd.DataFrame([dict(age=35, annual_income=90000.0, loan_amount=40000.0, loan_term_months=36,
                             employment_years=4.0, credit_history_months=60, num_existing_loans=1,
                             past_defaults=0, debt_to_income=0.44, home_ownership="company_housing")])
    assert 0 <= art["pipeline"].predict_proba(row[C.RAW_FEATURES])[0, 1] <= 1


def test_saved_predictions_reproduce(art, ev, split):
    s = art["pipeline"].predict_proba(split.X_test)[:, 1]
    assert np.allclose(s, ev["models"][art["model_name"]]["scores"])


def test_risky_applicant_scores_higher(art):
    base = dict(age=40, annual_income=120000.0, loan_amount=20000.0, loan_term_months=36, employment_years=10.0,
                credit_history_months=200, num_existing_loans=0, past_defaults=0, debt_to_income=0.17,
                home_ownership="own")
    risky = dict(base, annual_income=35000.0, loan_amount=100000.0, debt_to_income=2.86, past_defaults=3,
                 num_existing_loans=4, home_ownership="rent")
    s = art["pipeline"].predict_proba(pd.DataFrame([base, risky])[C.RAW_FEATURES])[:, 1]
    assert s[1] > s[0]


def test_missing_and_incompatible_artifacts(tmp_path):
    with pytest.raises(A.ArtifactError):
        A.load_model_artifact(tmp_path / "nope.joblib")
    bad = tmp_path / "bad.joblib"
    joblib.dump({"model": 1, "columns": []}, bad)          # starter-style artifact
    with pytest.raises(A.ArtifactError):
        A.load_model_artifact(bad)
    joblib.dump([1, 2, 3], bad)
    with pytest.raises(A.ArtifactError):
        A.load_model_artifact(bad)
