"""The model artifacts work with the instructor's original starter dashboard.

tests/fixtures/instructor_starter_app.py is the brief's app.py, byte-for-byte unchanged. The integration tests
recreate the layout the starter app describes - a folder holding only that app.py, loan_model.joblib and
loans.csv - and run it in a separate, isolated Python process (`python -I`), so this project's code cannot be
imported. Both cards must render, and the money card must equal the official money_saved at 0.50.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from quickloan import config as C
from quickloan.artifacts import version_mismatches
from quickloan.compat import HOME_LEVELS, STARTER_COLUMNS, StarterCompatible

STARTER = C.ROOT / "tests" / "fixtures" / "instructor_starter_app.py"
HAS_ART = C.MODEL_ARTIFACT.exists() and C.CHAMPION_ARTIFACT.exists()


def starter_encode(df, columns):
    """The starter app's own encode() - copied verbatim from the fixture."""
    return pd.get_dummies(df, columns=["home_ownership"]).reindex(columns=columns, fill_value=0)


# ------------------------------------------------------------------ unit tests (no artifacts needed)
def _raw(n=8, home=None):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"age": rng.integers(21, 64, n), "annual_income": 90000.0, "loan_amount": 40000.0,
                         "loan_term_months": 36, "employment_years": 4.0, "credit_history_months": 60,
                         "num_existing_loans": 1, "past_defaults": rng.integers(0, 3, n), "debt_to_income": 0.44,
                         "home_ownership": home or list(np.resize(["rent", "own", "mortgage"], n))})[C.RAW_FEATURES]


def test_starter_columns_match_get_dummies_naming():
    enc = pd.get_dummies(_raw(), columns=["home_ownership"])
    assert set(STARTER_COLUMNS) == set(enc.columns)
    assert STARTER_COLUMNS[-3:] == [f"home_ownership_{h}" for h in HOME_LEVELS]


def test_to_raw_round_trips_from_starter_encoding():
    raw = _raw()
    back = StarterCompatible.to_raw(starter_encode(raw, STARTER_COLUMNS))
    pd.testing.assert_frame_equal(back.reset_index(drop=True), raw.reset_index(drop=True), check_dtype=False)


def test_to_raw_passes_raw_input_through():
    raw = _raw()
    pd.testing.assert_frame_equal(StarterCompatible.to_raw(raw), raw)


def test_single_applicant_only_one_dummy_present():
    one = _raw(1, home=["own"])
    enc = starter_encode(one, STARTER_COLUMNS)            # starter creates one dummy, reindex adds the others as 0
    assert StarterCompatible.to_raw(enc)["home_ownership"].iloc[0] == "own"


def test_all_zero_dummies_become_unknown():
    enc = starter_encode(_raw(2), STARTER_COLUMNS)
    enc[[c for c in enc.columns if c.startswith("home_ownership_")]] = 0
    assert (StarterCompatible.to_raw(enc)["home_ownership"] == "unknown").all()


def test_missing_numeric_column_is_an_error():
    with pytest.raises(ValueError):
        StarterCompatible.to_raw(_raw().drop(columns="age"))


def test_version_mismatch_detection():
    from importlib.metadata import version
    same = {"scikit-learn": version("scikit-learn")}
    assert version_mismatches(same) == {}
    assert "scikit-learn" in version_mismatches({"scikit-learn": "0.0.1"})


def test_fixture_is_the_unmodified_instructor_file():
    src = STARTER.read_text(encoding="utf-8")                          # explicit: Windows defaults to cp1252
    assert 'M = joblib.load("loan_model.joblib")' in src and 'return M["model"], M["columns"]' in src
    assert 'pd.get_dummies(df, columns=["home_ownership"]).reindex(columns=COLUMNS, fill_value=0)' in src


# ------------------------------------------------------------------ integration: real artifacts, isolated process
art_only = pytest.mark.skipif(not HAS_ART, reason="artifacts not built yet")

RUNNER = r'''
import json, sys
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=180)
at.run()
out = {"exception": [str(e.value) for e in at.exception],
       "errors": [e.value for e in at.error], "success": [s.value for s in at.success],
       "metrics": {m.label: m.value for m in at.metric},
       "project_code_imported": any(k == "quickloan" or k.startswith("quickloan.") for k in sys.modules)}
print("JSON" + json.dumps(out))
'''


def run_starter(tmp_path, artifact):
    for src, dst in [(STARTER, "app.py"), (artifact, "loan_model.joblib"), (C.ROOT / "loans.csv", "loans.csv")]:
        shutil.copy(src, tmp_path / dst)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}                    # same decoding on Windows and Linux
    r = subprocess.run([sys.executable, "-I", "-c", RUNNER], cwd=tmp_path, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    line = next(l for l in r.stdout.splitlines() if l.startswith("JSON"))
    return json.loads(line[4:])


@art_only
@pytest.mark.parametrize("artifact", ["deployed", "champion"])
def test_instructor_starter_app_runs_with_our_artifact(tmp_path, artifact):
    path = C.MODEL_ARTIFACT if artifact == "deployed" else C.CHAMPION_ARTIFACT
    out = run_starter(tmp_path, path)
    assert out["exception"] == [], out["exception"]
    assert not out["project_code_imported"], "artifact needed this project's code to load"
    assert any("APPROVE" in s for s in out["success"])           # default applicant (35, AED 90k income, ...)

    import joblib
    name = joblib.load(path)["model_name"]
    lb = pd.read_csv(C.LEADERBOARD_CSV).set_index("model")       # official money_saved at 0.50 / 20 %
    money = next(v for k, v in out["metrics"].items() if "Money saved" in k)
    caught = next(v for k, v in out["metrics"].items() if "Defaulters caught" in k)
    wrong = next(v for k, v in out["metrics"].items() if "Good borrowers rejected" in k)
    assert money == f"AED {lb.loc[name, 'money_saved']:,.0f}"
    assert caught == f"{int(lb.loc[name, 'tp'])}/{int(lb.loc[name, 'tp'] + lb.loc[name, 'fn'])}"
    assert wrong == f"{int(lb.loc[name, 'fp'])}"


@art_only
@pytest.mark.parametrize("artifact", ["deployed", "champion"])
def test_starter_path_predictions_equal_raw_path(artifact):
    import joblib
    art = joblib.load(C.MODEL_ARTIFACT if artifact == "deployed" else C.CHAMPION_ARTIFACT)
    df = pd.read_csv(C.ROOT / "loans.csv").drop(columns="default")
    p_starter = art["model"].predict_proba(starter_encode(df, art["columns"]))[:, 1]
    p_raw = art["pipeline"].predict_proba(df[art["raw_features"]])[:, 1]
    assert np.array_equal(p_starter, p_raw)
    assert np.array_equal(art["model"].predict(starter_encode(df, art["columns"])), art["pipeline"].predict(df[art["raw_features"]]))
