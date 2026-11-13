"""Automated dashboard tests (Streamlit AppTest).

Every page renders without exceptions under both threshold policies; the interest slider, threshold
policy, focus model and sort control drive the right numbers; the simulator rejects a risky applicant
with warnings and reason codes; missing and incompatible artifacts show the fix-it screen instead of
crashing. Expected values are recomputed independently with the official `money_saved` function and the
official per-threshold search (`best_threshold`), not with the app's own fast path.
"""
import os
import re

import joblib
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from quickloan import config as C, finance as F
from ui.components import aed

pytestmark = pytest.mark.skipif(not C.MODEL_ARTIFACT.exists() or not C.EVAL_ARTIFACT.exists(),
                                reason="artifacts not built yet")
HARNESS = str(C.ROOT / "tests" / "page_harness.py")
TIMEOUT = 300


@pytest.fixture(scope="module")
def ev():
    return joblib.load(C.EVAL_ARTIFACT)


@pytest.fixture(autouse=True)
def fresh_caches():
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    for k in ("QUICKLOAN_MODEL_PATH", "QUICKLOAN_EVAL_PATH"):
        os.environ.pop(k, None)
    st.cache_resource.clear()
    st.cache_data.clear()


def run(page="overview", **state):
    at = AppTest.from_file(HARNESS, default_timeout=TIMEOUT)
    at.session_state["_page"] = page
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def expected(ev, interest: float, policy: str) -> tuple[str, float]:
    y, loan = ev["y_test"], ev["loan_test"]
    best = None
    for name in C.MODEL_ORDER:
        s = ev["models"][name]["scores"]
        t = 0.5 if policy == "benchmark" else F.best_threshold(ev["y_train"], ev["oof"][name], ev["loan_train"], interest)[0]
        p = (s >= t).astype(int)
        money = F.money_saved(y, p, loan, interest)
        recall = ((p == 1) & (y == 1)).sum()
        key = (money, recall, [-ord(c) for c in name])
        if best is None or key > best[0]:
            best = (key, name, money)
    return best[1], best[2]


def live_champion(at) -> tuple[str, str]:
    card = next(m.value for m in at.markdown if "Live champion" in m.value)
    name = re.search(r'ql-live-champ"[^>]*>([^<]+)<', card).group(1)
    money = re.search(r'ql-num">([^<]+)<', card).group(1)
    return name, money


# ------------------------------------------------------------------ every page renders
MARKERS = {"overview": "CREDIT RISK INTELLIGENCE", "leaderboard": "The Great Model Bake-Off",
           "financial": "Where the money comes from", "diagnostics": "Model diagnostics",
           "simulator": "Loan decision simulator", "methodology": "Methodology and data"}


@pytest.mark.parametrize("policy", ["benchmark", "business"])
@pytest.mark.parametrize("page", list(MARKERS))
def test_every_page_renders(page, policy):
    at = run(page, policy=policy)
    assert MARKERS[page] in text(at)


def test_app_entrypoint_runs():
    at = AppTest.from_file(str(C.ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    assert "CREDIT RISK INTELLIGENCE" in text(at)


# ------------------------------------------------------------------ scenario controls
@pytest.mark.parametrize("policy", ["benchmark", "business"])
@pytest.mark.parametrize("interest", [5, 20, 37, 50])
def test_slider_and_policy_recalculate_champion(ev, interest, policy):
    at = AppTest.from_file(HARNESS, default_timeout=TIMEOUT)
    at.session_state["_page"] = "overview"
    at.run()
    at.slider(key="interest_pct").set_value(interest)
    at.radio(key="policy").set_value(policy)
    at.run()
    assert not at.exception
    name, money = live_champion(at)
    exp_name, exp_money = expected(ev, interest / 100, policy)
    assert name == exp_name
    assert money == aed(exp_money)
    assert f"{interest}%" in text(at)                      # recommendation rewritten for this interest


def test_focus_model_drives_financial_page():
    at = run("financial", focus="Decision Tree")
    assert "Where the money comes from — Decision Tree" in text(at)


def test_sort_by_accuracy(ev):
    at = run("leaderboard", lb_sort="Accuracy")
    table = next(m.value for m in at.markdown if "ql-table" in m.value and "Money saved" in m.value)
    first = re.search(r"<tbody><tr[^>]*><td[^>]*>1</td><td><b>([^<]+)</b>", table).group(1)
    accs = {n: (ev["models"][n]["pred_050"] == ev["y_test"]).mean() for n in C.MODEL_ORDER}
    assert np.isclose(accs[first], max(accs.values()))


def test_leaderboard_shows_ci_and_baseline():
    t = text(run("leaderboard"))
    assert "95% CI" in t and "P(best)" in t and "BASELINE" in t


# ------------------------------------------------------------------ simulator
def _set(at, label, value):
    next(n for n in at.number_input if n.label == label).set_value(value)


def test_simulator_rejects_risky_applicant_with_warnings_and_reasons():
    at = run("simulator")
    _set(at, "Annual income (AED)", 25000)
    _set(at, "Loan amount (AED)", 120000)
    _set(at, "Past defaults", 3)
    _set(at, "Existing loans", 4)
    _set(at, "Age (years)", 70)
    at.button[0].click()
    at.run()
    assert not at.exception
    t = text(at)
    assert "REJECT" in t
    assert "Input warnings" in t and "Age 70 is outside the training range" in t
    assert "Main reasons for rejection" in t
    assert "Estimated probability of default" in t


def test_simulator_approves_default_applicant():
    assert "APPROVE" in text(run("simulator"))


def test_simulator_unsupported_model_explains_why():
    at = run("simulator", sim_model="KNN")
    assert "sampling-based approximations" in text(at)


# ------------------------------------------------------------------ failure screens
def test_missing_artifacts_show_fix_screen(tmp_path):
    os.environ["QUICKLOAN_MODEL_PATH"] = str(tmp_path / "missing.joblib")
    at = AppTest.from_file(HARNESS, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    t = text(at)
    assert "Artifacts unavailable" in t and "python train_models.py" in t


def test_incompatible_artifact_shows_fix_screen(tmp_path):
    bad = tmp_path / "starter_style.joblib"
    joblib.dump({"model": None, "columns": []}, bad)           # the starter app's artifact shape
    os.environ["QUICKLOAN_MODEL_PATH"] = str(bad)
    at = AppTest.from_file(HARNESS, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    assert "missing keys" in text(at)
