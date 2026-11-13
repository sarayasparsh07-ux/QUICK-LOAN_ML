"""Executive Overview model-selection cards and the accuracy-vs-recall label layout.

Card 1 must always show the 0.50 benchmark champion (even while the dashboard runs the business policy), card 2
the deployed model at its training-CV business threshold. Every expected number is recomputed here from the raw
scores with the official `money_saved` function, so nothing on the cards can be hard-coded.
"""
import itertools
import os
import re
import html as H

import joblib
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from quickloan import config as C, finance as F
from ui import charts as CH
from ui.components import aed, pct

pytestmark = pytest.mark.skipif(not C.MODEL_ARTIFACT.exists() or not C.EVAL_ARTIFACT.exists(),
                                reason="artifacts not built yet")
HARNESS = str(C.ROOT / "tests" / "page_harness.py")
QUALIFIER = ("Selected using the common holdout benchmark for assignment comparison. Its reported result is a "
             "model-ranking outcome, not an independent estimate of deployment performance.")
NOTE = ("The leaderboard compares all models using a common 0.50 classification threshold. The recommended "
        "deployment model is selected separately using training-only cross-validation and a business-oriented "
        "financial threshold. Both selections use the same financial payoff definition.")


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


def overview(interest: int, policy: str):
    at = AppTest.from_file(HARNESS, default_timeout=300)
    at.session_state["_page"] = "overview"
    at.run()
    at.slider(key="interest_pct").set_value(interest)
    at.radio(key="policy").set_value(policy)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def cards(at) -> list[dict]:
    block = next(m.value for m in at.markdown if 'class="ql-selects"' in m.value)
    out = []
    for c in re.findall(r'<div class="ql-select" .*?<div class="why">.*?</div></div>', block, flags=re.S):
        out.append({
            "eyebrow": H.unescape(re.search(r'class="eb">([^<]+)<', c).group(1)),
            "model": H.unescape(re.search(r'class="model">([^<]+)<', c).group(1)),
            "cells": {H.unescape(k): (H.unescape(v), H.unescape(s)) for k, v, s in re.findall(
                r'class="k">([^<]+)</div><div class="v ql-num">([^<]+)</div><div class="s">([^<]*)<', c)},
            "why": H.unescape(re.search(r'class="why">([^<]+)<', c).group(1)),
            "qual": H.unescape(q.group(1)) if (q := re.search(r'class="qual">([^<]+)<', c)) else None,
        })
    note = H.unescape(re.search(r'class="ql-select-note">([^<]+)<', block).group(1))
    return out, note


def benchmark_champion(ev, i):
    y, loan, best = ev["y_test"], ev["loan_test"], None
    for name in C.MODEL_ORDER:
        p = (ev["models"][name]["scores"] >= 0.5).astype(int)
        key = (F.money_saved(y, p, loan, i), ((p == 1) & (y == 1)).sum(), [-ord(c) for c in name])
        if best is None or key > best[0]:
            best = (key, name, p)
    return best[1], best[2]


@pytest.mark.parametrize("policy", ["benchmark", "business"])
@pytest.mark.parametrize("interest", [10, 20, 35])
def test_cards_show_both_selections_with_live_values(ev, interest, policy):
    i = interest / 100
    (c1, c2), note = cards(overview(interest, policy))
    y, loan = ev["y_test"], ev["loan_test"]

    # Card 1 - always the 0.50 benchmark champion, whatever policy the rest of the dashboard runs
    name, p = benchmark_champion(ev, i)
    assert c1["eyebrow"] == "Holdout leaderboard champion" and c1["model"] == name
    assert c1["cells"]["Threshold"][0] == "0.50"
    assert c1["cells"]["Money saved"][0] == aed(F.money_saved(y, p, loan, i), compact=True)
    assert c1["cells"]["Accuracy"][0] == pct((p == y).mean(), 2)
    assert c1["cells"]["Default recall"][0] == pct(((p == 1) & (y == 1)).sum() / y.sum())
    assert c1["why"] == "Highest financial result among models evaluated using the common benchmark threshold."
    assert c1["qual"] == QUALIFIER                                        # holdout-selection caveat, champion only
    assert c2["qual"] is None

    # Card 2 - deployed model, threshold re-searched on training OOF scores at this interest
    dep = ev["deployed_model"]
    t, cv_money = F.best_threshold(ev["y_train"], ev["oof"][dep], ev["loan_train"], i)
    p2 = (ev["models"][dep]["scores"] >= t).astype(int)
    assert c2["eyebrow"] == "Recommended deployment model" and c2["model"] == dep
    assert c2["cells"]["Deployment threshold"][0] == f"{t:.2f}"
    assert c2["cells"]["Training cross-validation"][0] == aed(cv_money, compact=True)
    assert c2["cells"]["Final holdout"][0] == aed(F.money_saved(y, p2, loan, i), compact=True)
    nest = ev["nested_cv"]["summary"]
    assert c2["cells"]["Nested CV (whole procedure)"][0] == aed(nest["procedure_mean_money_per_1000"], compact=True)
    assert c2["cells"]["Nested CV (whole procedure)"][1].startswith("per 1,000 loans at 20%")
    assert f"picked in {nest['selection_counts'].get(dep, 0)} of 5 outer folds" in c2["cells"]["Nested CV (whole procedure)"][1]
    assert note == NOTE


def page(name: str, policy: str = "benchmark"):
    at = AppTest.from_file(HARNESS, default_timeout=300)
    at.session_state["_page"] = name
    at.run()
    at.radio(key="policy").set_value(policy)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return " ".join(H.unescape(m.value) for m in at.markdown)


@pytest.mark.parametrize("policy", ["benchmark", "business"])
def test_champion_wording_is_consistent_across_pages(policy):
    """Leaderboard, Financial Performance and Methodology use the same qualification as the overview card."""
    assert QUALIFIER in page("leaderboard", policy)
    fin = page("financial", policy)
    assert "a model-ranking outcome, not an independent estimate of deployment performance" in fin
    assert "a model-ranking outcome, not an independent estimate of deployment performance" in page("methodology", policy)


def test_card_values_follow_the_interest_slider(ev):
    (a1, a2), _ = cards(overview(10, "business"))
    (b1, b2), _ = cards(overview(40, "business"))
    assert a1["cells"]["Money saved"][0] != b1["cells"]["Money saved"][0]
    assert a2["cells"]["Final holdout"][0] != b2["cells"]["Final holdout"][0]


# ------------------------------------------------------------------ scatter label layout
@pytest.mark.parametrize("policy", ["benchmark", "business"])
@pytest.mark.parametrize("interest", [0.05, 0.20, 0.35, 0.50])
def test_scatter_labels_never_overlap(ev, policy, interest):
    """At the layout width and 25 % / 50 % wider, no label overlaps another label or any other bubble, and every
    point keeps its true (accuracy, recall) coordinates."""
    rows = []
    for name in C.MODEL_ORDER:
        s = ev["models"][name]["scores"]
        t = 0.5 if policy == "benchmark" else F.threshold_at(ev["threshold_curves"][name], interest)
        p = (s >= t).astype(int)
        y = ev["y_test"]
        rows.append((name, (p == y).mean(), ((p == 1) & (y == 1)).sum() / y.sum(),
                     F.money_saved(y, p, ev["loan_test"], interest)))
    rows.sort(key=lambda r: -r[3])
    names_raw = [r[0] for r in rows]
    x, yv, m = (np.array([r[k] for r in rows]) for k in (1, 2, 3))
    size = 12 + 24 * np.clip(m, 0, None) / max(m.max(), 1)
    xr = [x.min() - 0.009, x.max() + 0.009]
    yr = [max(yv.min() - 0.07, 0), yv.max() + 0.09]
    names = [("★ " if k == 0 else "") + n for k, n in enumerate(names_raw)]
    off = CH._place_labels(x, yv, size / 2, names, xr, yr, *CH.SCATTER_PLOT_PX)
    W, Hp = CH.SCATTER_PLOT_PX
    for width in (W, W * 1.25, W * 1.5):
        _, _, bub, lab = CH.label_boxes(x, yv, size / 2, names, off, xr, yr, width, Hp)
        for a, b in itertools.combinations(lab, 2):
            assert CH._overlap(a, b) == 0
        for k, lb in enumerate(lab):
            assert all(CH._overlap(lb, bb) == 0 for j, bb in enumerate(bub) if j != k)


def test_scatter_keeps_every_point_and_hover(ev):
    import pandas as pd
    df = pd.DataFrame({"model": C.MODEL_ORDER, "family": [C.FAMILY[n] for n in C.MODEL_ORDER],
                       "accuracy": np.linspace(0.88, 0.90, 10), "recall": np.linspace(0.1, 0.5, 10),
                       "money_saved": np.linspace(-1e5, 4e6, 10)})
    CH.set_family_lookup(C.FAMILY)
    fig = CH.scatter_acc_recall(df, "Stacking")
    xs = np.concatenate([np.asarray(t.x) for t in fig.data])
    assert sorted(xs) == sorted(df["accuracy"]) and len(fig.layout.annotations) == 10
    assert all("Money saved" in t.hovertemplate for t in fig.data)
    assert {t.name for t in fig.data} == set(C.FAMILY.values())
