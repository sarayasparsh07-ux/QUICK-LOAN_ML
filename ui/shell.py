"""Everything the app does before a page renders: CSS, logo, artifact loading (with a graceful failure
screen), the sidebar scenario controls and the shared Ctx. Used by app.py and by the automated dashboard
tests, so the tests exercise exactly the production code path."""
from __future__ import annotations

import streamlit as st

from quickloan import config as C
from quickloan.artifacts import ArtifactError, version_mismatches

from . import components as UI
from . import pages
from .state import POLICIES, board, focus_model, load_artifacts
from .theme import CSS, P

PAGES = [  # (function name, title, material icon, url path)
    ("overview", "Executive Overview", ":material/space_dashboard:", "overview"),
    ("leaderboard", "Model Leaderboard", ":material/leaderboard:", "leaderboard"),
    ("financial", "Financial Performance", ":material/payments:", "financial"),
    ("diagnostics", "Model Diagnostics", ":material/monitoring:", "diagnostics"),
    ("simulator", "Loan Decision Simulator", ":material/fact_check:", "simulator"),
    ("methodology", "Methodology and Data", ":material/menu_book:", "methodology"),
]


def build() -> pages.Ctx | None:
    """Render the shared chrome and return the page context, or None (after showing a message) if the
    artifacts are missing or incompatible."""
    UI.html(CSS)
    st.logo(str(C.ROOT / "assets" / "logo.svg"), size="large", icon_image=str(C.ROOT / "assets" / "icon.svg"))
    try:
        art, ev = load_artifacts()
    except ArtifactError as e:
        UI.html('<div class="ql-hero"><div class="ql-eyebrow" style="color:#FF727F">Artifacts unavailable</div>'
                '<h1>QUICKLOAN<span class="bar">|</span>CREDIT RISK INTELLIGENCE</h1>'
                f'<div class="sub">{UI.esc(e)}</div><div class="ql-chips"><span class="ql-chip">Fix: run '
                '<b>&nbsp;python train_models.py&nbsp;</b> in this folder (needs loans.csv), then reload.</span></div></div>')
        return None

    mism = version_mismatches(art["training"]["versions"])
    if mism:
        detail = " · ".join(f"{k} {a} → {b}" for k, (a, b) in mism.items())
        UI.html(f'<div class="ql-callout" style="--c:{P["warn"]};margin-bottom:14px"><b>Library versions differ from '
                f'training</b> ({UI.esc(detail)}). Results may not reproduce exactly. Install the pinned versions with '
                '<code>pip install -r requirements.txt</code>, or re-run <code>python train_models.py</code>.</div>')

    with st.sidebar:
        UI.side_heading("Scenario controls")
        interest_pct = st.slider("Foregone interest (% of principal)", int(C.INTEREST_MIN * 100),
                                 int(C.INTEREST_MAX * 100), int(C.DEFAULT_INTEREST * 100), 1, key="interest_pct",
                                 format="%d%%",
                                 help="What rejecting a good customer costs, as a share of the loan. Not an annual APR.")
        policy = st.radio("Decision threshold", list(POLICIES), format_func=POLICIES.get, key="policy",
                          help="0.50 is the assignment's common benchmark. Business thresholds are each model's "
                               "money-maximising cut-off, re-optimised for the interest above on training out-of-fold "
                               "predictions (never the test set), and reported separately.")
        st.selectbox("Focus model", ["auto"] + C.MODEL_ORDER, key="focus",
                     format_func=lambda m: "★ Champion (auto)" if m == "auto" else
                     (f"{m}  ◆ deployed" if m == art["model_name"] else m),
                     help="Drives the financial breakdown, diagnostics and highlights.")

    interest = interest_pct / 100
    b = board(ev, interest, policy)
    ctx = pages.Ctx(art=art, ev=ev, b=b, focus=focus_model(b), interest=interest, policy=policy)

    with st.sidebar:
        UI.html(f"""<div class="ql-callout" style="--c:{P['emerald']};padding:12px 14px;font-size:12.5px;margin-top:8px">
<div class="ql-eyebrow" style="font-size:10.5px">Live champion</div>
<div class="ql-live-champ" style="font-size:15px;color:{P['t1']};font-weight:700;margin:3px 0">{UI.esc(b.champion)}</div>
<span class="ql-num">{UI.aed(b.row(b.champion)['money_saved'])}</span> saved at {interest_pct}%</div>
<div class="ql-muted" style="margin-top:14px">Holdout: {len(ev['y_test']):,} loans · deployed model:
{UI.esc(art['display_name'])}<br>Educational analytics — not connected to live lending.</div>""")
    return ctx
