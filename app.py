"""QuickLoan | Credit Risk Intelligence — Streamlit dashboard.

    python train_models.py      # once, builds loan_model.joblib + artifacts/evaluation_results.joblib
    streamlit run app.py

The app never retrains. It loads the saved pipelines and holdout predictions, and recomputes every money
figure with the official `money_saved` function whenever the interest assumption or threshold policy changes.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="QuickLoan · Credit Risk Intelligence", page_icon="◆", layout="wide",
                   initial_sidebar_state="auto")

from ui import pages, shell  # noqa: E402

ctx = shell.build()
if ctx is None:          # artifacts missing or incompatible: shell already showed how to fix it
    st.stop()

nav = st.navigation([
    st.Page(lambda fn=fn: getattr(pages, fn)(ctx), title=title, icon=icon, url_path=path, default=(i == 0))
    for i, (fn, title, icon, path) in enumerate(shell.PAGES)
], position="sidebar")
nav.run()
