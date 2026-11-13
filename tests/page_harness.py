"""Test harness: runs the production shell, then renders one page chosen via session state.
Used by tests/test_dashboard.py (Streamlit AppTest cannot click st.navigation links built from callables)."""
import streamlit as st

from ui import pages, shell

ctx = shell.build()
if ctx is not None:
    getattr(pages, st.session_state.get("_page", "overview"))(ctx)
