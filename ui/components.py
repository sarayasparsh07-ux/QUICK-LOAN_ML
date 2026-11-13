"""Reusable HTML components. Every builder returns a single-line HTML string (no leading indentation,
so Markdown never mistakes it for a code block) and `html()` renders it."""
from __future__ import annotations

import html as _h

import streamlit as st

from quickloan.config import FAMILY_COLOR, PALETTE as P


def html(s: str) -> None:
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def esc(x) -> str:
    return _h.escape(str(x))


# ---------------------------------------------------------------- formatting
def aed(v: float, compact: bool = False, sign: bool = False) -> str:
    neg = v < 0
    a = abs(v)
    if compact and a >= 1e6:
        body = f"{a / 1e6:,.2f}M"
    elif compact and a >= 1e4:
        body = f"{a / 1e3:,.0f}K"
    else:
        body = f"{a:,.0f}"
    s = "−" if neg else ("+" if sign and v > 0 else "")
    return f"{s}AED {body}"


def pct(v: float, d: int = 1) -> str:
    return f"{v * 100:.{d}f}%"


def tone(v: float) -> str:
    return P["pos"] if v > 0 else (P["neg"] if v < 0 else P["neutral"])


# ---------------------------------------------------------------- building blocks
def section(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eb = f'<div class="ql-eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    html(f'<div class="ql-section">{eb}<h2>{esc(title)}</h2>{sub}</div>')


def chart_header(title: str, subtitle: str = "") -> None:
    sub = f'<div class="ql-chart-s">{subtitle}</div>' if subtitle else ""
    html(f'<div class="ql-chart-h">{esc(title)}</div>{sub}')


def family_tag(family: str) -> str:
    c = FAMILY_COLOR.get(family, P["neutral"])
    return f'<span class="ql-tag" style="color:{c}">{esc(family)}</span>'


def badge(kind: str) -> str:
    label = {"champ": "★ CHAMPION", "sel": "◉ FOCUS", "dep": "◆ DEPLOYED"}[kind]
    return f'<span class="ql-badge {kind}">{label}</span>'


def hero(n_loans: int, focus: str, interest: float, policy_label: str) -> None:
    html(f"""
<div class="ql-hero">
  <div class="ql-eyebrow">AI-Powered Financial Analytics</div>
  <h1>QUICKLOAN<span class="bar">|</span>CREDIT RISK INTELLIGENCE</h1>
  <div class="sub">Machine Learning. Financial Decisions. Smarter Lending.</div>
  <div class="ql-chips">
    <span class="ql-chip"><span class="ql-dot"></span>Holdout evaluation · <b class="ql-num">{n_loans:,}</b> loans</span>
    <span class="ql-chip">Focus model · <b>{esc(focus)}</b></span>
    <span class="ql-chip">Foregone interest · <b class="ql-num">{interest * 100:.0f}%</b> of principal</span>
    <span class="ql-chip">Decision threshold · <b>{esc(policy_label)}</b></span>
    <span class="ql-chip">Educational model · not a live lending system</span>
  </div>
</div>""")


def kpi(label: str, value: str, sub: str, color: str, icon: str) -> str:
    # Currency unit rendered small so the figure itself never truncates; long text values wrap smaller.
    body = esc(value)
    for pre in ("−AED ", "+AED ", "AED "):
        if value.startswith(pre):
            body = f'{pre[0] if pre[0] in "−+" else ""}<span class="u">AED</span>{esc(value[len(pre):])}'
            break
    long_text = not any(ch.isdigit() for ch in value) and len(value) > 10
    cls = "v ql-num txt" if long_text else "v ql-num"
    return (f'<div class="ql-kpi"><div class="top"><div class="k">{esc(label)}</div>'
            f'<div class="ic" style="background:{color}22;color:{color}">{icon}</div></div>'
            f'<div class="{cls}" style="color:{color}" title="{esc(value)}">{body}</div>'
            f'<div class="s">{sub}</div><div class="accent" style="background:{color}"></div></div>')


def kpi_row(cards: list[str]) -> None:
    html('<div class="ql-kpis">' + "".join(cards) + "</div>")


def tile(label: str, value: str, sub: str, color: str) -> str:
    return (f'<div class="ql-tile" style="--c:{color}"><div class="k">{esc(label)}</div>'
            f'<div class="v ql-num">{esc(value)}</div><div class="s">{sub}</div></div>')


def tiles(items: list[str]) -> None:
    html('<div class="ql-tiles">' + "".join(items) + "</div>")


def callout(body: str, title: str = "", color: str = P["teal"]) -> None:
    t = f"<h4>{esc(title)}</h4>" if title else ""
    html(f'<div class="ql-callout" style="--c:{color}">{t}{body}</div>')


def group_header(n: int, title: str) -> None:
    html(f'<div class="ql-group-h"><span class="n">{n}</span>{esc(title)}</div>')


def brand() -> None:
    html("""<div class="ql-brand"><div class="logo">Q</div>
<div><div class="name">QUICKLOAN</div><div class="tag">Credit risk intelligence</div></div></div>""")


def side_heading(t: str) -> None:
    html(f'<div class="ql-side-h">{esc(t)}</div>')


def _ci(lo: float, hi: float) -> str:
    """Compact interval, e.g. 3.64–6.33M (units in the column header)."""
    f = lambda v: (f"{v/1e6:.2f}" if abs(v) >= 1e5 else f"{v/1e6:.3f}")
    return f"{'−' if lo < 0 else ''}{f(abs(lo))}–{'−' if hi < 0 else ''}{f(abs(hi))}M"


def leaderboard_table(lb, champion: str, focus: str, deployed: str, baseline: dict | None = None) -> None:
    """lb: DataFrame with model, family, accuracy, recall, precision, f1, fp, money_saved, ci_low, ci_high,
    p_best (already sorted). `baseline` (the untuned shallow tree) is appended unranked below a divider."""
    rows = []
    for i, r in enumerate(lb.itertuples(index=False), 1):
        cls = " ".join(c for c, cond in [("champ", r.model == champion), ("sel", r.model == focus)] if cond)
        badges = (badge("champ") if r.model == champion else "") + \
                 (badge("sel") if r.model == focus and r.model != champion else "") + \
                 (badge("dep") if r.model == deployed else "")
        col = tone(r.money_saved)
        tie = "" if r.model == champion or r.distinguishable else \
            '<span class="ql-tie" title="95% CI of the gap to the champion includes zero">≈</span>'
        rows.append(
            f'<tr class="{cls}"><td class="rank">{i}</td><td><b>{esc(r.model)}</b>{badges}</td>'
            f'<td>{family_tag(r.family)}</td><td>{pct(r.accuracy, 2)}</td><td>{pct(r.recall)}</td>'
            f'<td>{pct(r.precision)}</td><td>{pct(r.f1)}</td><td>{r.fp:,}</td>'
            f'<td class="money" style="color:{col}">{aed(r.money_saved)}</td>'
            f'<td class="ci">{_ci(r.ci_low, r.ci_high)}</td>'
            f'<td>{tie}{r.p_best * 100:.0f}%</td></tr>')
    if baseline:
        b = baseline
        rows.append(
            f'<tr class="baseline"><td class="rank">–</td><td><b>{esc(b["name"])}</b>'
            f'<span class="ql-badge base">BASELINE</span></td><td>{family_tag("Baseline")}</td>'
            f'<td>{pct(b["accuracy"], 2)}</td><td>{pct(b["recall"])}</td><td>{pct(b["precision"])}</td>'
            f'<td>{pct(b["f1"])}</td><td>{b["fp"]:,}</td>'
            f'<td class="money" style="color:{tone(b["money_saved"])}">{aed(b["money_saved"])}</td>'
            f'<td class="ci">not ranked</td><td>–</td></tr>')
    html('<div class="ql-table-wrap"><table class="ql-table"><thead><tr>'
         '<th>#</th><th>Model</th><th>Family</th><th>Accuracy</th><th>Recall</th><th>Precision</th>'
         '<th>F1</th><th>Good<br>rejected</th><th>Money saved</th><th>95% CI<br>(AED)</th><th>P(best)</th></tr></thead><tbody>'
         + "".join(rows) + "</tbody></table></div>")


def decision_panel(approve: bool, model: str, score: float, threshold: float, score_label: str,
                   interpretation: str, policy: str, extra: list[tuple[str, str]] | None = None) -> None:
    c = P["emerald"] if approve else P["neg"]
    cg = "rgba(53,214,160,0.16)" if approve else "rgba(255,114,127,0.16)"
    word = "✓ APPROVE" if approve else "✕ REJECT"
    html(f"""
<div class="ql-decision" style="--c:{c};--cg:{cg}">
  <div class="ql-eyebrow" style="color:{c}">Model recommendation</div>
  <div class="verdict">{word}</div>
  <div class="ql-muted">{esc(interpretation)}</div>
  <div class="meta">
    <div>Model<b>{esc(model)}</b></div>
    <div>{esc(score_label)}<b class="ql-num">{score:.3f}</b></div>
    <div>Decision threshold<b class="ql-num">{threshold:.2f} · {esc(policy)}</b></div>
    <div>Rule<b>Reject if score ≥ threshold</b></div>
    {"".join(f'<div>{esc(k)}<b class="ql-num">{esc(v)}</b></div>' for k, v in (extra or []))}
  </div>
</div>""")


def selection_card(eyebrow: str, model: str, rows: list[tuple[str, str, str]], why: str, color: str,
                   badge_text: str, qualifier: str = "") -> str:
    """Compact model-selection card. rows = [(label, value, sub-text)]; `qualifier` is an optional one-line
    methodological caveat shown at the foot of the card in muted text."""
    cells = "".join(f'<div class="cell"><div class="k">{esc(k)}</div><div class="v ql-num">{esc(v)}</div>'
                    f'<div class="s">{esc(sub)}</div></div>' for k, v, sub in rows)
    qual = f'<div class="qual">{esc(qualifier)}</div>' if qualifier else ""
    return (f'<div class="ql-select" style="--c:{color}"><div class="head"><span class="eb">{esc(eyebrow)}</span>'
            f'<span class="tag">{esc(badge_text)}</span></div><div class="model">{esc(model)}</div>'
            f'<div class="grid">{cells}</div><div class="why">{esc(why)}</div>{qual}</div>')


def selection_cards(left: str, right: str, note: str) -> None:
    html(f'<div class="ql-selects">{left}{right}</div><div class="ql-select-note">{note}</div>')
