"""Design system: colour tokens, the global CSS layer and the Plotly chart theme.

Styling hooks used: `.stApp`, `[data-testid=...]` attributes and the documented `st-key-<key>` class
that Streamlit adds to keyed containers. Everything else is our own HTML.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from quickloan.config import FAMILY_COLOR, PALETTE as P

SPACE = {1: 4, 2: 8, 3: 12, 4: 16, 5: 24, 6: 32, 7: 40}

CSS = f"""
<style>
:root {{
  --bg0:{P['bg0']}; --bg1:{P['bg1']}; --bg2:{P['bg2']}; --bg3:{P['bg3']};
  --teal:{P['teal']}; --emerald:{P['emerald']}; --cyan:{P['cyan']}; --blue:{P['blue']};
  --violet:{P['violet']}; --amber:{P['amber']}; --pos:{P['pos']}; --neg:{P['neg']}; --warn:{P['warn']};
  --neutral:{P['neutral']}; --t1:{P['t1']}; --t2:{P['t2']}; --t3:{P['t3']};
  --glass:rgba(20,35,56,0.72); --glass-strong:rgba(22,38,62,0.86);
  --line:rgba(160,185,220,0.12); --line-strong:rgba(160,185,220,0.22);
  --r:16px; --r-sm:10px;
}}

/* ---------- Layered background: base gradient, ambient light, technical grid ---------- */
.stApp {{
  background-color: var(--bg0);
  background-image:
    radial-gradient(1100px 620px at 8% -6%, rgba(41,211,194,0.16), transparent 62%),
    radial-gradient(900px 560px at 104% 4%, rgba(99,140,255,0.13), transparent 60%),
    radial-gradient(1000px 700px at 62% 112%, rgba(167,139,250,0.09), transparent 62%),
    linear-gradient(rgba(160,185,220,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(160,185,220,0.035) 1px, transparent 1px),
    linear-gradient(rgba(160,185,220,0.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(160,185,220,0.018) 1px, transparent 1px),
    linear-gradient(165deg, #0B1628 0%, #080F1C 45%, #0A1222 100%);
  background-size: auto, auto, auto, 96px 96px, 96px 96px, 24px 24px, 24px 24px, auto;
  background-attachment: fixed;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"], .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1360px; }}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(8,15,28,0.97), rgba(11,20,36,0.97)) !important;
  border-right: 1px solid var(--line);
}}
[data-testid="stSidebarNav"] {{ padding-top: 4px; }}
[data-testid="stSidebarNavLink"] {{ border-radius: 10px; margin: 2px 6px; padding: 7px 10px; transition: background .18s ease; position: relative; }}
[data-testid="stSidebarNavLink"]:hover {{ background: rgba(41,211,194,0.07); }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: rgba(41,211,194,0.13); }}
[data-testid="stSidebarNavLink"][aria-current="page"]::before {{
  content:""; position:absolute; left:-6px; top:9px; bottom:9px; width:3px; border-radius:3px; background: var(--teal);
}}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: var(--t1) !important; font-weight: 600; }}
[data-testid="stSidebarNavSeparator"] {{ border-color: var(--line); }}

/* ---------- Glass panels (keyed containers) ---------- */
div[class*="st-key-panel_"], div[class*="st-key-glow_"] {{
  background: var(--glass);
  border: 1px solid var(--line);
  border-radius: var(--r);
  padding: 18px 20px 14px;
  box-shadow: 0 10px 30px rgba(2,6,16,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
}}
div[class*="st-key-glow_"] {{
  background:
    radial-gradient(600px 260px at 12% 0%, rgba(41,211,194,0.10), transparent 70%),
    var(--glass-strong);
  border-color: rgba(41,211,194,0.20);
}}

/* ---------- Typography helpers ---------- */
.ql-num {{ font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1; }}
.ql-eyebrow {{ font-size: 11.5px; letter-spacing: .16em; text-transform: uppercase; color: var(--teal); font-weight: 600; }}
.ql-section {{ margin: 30px 0 14px; }}
.ql-section h2 {{ font-size: 22px; font-weight: 700; color: var(--t1); margin: 4px 0 4px; letter-spacing: -0.01em; padding: 0; }}
.ql-section p {{ font-size: 13.5px; color: var(--t2); margin: 0; max-width: 900px; line-height: 1.55; }}
.ql-chart-h {{ margin: 0 0 2px; font-size: 16px; font-weight: 650; color: var(--t1); }}
.ql-chart-s {{ margin: 0 0 6px; font-size: 12.5px; color: var(--t3); line-height: 1.45; }}
.ql-muted {{ color: var(--t3); font-size: 12px; line-height: 1.5; }}

/* ---------- Hero ---------- */
.ql-hero {{
  position: relative; overflow: hidden; border-radius: 20px; padding: 30px 32px 26px;
  border: 1px solid rgba(41,211,194,0.18);
  background:
    radial-gradient(520px 240px at 0% 0%, rgba(41,211,194,0.20), transparent 70%),
    radial-gradient(520px 260px at 100% 100%, rgba(99,140,255,0.16), transparent 70%),
    linear-gradient(135deg, rgba(22,36,58,0.92), rgba(11,20,36,0.92));
  box-shadow: 0 20px 50px rgba(2,6,16,0.45);
}}
.ql-hero::after {{
  content:""; position:absolute; inset:0; pointer-events:none; opacity:.5;
  background-image: linear-gradient(115deg, transparent 0 62%, rgba(100,214,245,0.07) 62% 62.4%, transparent 62.4% 70%, rgba(100,214,245,0.05) 70% 70.3%, transparent 70.3%);
}}
.ql-hero h1 {{ font-size: 34px; font-weight: 800; letter-spacing: .01em; color: var(--t1); margin: 8px 0 6px; padding:0; line-height:1.15; }}
.ql-hero h1 .bar {{ color: var(--teal); font-weight: 400; margin: 0 6px; }}
.ql-hero .sub {{ font-size: 15px; color: var(--t2); margin-bottom: 18px; }}
.ql-chips {{ display:flex; flex-wrap:wrap; gap:8px; position:relative; z-index:1; }}
.ql-chip {{ display:inline-flex; align-items:center; gap:7px; font-size:12.5px; color: var(--t2);
  background: rgba(8,15,28,0.55); border:1px solid var(--line); border-radius: 999px; padding: 5px 12px; }}
.ql-chip b {{ color: var(--t1); font-weight: 600; }}
.ql-dot {{ width:7px; height:7px; border-radius:50%; background: var(--emerald); box-shadow: 0 0 0 3px rgba(53,214,160,0.18); }}

/* ---------- KPI cards ---------- */
.ql-kpis {{ display:grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap: 14px; margin-top: 18px; position: relative; }}
.ql-kpis::before {{ content:""; position:absolute; inset:-30px -10px; z-index:0; pointer-events:none;
  background: radial-gradient(700px 160px at 50% 50%, rgba(41,211,194,0.07), transparent 70%); }}
.ql-kpi {{ position:relative; z-index:1; background: var(--glass); border:1px solid var(--line); border-radius: var(--r);
  padding: 16px 16px 14px; transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; min-width:0; }}
.ql-kpi:hover {{ transform: translateY(-2px); border-color: var(--line-strong); box-shadow: 0 12px 26px rgba(2,6,16,0.4); }}
.ql-kpi .top {{ display:flex; align-items:center; justify-content:space-between; gap:8px; }}
.ql-kpi .k {{ font-size: 12.5px; color: var(--t2); font-weight: 550; line-height:1.3; }}
.ql-kpi .ic {{ width:26px; height:26px; flex:0 0 26px; border-radius:8px; display:grid; place-items:center; font-size:13px; }}
.ql-kpi .v {{ font-size: 28px; font-weight: 750; margin: 10px 0 4px; letter-spacing:-0.01em; white-space:nowrap; line-height:1.15; min-height:34px; }}
.ql-kpi .v .u {{ font-size: 13px; font-weight: 650; color: var(--t3); margin-right: 5px; letter-spacing: .04em; vertical-align: 3px; }}
.ql-kpi .v.txt {{ font-size: 21px; white-space: normal; line-height: 1.2; }}
.ql-kpi .s {{ font-size: 11.5px; color: var(--t3); line-height:1.45; }}
.ql-kpi .accent {{ position:absolute; left:16px; right:16px; bottom:0; height:2px; border-radius:2px; opacity:.85; }}

/* ---------- Metric tiles (financial section) ---------- */
.ql-tiles {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:14px; }}
.ql-tile {{ background: var(--glass); border:1px solid var(--line); border-radius: var(--r); padding: 16px 18px; border-top: 3px solid var(--c); }}
.ql-tile .k {{ font-size:12.5px; color: var(--t2); font-weight:550; }}
.ql-tile .v {{ font-size: 25px; font-weight: 750; margin: 8px 0 2px; color: var(--c); white-space:nowrap; }}
.ql-tile .s {{ font-size: 12px; color: var(--t3); line-height:1.45; }}

/* ---------- Tags / status ---------- */
.ql-tag {{ display:inline-block; font-size:11px; font-weight:600; letter-spacing:.02em; padding:2px 9px; border-radius:999px;
  border:1px solid currentColor; background: rgba(8,15,28,0.4); white-space:nowrap; }}
.ql-badge {{ display:inline-flex; align-items:center; gap:4px; font-size:10px; font-weight:700; letter-spacing:.05em;
  padding:2px 6px; border-radius:6px; margin-left:6px; vertical-align: middle; white-space:nowrap; }}
.ql-badge.champ {{ color:#06231B; background: var(--emerald); }}
.ql-badge.sel {{ color: var(--cyan); border:1px solid var(--cyan); }}
.ql-badge.dep {{ color: var(--amber); border:1px solid var(--amber); }}

/* ---------- Leaderboard table ---------- */
.ql-table-wrap {{ overflow-x:auto; border-radius: 12px; }}
table.ql-table {{ width:100%; border-collapse: separate; border-spacing:0; font-size:13px; min-width: 960px; }}
.ql-table th:last-child, .ql-table td:last-child {{ padding-right: 14px; }}
.ql-table th {{ text-align:right; color: var(--t3); font-weight:600; font-size:11px; letter-spacing:.05em; text-transform:uppercase;
  padding: 10px 8px; vertical-align: bottom; line-height: 1.3; border-bottom: 1px solid var(--line-strong); white-space:nowrap; }}
.ql-table th:nth-child(-n+3), .ql-table td:nth-child(-n+3) {{ text-align:left; }}
.ql-table td {{ padding: 11px 8px; border-bottom: 1px solid var(--line); text-align:right; color: var(--t1); white-space:nowrap; font-variant-numeric: tabular-nums; }}
.ql-table tr:hover td {{ background: rgba(100,214,245,0.04); }}
.ql-table tr.champ td {{ background: rgba(53,214,160,0.08); }}
.ql-table tr.sel td:first-child {{ box-shadow: inset 3px 0 0 var(--cyan); }}
.ql-table td.rank {{ color: var(--t3); width: 40px; }}
.ql-table td.money {{ font-weight: 700; }}
.ql-table td.ci {{ color: var(--t2); font-size: 12.5px; }}
.ql-table tr.baseline td {{ border-top: 1px dashed var(--line-strong); color: var(--t2); background: rgba(139,155,178,0.05); }}
.ql-tie {{ color: var(--amber); font-weight: 700; margin-right: 4px; }}
.ql-badge.base {{ color: var(--neutral); border:1px solid var(--neutral); }}
.ql-bar {{ display:inline-block; height:6px; border-radius:3px; vertical-align:middle; margin-left:8px; opacity:.85; }}

/* ---------- Model-selection cards ---------- */
.ql-selects {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:14px; margin-top: 4px; }}
.ql-select {{ position:relative; background: var(--glass); border:1px solid var(--line); border-radius: var(--r);
  border-top: 3px solid var(--c); padding: 16px 18px 14px; box-shadow: 0 10px 26px rgba(2,6,16,0.32);
  transition: border-color .18s ease, transform .18s ease; display:flex; flex-direction:column; }}
.ql-select:hover {{ transform: translateY(-1px); border-color: var(--line-strong); border-top-color: var(--c); }}
.ql-select .head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; }}
.ql-select .eb {{ font-size: 11.5px; letter-spacing:.14em; text-transform: uppercase; color: var(--c); font-weight: 700; }}
.ql-select .tag {{ font-size: 11px; font-weight: 700; color: var(--c); border:1px solid var(--c); border-radius: 999px;
  padding: 2px 9px; white-space: nowrap; background: rgba(8,15,28,0.45); }}
.ql-select .model {{ font-size: 22px; font-weight: 750; color: var(--t1); margin: 6px 0 10px; letter-spacing:-0.01em; }}
.ql-select .grid {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px 16px; }}
.ql-select .cell .k {{ font-size: 11.5px; color: var(--t3); font-weight: 600; }}
.ql-select .cell .v {{ font-size: 16px; color: var(--t1); font-weight: 700; margin-top: 2px; white-space: nowrap; }}
.ql-select .cell .s {{ font-size: 11.5px; color: var(--t3); line-height: 1.4; }}
.ql-select .why {{ margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--line); font-size: 12.5px;
  color: var(--t2); line-height: 1.5; }}
.ql-select .qual {{ margin-top: auto; padding: 7px 0 0 10px; border-left: 2px solid var(--c); font-size: 11.5px;
  color: var(--t3); line-height: 1.45; font-style: italic; }}
.ql-select .why + .qual {{ margin-top: 10px; }}
.ql-select-note {{ margin-top: 10px; font-size: 12.5px; color: var(--t3); line-height: 1.55; max-width: 1100px; }}

/* ---------- Callouts ---------- */
.ql-callout {{ border-radius: 14px; padding: 16px 18px; border:1px solid var(--line); background: rgba(16,29,49,0.78);
  border-left: 3px solid var(--c, var(--teal)); color: var(--t2); font-size: 14px; line-height: 1.6; }}
.ql-callout b {{ color: var(--t1); }}
.ql-callout h4 {{ margin: 0 0 6px; font-size: 15px; color: var(--t1); padding:0; }}

/* ---------- Decision panel ---------- */
.ql-decision {{ border-radius: 18px; padding: 22px 22px 18px; border:1px solid var(--c); position:relative; overflow:hidden;
  background: radial-gradient(420px 200px at 0% 0%, var(--cg), transparent 70%), var(--glass-strong); }}
.ql-decision .verdict {{ font-size: 32px; font-weight: 800; color: var(--c); letter-spacing:.04em; margin: 4px 0 2px; }}
.ql-decision .meta {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:10px 18px; margin-top:14px; }}
.ql-decision .meta div {{ font-size: 12px; color: var(--t3); }}
.ql-decision .meta b {{ display:block; font-size: 15px; color: var(--t1); font-weight: 650; margin-top: 2px; }}
.ql-group-h {{ display:flex; align-items:center; gap:8px; font-size: 13px; font-weight: 650; color: var(--t1); margin: 0 0 6px; }}
.ql-group-h span.n {{ width:22px; height:22px; border-radius:7px; display:grid; place-items:center; font-size:11.5px;
  background: rgba(41,211,194,0.14); color: var(--teal); }}

/* ---------- Brand (sidebar) ---------- */
.ql-brand {{ display:flex; align-items:center; gap:10px; padding: 2px 4px 10px; }}
.ql-brand .logo {{ width:34px; height:34px; border-radius:10px; display:grid; place-items:center;
  background: linear-gradient(135deg, var(--teal), var(--blue)); color:#04121A; font-weight:800; font-size:16px; }}
.ql-brand .name {{ font-size: 15px; font-weight: 750; color: var(--t1); letter-spacing:.04em; }}
.ql-brand .tag {{ font-size: 10.5px; color: var(--t3); letter-spacing:.12em; text-transform:uppercase; }}
.ql-side-h {{ font-size: 10.5px; letter-spacing:.14em; text-transform:uppercase; color: var(--t3); margin: 14px 0 4px; font-weight:600; }}

/* ---------- Widgets polish ---------- */
.stButton > button, .stFormSubmitButton > button {{ transition: transform .12s ease, box-shadow .18s ease; border-radius: 10px; font-weight: 600; }}
.stFormSubmitButton > button:hover {{ box-shadow: 0 6px 18px rgba(41,211,194,0.25); }}
.stFormSubmitButton > button:active {{ transform: translateY(1px); }}
[data-testid="stForm"] {{ border: none; padding: 0; }}
[data-testid="stExpander"] details {{ background: rgba(16,29,49,0.6); border-radius: 12px; }}
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}

/* ---------- Responsive ---------- */
@media (max-width: 1280px) {{ .ql-kpis {{ grid-template-columns: repeat(3, minmax(0,1fr)); }} }}
@media (max-width: 900px) {{
  .ql-tiles {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
  .ql-selects {{ grid-template-columns: 1fr; }}
  .ql-hero h1 {{ font-size: 26px; }}
}}
@media (max-width: 640px) {{
  .ql-kpis, .ql-tiles {{ grid-template-columns: 1fr; }}
  .ql-hero {{ padding: 22px 18px; }}
  .ql-decision .meta {{ grid-template-columns: 1fr; }}
}}
</style>
"""

# ---------------------------------------------------------------- Plotly theme
FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"
GRID = "rgba(170,184,205,0.09)"
AXIS = "rgba(170,184,205,0.22)"

pio.templates["quickloan"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color=P["t2"], size=12.5),
    colorway=[P["teal"], P["blue"], P["violet"], P["amber"], P["cyan"], P["emerald"], P["neg"], P["neutral"]],
    xaxis=dict(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickcolor=AXIS, ticks="outside",
               title=dict(font=dict(size=12.5, color=P["t2"]), standoff=10), automargin=True),
    yaxis=dict(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickcolor=AXIS,
               title=dict(font=dict(size=12.5, color=P["t2"]), standoff=10), automargin=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=P["t2"], size=12), orientation="h",
                yanchor="bottom", y=1.02, xanchor="left", x=0),
    hoverlabel=dict(bgcolor=P["bg3"], bordercolor="rgba(160,185,220,0.25)",
                    font=dict(family=FONT, color=P["t1"], size=12.5)),
    margin=dict(l=10, r=20, t=30, b=10),
))
pio.templates.default = "quickloan"

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

__all__ = ["CSS", "P", "FAMILY_COLOR", "PLOTLY_CONFIG", "SPACE"]
