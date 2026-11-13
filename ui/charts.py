"""Plotly figure builders. Every chart: transparent background, labelled axes with units, no clipped labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from quickloan.config import FAMILY_COLOR, PALETTE as P

from .components import aed

H_PER_ROW = 34


def _aed_axis(v):
    a = abs(v)
    return ("−" if v < 0 else "") + (f"{a/1e6:.1f}M" if a >= 1e6 else f"{a/1e3:.0f}K" if a >= 1e3 else f"{a:.0f}")


def _bar_height(n):
    return max(260, 70 + n * H_PER_ROW)


def _order(df, col, ascending=True):
    return df.sort_values(col, ascending=ascending)


# ---------------------------------------------------------------- leaderboard bars
def accuracy_bars(df: pd.DataFrame, champion: str) -> go.Figure:
    d = _order(df, "accuracy")
    colors = [FAMILY_COLOR[f] for f in d["family"]]
    fig = go.Figure(go.Bar(
        x=d["accuracy"], y=d["model"], orientation="h", marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v*100:.1f}%" for v in d["accuracy"]], textposition="auto", cliponaxis=False,
        insidetextfont=dict(color="#07131D", size=12), outsidetextfont=dict(color=P["t1"], size=12),
        insidetextanchor="end", constraintext="none", hovertemplate="%{y}<br>Accuracy %{x:.2%}<extra></extra>"))
    fig.update_layout(height=_bar_height(len(d)), margin=dict(l=10, r=56, t=10, b=10), bargap=0.32,
                      xaxis=dict(title="Accuracy (%)", range=[0, 1], tickformat=".0%", dtick=0.2),
                      yaxis=dict(title="Model"), showlegend=False)
    return fig


def recall_bars(df: pd.DataFrame, champion: str) -> go.Figure:
    d = _order(df, "recall")
    top = d["recall"].max()
    colors = [P["teal"] if v >= top - 1e-9 else FAMILY_COLOR[f] for v, f in zip(d["recall"], d["family"])]
    fig = go.Figure(go.Bar(
        x=d["recall"], y=d["model"], orientation="h", marker=dict(color=colors),
        text=[f"{v*100:.1f}%  ({tp}/{tp+fn})" for v, tp, fn in zip(d["recall"], d["tp"], d["fn"])],
        textposition="outside", cliponaxis=False, textfont=dict(color=P["t1"], size=12),
        hovertemplate="%{y}<br>Default recall %{x:.2%}<extra></extra>"))
    fig.update_layout(height=_bar_height(len(d)), margin=dict(l=10, r=40, t=10, b=10), bargap=0.32,
                      xaxis=dict(title="Default recall (%) — share of defaulters caught", tickformat=".0%",
                                 range=[0, max(top * 1.38, 0.1)]),
                      yaxis=dict(title="Model"), showlegend=False)
    return fig


def money_bars(df: pd.DataFrame, champion: str, focus: str, height: int | None = None) -> go.Figure:
    d = _order(df, "money_saved")
    v = d["money_saved"].to_numpy()
    colors = [P["emerald"] if x >= 0 else P["neg"] for x in v]
    lines = [P["t1"] if m == champion else (P["cyan"] if m == focus else "rgba(0,0,0,0)") for m in d["model"]]
    widths = [2 if m in (champion, focus) else 0 for m in d["model"]]
    opac = [1.0 if m == champion else 0.62 for m in d["model"]]
    # compact labels fit inside the bar even at phone width; exact AED values are in the hover and the tables
    labels = [("★ " if m == champion else "") + aed(x, compact=True) for m, x in zip(d["model"], v)]
    lo, hi = min(v.min(), 0), max(v.max(), 0)
    span = (hi - lo) or 1
    fig = go.Figure(go.Bar(
        x=v, y=d["model"], orientation="h",
        marker=dict(color=colors, opacity=opac, line=dict(color=lines, width=widths)),
        text=labels, textposition="auto", cliponaxis=False, textfont=dict(size=12),
        insidetextfont=dict(color="#06231B", size=12), outsidetextfont=dict(color=P["t1"], size=12),
        insidetextanchor="end", constraintext="none",
        hovertemplate="%{y}<br>Money saved %{x:,.0f} AED<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=P["t3"], width=1, dash="dot"))
    fig.update_layout(
        height=height or _bar_height(len(d)), margin=dict(l=10, r=30, t=10, b=10), bargap=0.32, showlegend=False,
        xaxis=dict(title="Net money saved vs approving everyone (AED)",
                   range=[lo - span * (0.34 if lo < 0 else 0.02), hi + span * 0.30],
                   tickvals=_nice_ticks(lo, hi), ticktext=[_aed_axis(t) for t in _nice_ticks(lo, hi)]),
        yaxis=dict(title="Model"))
    return fig


def _nice_ticks(lo, hi, n=5):
    span = (hi - lo) or 1
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min([s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw])
    start = np.floor(lo / step) * step
    return [float(x) for x in np.arange(start, hi + step * 0.5, step)]


SCATTER_H = 500
# Plot-area size the labels are laid out for. Deliberately the *narrowest* common desktop layout (sidebar open,
# ~1280 px window): wider screens only push the points further apart, so labels placed here never collide there.
SCATTER_PLOT_PX = (660, 370)
LABEL_FONT = 11.5
LABEL_H = 22                     # 18 px text box + 4 px breathing room


def _label_w(text: str) -> float:
    return 6.6 * len(text) + 10                                         # px at 11.5 px Inter, plus padding


def scatter_acc_recall(df: pd.DataFrame, champion: str) -> go.Figure:
    d = df.copy()
    m = d["money_saved"].to_numpy()
    size = 12 + 24 * np.clip(m, 0, None) / max(m.max(), 1)          # marker diameter in px (bubble size = money)
    xr = [d["accuracy"].min() - 0.009, d["accuracy"].max() + 0.009]
    yr = [max(d["recall"].min() - 0.07, 0), d["recall"].max() + 0.09]
    names = [("★ " if n == champion else "") + n for n in d["model"]]
    offsets = _place_labels(d["accuracy"].to_numpy(), d["recall"].to_numpy(), size / 2, names, xr, yr,
                            *SCATTER_PLOT_PX)
    fig = go.Figure()
    for fam, g in d.groupby("family", sort=False):
        fig.add_trace(go.Scatter(
            x=g["accuracy"], y=g["recall"], mode="markers", name=fam, text=g["model"],
            marker=dict(size=size[g.index], color=FAMILY_COLOR[fam], opacity=0.78,
                        line=dict(color=[P["t1"] if n == champion else "rgba(8,15,28,0.9)" for n in g["model"]],
                                  width=[2.5 if n == champion else 1 for n in g["model"]])),
            customdata=np.stack([g["money_saved"]], axis=-1),
            hovertemplate="<b>%{text}</b> · " + fam + "<br>Accuracy %{x:.2%}<br>Default recall %{y:.2%}"
                          "<br>Money saved %{customdata[0]:,.0f} AED<extra></extra>", cliponaxis=False))
    for i, (ax, ay) in enumerate(offsets):
        fig.add_annotation(x=d["accuracy"].iat[i], y=d["recall"].iat[i], text=names[i], ax=ax, ay=ay,
                           showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="rgba(170,184,205,0.45)",
                           standoff=size[i] / 2 + 1,
                           font=dict(size=LABEL_FONT, color=P["t1"] if names[i][0] == "★" else P["t2"]),
                           bgcolor="rgba(8,15,28,0.80)", borderpad=2, captureevents=False)
    fig.update_layout(height=SCATTER_H, margin=dict(l=10, r=20, t=56, b=10),
                      xaxis=dict(title="Accuracy (%) — note the narrow range", tickformat=".1%", range=xr),
                      yaxis=dict(title="Default recall (%)", tickformat=".0%", range=yr),
                      legend=dict(title=dict(text="Family  ", font=dict(color=P["t3"])), orientation="h",
                                  yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


def _overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _seg_hits_box(p, q, b, pad=1.0) -> bool:
    """Does segment p-q pass through box b? (Liang-Barsky clip)"""
    x0, y0, dx, dy = p[0], p[1], q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-dx, x0 - (b[0] - pad)), (dx, (b[2] + pad) - x0), (-dy, y0 - (b[1] - pad)), (dy, (b[3] + pad) - y0)):
        if pk == 0:
            if qk < 0:
                return False
            continue
        r = qk / pk
        if pk < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False
    return True


def label_boxes(x, y, radii, names, offsets, xr, yr, w_px, h_px):
    """Screen-space boxes of every bubble and label for a given placement (used by the placer and the tests)."""
    sx, sy = w_px / (xr[1] - xr[0]), h_px / (yr[1] - yr[0])
    px, py = (np.asarray(x) - xr[0]) * sx, (yr[1] - np.asarray(y)) * sy
    bubbles = [(px[i] - radii[i], py[i] - radii[i], px[i] + radii[i], py[i] + radii[i]) for i in range(len(px))]
    labels = []
    for i, (ax, ay) in enumerate(offsets):
        w = _label_w(names[i])
        cx, cy = px[i] + ax, py[i] + ay
        labels.append((cx - w / 2, cy - LABEL_H / 2, cx + w / 2, cy + LABEL_H / 2))
    return px, py, bubbles, labels


def _place_labels(x, y, radii, names, xr, yr, w_px, h_px, widths=None):
    """Label placement in pixel space. Each bubble's label is tried on rings of 24 directions at growing distances;
    a spot is accepted only if its box is inside the plot and clear of every bubble, every label already placed
    and every leader line already drawn, and its own leader line crosses no other bubble or label. Because the
    chart is responsive, every candidate is checked at several plot widths (w_px up to ~1.5x) at once. Biggest
    (most valuable) bubbles are labelled first. If no clear spot exists, the spot with the least overlap is used.
    Returns (ax, ay) annotation offsets in px; data coordinates are never moved."""
    widths = widths or (w_px, w_px * 1.25, w_px * 1.5)
    n = len(x)
    sy = h_px / (yr[1] - yr[0])
    py = (yr[1] - np.asarray(y)) * sy
    pxs = [(np.asarray(x) - xr[0]) * (w / (xr[1] - xr[0])) for w in widths]
    bubbles = [[(px[i] - radii[i] - 2, py[i] - radii[i] - 2, px[i] + radii[i] + 2, py[i] + radii[i] + 2)
                for i in range(n)] for px in pxs]
    placed = [[] for _ in widths]
    leaders = [[] for _ in widths]
    out = [(0.0, -30.0)] * n
    angles = np.deg2rad(np.r_[[-90, -60, -120, 0, 180, -30, -150, 30, 150, 60, 120, 90],
                              [-75, -105, -45, -135, -15, -165, 15, 165, 45, 135, 75, 105]])
    for i in np.argsort(-np.asarray(radii), kind="mergesort"):
        w = _label_w(names[i])
        best, best_cost = None, np.inf
        for k, dist in enumerate(radii[i] + np.array([16, 28, 42, 58, 76, 96, 120])):
            for a in angles:
                ca, sa = np.cos(a), np.sin(a)
                ox = dist * ca + np.sign(ca) * w / 2 * (abs(ca) > 0.3)
                oy = dist * sa + np.sign(sa) * LABEL_H / 2 * (abs(sa) > 0.3)
                cost, geo = 0.01 * k, []                                       # prefer short leaders when tied
                for wi, (W, px) in enumerate(zip(widths, pxs)):
                    cx, cy = px[i] + ox, py[i] + oy
                    b = (cx - w / 2, cy - LABEL_H / 2, cx + w / 2, cy + LABEL_H / 2)
                    seg = ((px[i], py[i]), (min(max(px[i], b[0]), b[2]), min(max(py[i], b[1]), b[3])))
                    cost += 4 * (max(0, -b[0]) + max(0, b[2] - W) + max(0, -b[1]) + max(0, b[3] - h_px))
                    cost += sum(_overlap(b, o) for j, o in enumerate(bubbles[wi]) if j != i)
                    cost += 2 * sum(_overlap(b, o) for o in placed[wi])
                    cost += 150 * sum(_seg_hits_box(s0, s1, b) for s0, s1 in leaders[wi])
                    cost += 150 * sum(_seg_hits_box(*seg, o) for j, o in enumerate(bubbles[wi]) if j != i)
                    cost += 150 * sum(_seg_hits_box(*seg, o) for o in placed[wi])
                    geo.append((b, seg))
                if cost < best_cost:
                    best, best_cost = ((ox, oy), geo), cost
            if best_cost < 1:                                                 # clear spot found on this ring
                break
        off, geo = best
        out[i] = off
        for wi, (b, seg) in enumerate(geo):
            placed[wi].append(b)
            leaders[wi].append(seg)
    return out


# ---------------------------------------------------------------- financial
def waterfall(b: dict, net: float) -> go.Figure:
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["relative", "relative", "total"],
        x=["Loans saved<br>(caught defaulters)", "Interest lost<br>(good customers rejected)", "Net money saved"],
        y=[b["loans_saved"], -b["interest_lost"], net],
        text=[aed(b["loans_saved"], sign=True), aed(-b["interest_lost"]), aed(net)],
        textposition="outside", cliponaxis=False, textfont=dict(color=P["t1"], size=13),
        increasing=dict(marker=dict(color=P["emerald"])), decreasing=dict(marker=dict(color=P["amber"])),
        totals=dict(marker=dict(color=P["teal"] if net >= 0 else P["neg"])),
        connector=dict(line=dict(color=P["t3"], width=1, dash="dot")),
        hovertemplate="%{x}<br>%{y:,.0f} AED<extra></extra>"))
    top = max(b["loans_saved"], net, 1)
    lo = min(0, net)
    ticks = _nice_ticks(lo, top * 1.0)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      yaxis=dict(title="AED (vs approving everyone)", range=[lo * 1.25, top * 1.2],
                                 tickvals=ticks, ticktext=[_aed_axis(t) for t in ticks]),
                      xaxis=dict(title="Contribution to the official money-saved metric"))
    return fig


def default_principal_split(b: dict, total_def: float) -> go.Figure:
    caught, missed = b["loans_saved"], b["missed_default_principal"]
    fig = go.Figure()
    for val, name, col, n in [(caught, "Caught (rejected)", P["emerald"], b["n_tp"]),
                              (missed, "Missed (approved) — exposure", P["neg"], b["n_fn"])]:
        share = val / total_def if total_def else 0
        fig.add_trace(go.Bar(
            x=[val], y=[""], orientation="h", name=f"{name} · {aed(val, compact=True)} · {n} loans",
            marker=dict(color=col), text=[f"{share*100:.0f}%" if share >= 0.08 else ""], textposition="inside",
            insidetextanchor="middle", textfont=dict(color="#07131D", size=13),
            hovertemplate=f"{name}<br>%{{x:,.0f}} AED · {n} loans<extra></extra>"))
    ticks = _nice_ticks(0, total_def)
    fig.update_layout(barmode="stack", height=250, margin=dict(l=10, r=10, t=10, b=70), bargap=0.25,
                      xaxis=dict(title="Principal of all holdout defaulters (AED)", range=[0, total_def],
                                 tickvals=ticks, ticktext=[_aed_axis(t) for t in ticks]),
                      yaxis=dict(title="", showticklabels=False),
                      legend=dict(orientation="v", y=-0.62, yanchor="top", x=0, traceorder="normal"))
    return fig


def sensitivity_lines(sens: pd.DataFrame, focus: str, champion: str, interest: float) -> go.Figure:
    fig = go.Figure()
    for name, g in sens.groupby("model", sort=False):
        hi = name in (focus, champion)
        fam_col = FAMILY_COLOR.get(_family_of(name), P["neutral"])
        fig.add_trace(go.Scatter(
            x=g["interest"], y=g["money_saved"], mode="lines", name=name,
            line=dict(width=3.2 if name == focus else (2.4 if hi else 1.2),
                      color=P["cyan"] if name == focus else (P["emerald"] if name == champion else fam_col)),
            opacity=1 if hi else 0.38,
            hovertemplate=f"{name}<br>Interest %{{x:.0%}}<br>%{{y:,.0f}} AED<extra></extra>"))
    fig.add_vline(x=interest, line=dict(color=P["amber"], width=1.4, dash="dash"),
                  annotation_text=f"current {interest*100:.0f}%", annotation_position="top",
                  annotation_font=dict(color=P["amber"], size=12))
    fig.add_hline(y=0, line=dict(color=P["t3"], width=1, dash="dot"))
    lo, hi_ = sens["money_saved"].min(), sens["money_saved"].max()
    ticks = _nice_ticks(min(lo, 0), hi_)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(title="Foregone-interest assumption (% of principal)", tickformat=".0%",
                                 range=[0.05, 0.50], dtick=0.05),
                      yaxis=dict(title="Net money saved (AED)", tickvals=ticks, ticktext=[_aed_axis(t) for t in ticks]),
                      legend=dict(orientation="v", x=1.01, y=1, yanchor="top", xanchor="left", font=dict(size=11)))
    return fig


_FAM = {}


def set_family_lookup(d: dict):
    _FAM.update(d)


def _family_of(name):
    return _FAM.get(name, "Single")


def strategy_bars(items: list[tuple[str, float]]) -> go.Figure:
    """Value labels live in the category axis (name + amount), so no bar text can collide or clip."""
    vals = [i[1] for i in items]
    cols = [P["emerald"] if v > 0 else (P["neg"] if v < 0 else P["neutral"]) for v in vals]
    labels = [f"{n}<br><b><span style='color:{c}'>{aed(v)}</span></b>" for (n, v), c in zip(items, cols)]
    lo, hi = min(min(vals), 0), max(max(vals), 0)
    span = (hi - lo) or 1
    ticks = _nice_ticks(lo, hi)
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker=dict(color=cols),
        hovertemplate="%{x:,.0f} AED<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=P["t3"], width=1, dash="dot"))
    fig.update_layout(height=80 + 62 * len(items), margin=dict(l=10, r=20, t=10, b=10), bargap=0.42,
                      xaxis=dict(title="Net money saved vs approve-all (AED)",
                                 range=[lo - span * 0.06, hi + span * 0.06],
                                 tickvals=ticks, ticktext=[_aed_axis(t) for t in ticks]),
                      yaxis=dict(title="", autorange="reversed", tickfont=dict(size=12)), showlegend=False)
    return fig


# ---------------------------------------------------------------- diagnostics
def confusion(tn, fp, fn, tp) -> go.Figure:
    z = np.array([[tn, fp], [fn, tp]])
    row_tot = z.sum(axis=1, keepdims=True)
    labels = [["TN · correct approval", "FP · good customer rejected"],
              ["FN · defaulter approved", "TP · defaulter caught"]]
    text = [[f"<b>{z[i, j]:,}</b><br>{z[i, j] / row_tot[i, 0] * 100:.1f}% of row<br><span style='font-size:11px'>{labels[i][j]}</span>"
             for j in range(2)] for i in range(2)]
    # colour by outcome type (good = teal, bad = coral), intensity by row share
    share = z / row_tot
    good = np.array([[1, 0], [0, 1]])
    zc = np.where(good == 1, share, -share)
    fig = go.Figure(go.Heatmap(
        z=zc, x=["Predicted APPROVE (0)", "Predicted REJECT (1)"], y=["Actual REPAID (0)", "Actual DEFAULTED (1)"],
        text=text, texttemplate="%{text}", textfont=dict(color=P["t1"], size=13), showscale=False,
        zmin=-1, zmax=1, xgap=4, ygap=4,
        colorscale=[[0, "#7A2B3A"], [0.5, P["bg3"]], [1, "#0F6E66"]],
        hovertemplate="%{y} · %{x}<extra></extra>"))
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="Model decision", side="bottom", showgrid=False, ticks=""),
                      yaxis=dict(title="Actual outcome", autorange="reversed", showgrid=False, ticks=""))
    return fig


def roc_curves(models: dict, focus: str, aucs: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="No skill",
                             line=dict(color=P["t3"], dash="dot", width=1), hoverinfo="skip"))
    for n, r in models.items():
        c = r["curves"]["roc"]
        f = n == focus
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=f"{n} ({aucs[n]:.3f})",
                                 line=dict(width=3 if f else 1.1, color=P["cyan"] if f else FAMILY_COLOR[r["family"]]),
                                 opacity=1 if f else 0.4, hovertemplate=f"{n}<br>FPR %{{x:.2f}} · TPR %{{y:.2f}}<extra></extra>"))
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="False positive rate (good customers rejected)", range=[0, 1], tickformat=".0%"),
                      yaxis=dict(title="True positive rate (defaulters caught)", range=[0, 1.02], tickformat=".0%"),
                      legend=dict(orientation="h", x=0, y=-0.2, yanchor="top", xanchor="left",
                                  font=dict(size=10.5), entrywidth=150, entrywidthmode="pixels"))
    return fig


def pr_curves(models: dict, focus: str, aps: dict, prevalence: float) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=prevalence, line=dict(color=P["t3"], dash="dot", width=1),
                  annotation_text=f"no-skill = holdout default rate {prevalence*100:.2f}%", annotation_position="bottom right",
                  annotation_font=dict(color=P["t3"], size=11))
    for n, r in models.items():
        c = r["curves"]["pr"]
        f = n == focus
        fig.add_trace(go.Scatter(x=c["recall"], y=c["precision"], mode="lines", name=f"{n} ({aps[n]:.3f})",
                                 line=dict(width=3 if f else 1.1, color=P["cyan"] if f else FAMILY_COLOR[r["family"]]),
                                 opacity=1 if f else 0.4, hovertemplate=f"{n}<br>Recall %{{x:.2f}} · Precision %{{y:.2f}}<extra></extra>"))
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="Recall (share of defaulters caught)", range=[0, 1], tickformat=".0%"),
                      yaxis=dict(title="Precision (share of rejections that were defaulters)", range=[0, 1.02], tickformat=".0%"),
                      legend=dict(orientation="h", x=0, y=-0.2, yanchor="top", xanchor="left",
                                  font=dict(size=10.5), entrywidth=150, entrywidthmode="pixels"))
    return fig


def importance_bars(imp: pd.DataFrame) -> go.Figure:
    d = imp.sort_values("importance_mean")
    mean, std = d["importance_mean"].to_numpy(), d["importance_std"].to_numpy()
    hi = (mean + std).max()
    lo = min((mean - std).min(), 0)
    pad = (hi - lo) * 0.03
    fig = go.Figure(go.Bar(
        x=mean, y=d["feature"], orientation="h",
        error_x=dict(type="data", array=std, color=P["t3"], thickness=1.2, width=4),
        marker=dict(color=[P["teal"] if v > 0.005 else P["neutral"] for v in mean]),
        hovertemplate="%{y}<br>ROC-AUC drop %{x:.4f}<extra></extra>"))
    # value labels sit to the right of the error bar (or of zero), never on the bar, whisker or axis labels
    fig.add_trace(go.Scatter(
        x=np.maximum(mean + std, 0) + pad, y=d["feature"], mode="text", text=[f"{v:+.3f}" for v in mean],
        textposition="middle right", textfont=dict(color=P["t1"], size=11.5), hoverinfo="skip", cliponaxis=False))
    fig.update_layout(height=_bar_height(len(d)), margin=dict(l=10, r=20, t=10, b=10), showlegend=False, bargap=0.35,
                      xaxis=dict(title="Drop in validation ROC-AUC when the input is shuffled",
                                 range=[lo - pad * 2, hi + (hi - lo) * 0.22], zeroline=True),
                      yaxis=dict(title="Raw input"))
    return fig


def calibration_plot(models: dict, focus: str) -> go.Figure:
    c = models[focus]["curves"]["calibration"]
    mx = max(max(c["mean_pred"]), max(c["frac_pos"])) * 1.1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect calibration",
                             line=dict(color=P["t3"], dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=c["mean_pred"], y=c["frac_pos"], mode="lines+markers", name=focus,
                             line=dict(color=P["cyan"], width=2.5), marker=dict(size=8),
                             hovertemplate="Mean score %{x:.2f}<br>Observed default rate %{y:.2f}<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(title="Mean model score in bin (deciles)", range=[0, mx], tickformat=".0%"),
                      yaxis=dict(title="Observed default rate in bin", range=[0, mx], tickformat=".0%"))
    return fig


# ---------------------------------------------------------------- methodology / EDA
def default_rate_bars(g: pd.DataFrame, overall: float, title_x: str) -> go.Figure:
    cols = [P["neg"] if v > overall * 1.15 else (P["emerald"] if v < overall * 0.85 else P["amber"]) for v in g["default_rate"]]
    fig = go.Figure(go.Bar(
        x=g["group"].astype(str), y=g["default_rate"], marker=dict(color=cols),
        text=[f"{v*100:.1f}%<br><span style='font-size:10px'>n={n:,}</span>" for v, n in zip(g["default_rate"], g["n"])],
        textposition="outside", cliponaxis=False, textfont=dict(color=P["t1"], size=11.5),
        hovertemplate="%{x}<br>Default rate %{y:.1%}<extra></extra>"))
    fig.add_hline(y=overall, line=dict(color=P["t3"], dash="dot", width=1))      # labelled in the panel subtitle
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      xaxis=dict(title=title_x, type="category"),
                      yaxis=dict(title="Default rate (%)", tickformat=".0%", range=[0, g["default_rate"].max() * 1.35]))
    return fig


def benchmark_compare(ours: pd.DataFrame, ref: dict) -> go.Figure:
    d = ours.sort_values("money_saved")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[ref[m]["money"] for m in d["model"]], y=d["model"], orientation="h",
                         name="Instructor reference", marker=dict(color="rgba(139,155,178,0.45)"),
                         hovertemplate="%{y}<br>Reference %{x:,.0f} AED<extra></extra>"))
    fig.add_trace(go.Bar(x=d["money_saved"], y=d["model"], orientation="h", name="This build",
                         marker=dict(color=P["teal"]), hovertemplate="%{y}<br>This build %{x:,.0f} AED<extra></extra>"))
    hi = max(d["money_saved"].max(), max(r["money"] for r in ref.values()))
    lo = min(d["money_saved"].min(), 0)
    ticks = _nice_ticks(lo, hi)
    fig.update_layout(barmode="group", height=_bar_height(len(d)) + 60, margin=dict(l=10, r=10, t=40, b=10),
                      bargap=0.25, xaxis=dict(title="Money saved at 20 % (AED)", tickvals=ticks,
                                              ticktext=[_aed_axis(t) for t in ticks], range=[lo, hi * 1.05]),
                      yaxis=dict(title="Model"))
    return fig


def corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, xgap=2, ygap=2,
        colorscale=[[0, P["neg"]], [0.5, P["bg2"]], [1, P["teal"]]],
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=10.5, color=P["t1"]),
        colorbar=dict(title=dict(text="r", font=dict(color=P["t2"])), tickfont=dict(color=P["t2"]), thickness=12),
        hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>"))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="", tickangle=-35, showgrid=False), yaxis=dict(title="", autorange="reversed", showgrid=False))
    return fig


# ---------------------------------------------------------------- additions: uncertainty, thresholds, reasons, calibration, fairness
def ci_forest(df: pd.DataFrame, champion: str, focus: str, B: int) -> go.Figure:
    """Point estimate + 95 % paired-bootstrap interval per model; champion's interval shaded across the chart."""
    d = df.sort_values("money_saved")
    lo, hi = float(d["ci_low"].min()), float(d["ci_high"].max())
    span = (hi - lo) or 1
    ch = d[d["model"] == champion].iloc[0]
    fig = go.Figure()
    fig.add_vrect(x0=ch["ci_low"], x1=ch["ci_high"], fillcolor="rgba(53,214,160,0.08)", line_width=0, layer="below")
    for r in d.itertuples(index=False):
        col = P["emerald"] if r.model == champion else (P["cyan"] if r.model == focus else
              (P["neutral"] if not r.distinguishable else P["neg"]))
        fig.add_trace(go.Scatter(x=[r.ci_low, r.ci_high], y=[r.model, r.model], mode="lines",
                                 line=dict(color=col, width=3 if r.model in (champion, focus) else 2),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=[r.money_saved], y=[r.model], mode="markers", showlegend=False,
            marker=dict(size=11 if r.model == champion else 9, color=col, line=dict(color=P["bg1"], width=1.5),
                        symbol="diamond" if r.model == champion else "circle"),
            hovertemplate=(f"<b>{r.model}</b><br>Money saved %{{x:,.0f}} AED<br>95% CI {r.ci_low:,.0f} – {r.ci_high:,.0f}"
                           f"<br>P(best) {r.p_best:.1%}<extra></extra>")))
    # P(best) column on the right, outside the plotting area
    for r in d.itertuples(index=False):
        fig.add_annotation(x=1.01, xref="paper", y=r.model, yref="y", xanchor="left", showarrow=False,
                           text=f"{r.p_best*100:.0f}%", font=dict(color=P["t1"] if r.p_best >= 0.1 else P["t3"], size=12))
    fig.add_annotation(x=1.01, xref="paper", y=1.0, yref="paper", yanchor="bottom", xanchor="left", showarrow=False,
                       text="P(best)", font=dict(color=P["t3"], size=11))
    ticks = _nice_ticks(lo, hi)
    fig.update_layout(height=_bar_height(len(d)), margin=dict(l=10, r=70, t=24, b=10), showlegend=False,
                      xaxis=dict(title=f"Money saved vs approving everyone (AED) · 95% paired bootstrap, {B:,} resamples",
                                 range=[lo - span * 0.04, hi + span * 0.04], tickvals=ticks,
                                 ticktext=[_aed_axis(t) for t in ticks]),
                      yaxis=dict(title="Model"))
    return fig


def threshold_by_interest(curves: dict, focus: str, interest: float, break_even) -> go.Figure:
    grid = np.round(np.arange(0.05, 0.501, 0.01), 2)
    from quickloan import finance as F
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=[break_even(i) for i in grid], mode="lines", name="Break-even i/(1+i) (calibrated scores)",
                             line=dict(color=P["t3"], dash="dot", width=1.4)))
    for n, c in curves.items():
        f = n == focus
        col = P["cyan"] if f else FAMILY_COLOR.get(_family_of(n), P["neutral"])
        fig.add_trace(go.Scatter(x=grid, y=[F.threshold_at(c, i) for i in grid], mode="lines", name=n,
                                 line=dict(width=3.2 if f else 1.6, color=col if f else _fade(col, 0.55), shape="hv"),
                                 hovertemplate=f"{n}<br>interest %{{x:.0%}} → threshold %{{y:.2f}}<extra></extra>"))
    fig.add_vline(x=interest, line=dict(color=P["amber"], width=1.4, dash="dash"))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="Foregone-interest assumption (% of principal)", tickformat=".0%", range=[0.05, 0.50], dtick=0.05),
                      yaxis=dict(title="Money-maximising threshold (training OOF)", range=[0, 0.6], tickformat=".2f"),
                      legend=dict(orientation="v", x=1.01, y=1, xanchor="left", yanchor="top", font=dict(size=10.5)))
    return fig


def _fade(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def explanation_bars(table: pd.DataFrame, units: str, k: int = 8) -> go.Figure:
    """Signed contribution bars. The value is printed inside the category label (right-aligned, coloured), so no
    text sits beside the bars - nothing can collide with the axis labels at any width."""
    d = table.head(k).iloc[::-1]
    cols = [P["neg"] if v > 0 else P["emerald"] for v in d["contribution"]]
    labels = [f"{l} = {v}  <b><span style='color:{c}'>{x:+.3f}</span></b>"
              for l, v, x, c in zip(d["label"], d["value"], d["contribution"], cols)]
    m = float(d["contribution"].abs().max()) or 1
    fig = go.Figure(go.Bar(
        x=d["contribution"], y=labels, orientation="h", marker=dict(color=cols),
        hovertemplate="%{y}<br>%{x:+.4f} " + units + "<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=P["t3"], width=1))
    fig.update_layout(height=60 + 34 * len(d), margin=dict(l=10, r=20, t=10, b=10), showlegend=False, bargap=0.35,
                      xaxis=dict(title=f"Contribution ({units}) · coral raises risk, emerald lowers it",
                                 range=[-m * 1.08, m * 1.08], nticks=5),
                      yaxis=dict(title="", tickfont=dict(size=12)))
    return fig


def calibration_compare(curve_raw: dict, curve_cal: dict, label: str) -> go.Figure:
    mx = max(max(curve_raw["mean_pred"]), max(curve_raw["frac_pos"]), max(curve_cal["mean_pred"]), max(curve_cal["frac_pos"])) * 1.1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect calibration",
                             line=dict(color=P["t3"], dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=curve_raw["mean_pred"], y=curve_raw["frac_pos"], mode="lines+markers", name=f"{label} raw score",
                             line=dict(color=P["neutral"], width=2), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=curve_cal["mean_pred"], y=curve_cal["frac_pos"], mode="lines+markers", name=f"{label} calibrated",
                             line=dict(color=P["cyan"], width=2.6), marker=dict(size=8)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(title="Mean predicted value in bin (deciles)", range=[0, mx], tickformat=".0%"),
                      yaxis=dict(title="Observed default rate in bin", range=[0, mx], tickformat=".0%"))
    return fig


def fairness_age_bars(seg_with: pd.DataFrame, seg_without: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = go.Figure()
    for seg, name, col in [(seg_with, "With age", P["blue"]), (seg_without, "Without age", P["teal"])]:
        fig.add_trace(go.Bar(x=seg["group"].astype(str), y=seg[metric], name=name, marker=dict(color=col),
                             text=[f"{v*100:.1f}%" for v in seg[metric]], textposition="outside", cliponaxis=False,
                             textfont=dict(color=P["t1"], size=11.5),
                             hovertemplate=f"{name}<br>%{{x}}: %{{y:.1%}}<extra></extra>"))
    top = max(seg_with[metric].max(), seg_without[metric].max())
    fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=40, b=10), bargap=0.3,
                      xaxis=dict(title="Age band", type="category"),
                      yaxis=dict(title=title, tickformat=".0%", range=[0, top * 1.3 if top else 1]))
    return fig
