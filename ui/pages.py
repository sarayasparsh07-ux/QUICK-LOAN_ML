"""The six dashboard pages. Each receives a Ctx and renders itself; no page computes money on its own."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from quickloan import config as C
from quickloan import evaluation as E
from quickloan import finance as F
from quickloan.features import CANDIDATES

from . import charts as CH
from . import components as UI
from . import narrative as N
from .state import POLICIES, Board, board, explainer, predictions, sensitivity, thresholds
from .theme import FONT, P, PLOTLY_CONFIG


@dataclass
class Ctx:
    art: dict
    ev: dict
    b: Board
    focus: str
    interest: float
    policy: str

    @property
    def deployed(self) -> str:
        return self.art["model_name"]

    @property
    def policy_label(self) -> str:
        return POLICIES[self.policy]


def plot(fig: go.Figure, key: str) -> None:
    # Attach the design-system template explicitly: Streamlit swaps Plotly's global default template on import.
    fig.layout.template = pio.templates["quickloan"]
    # Explicit values: Streamlit's frontend fills *unset* backgrounds with app theme colours.
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, color=P["t2"]))
    fig.update_yaxes(ticks="outside", ticklen=6, tickcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, config=PLOTLY_CONFIG, key=key, width="stretch", theme=None)


def table(df: pd.DataFrame, column_config: dict | None = None, **kw) -> None:
    """st.dataframe with consistent fixed-decimal percentages (Streamlit's 'percent' format drops trailing zeros)."""
    df = df.copy()
    cfg = dict(column_config or {})
    for col, c in cfg.items():
        if isinstance(c, dict) and c.get("type_config", {}).get("format") == "percent" and col in df:
            df[col] = df[col] * 100
            c = {**c, "type_config": {**c["type_config"], "format": "%.2f%%" if "accuracy" in col else "%.1f%%"}}
            cfg[col] = c
    st.dataframe(df, column_config=cfg, **kw)


def policy_note(ctx: Ctx) -> str:
    if ctx.policy == "benchmark":
        return "All models use the common 0.50 threshold (the assignment's comparison)."
    return (f"Each model uses its own money-maximising threshold, re-optimised for {ctx.interest*100:.0f}% interest on "
            "training-set out-of-fold predictions (never the test set) — a separate analysis from the 0.50 benchmark.")


# =========================================================================== 1. Executive overview
def overview(ctx: Ctx) -> None:
    b, ev = ctx.b, ctx.ev
    UI.hero(len(ev["y_test"]), ctx.focus, ctx.interest, ctx.policy_label)
    L = b.row(b.champion)
    n_def = int(ev["y_test"].sum())
    total_def = b.baselines["total_default_principal"]
    UI.kpi_row([
        UI.kpi("Total evaluated loans", f"{len(ev['y_test']):,}", "Holdout · 25% stratified split, never used for tuning",
               P["blue"], "▤"),
        UI.kpi("Default rate (holdout)", UI.pct(ev["y_test"].mean(), 2),
               f"{n_def} of {len(ev['y_test']):,} test loans · {aed_c(total_def)} principal at risk · full dataset "
               f"{UI.pct(ev['audit']['default_rate'], 2)}", P["neg"], "!"),
        UI.kpi("Highest money saved", UI.aed(L["money_saved"], compact=True),
               f"vs approving everyone · at {ctx.interest*100:.0f}% interest", P["emerald"], "↑"),
        UI.kpi("Champion model", b.champion,
               f"{C.FAMILY[b.champion]} family · best under {'the 0.50 benchmark' if ctx.policy == 'benchmark' else 'CV business thresholds'}",
               P["teal"], "★"),
        UI.kpi("Default recall", UI.pct(L["recall"]), f"{int(L['tp'])} of {n_def} defaulters caught by the champion",
               P["cyan"], "◎"),
        UI.kpi("Loss avoided vs approve-all", UI.pct(L["money_saved"] / total_def),
               f"of the {aed_c(total_def)} approve-all default loss, net of interest forgone",
               P["emerald"] if L["money_saved"] > 0 else P["neg"], "⇡"),
    ])

    model_selection_cards(ctx)

    acc_sp, money_sp = N.spread_numbers(b)
    UI.section("The accuracy illusion", "Same data, same 1,500 test loans. Accuracy barely moves between models; "
               "the money does.", eyebrow="Executive overview")
    c1, c2 = st.columns([1.65, 1], gap="medium")
    with c1:
        with st.container(key="glow_ov_money"):
            UI.chart_header("Net money saved vs approving everyone",
                            f"Holdout · {ctx.policy_label} · foregone interest {ctx.interest*100:.0f}% · ★ = champion")
            plot(CH.money_bars(b.df, b.champion, ctx.focus, height=520), "ov_money")
    with c2:
        with st.container(key="panel_ov_spread"):
            UI.chart_header("Spread across all ten models")
            UI.html(f"""
<div style="display:grid;gap:12px;margin-top:8px">
 <div class="ql-tile" style="--c:{P['neutral']}"><div class="k">Accuracy — best minus worst</div>
  <div class="v ql-num">{acc_sp*100:.1f} pts</div><div class="s">{UI.pct(b.df['accuracy'].min(),1)} to {UI.pct(b.df['accuracy'].max(),1)} — barely any difference</div></div>
 <div class="ql-tile" style="--c:{P['emerald']}"><div class="k">Money saved — best minus worst</div>
  <div class="v ql-num">{UI.aed(money_sp, compact=True)}</div><div class="s">the decision QuickLoan actually cares about</div></div>
 <div class="ql-tile" style="--c:{P['amber']}"><div class="k">Reference strategies</div>
  <div class="s" style="font-size:12.5px;line-height:1.8;margin-top:6px">
  Approve everyone: <b style="color:{P['t1']}" class="ql-num">AED 0</b><br>
  Reject everyone: <b style="color:{UI.tone(b.baselines['reject_all'])}" class="ql-num">{UI.aed(b.baselines['reject_all'])}</b><br>
  Champion: <b style="color:{P['emerald']}" class="ql-num">{UI.aed(L['money_saved'])}</b></div></div>
 <div class="ql-tile" style="--c:{P['violet']}"><div class="k">Holdout uncertainty (95% bootstrap)</div>
  <div class="s" style="font-size:12.5px;line-height:1.8;margin-top:6px">
  Champion interval: <b style="color:{P['t1']}" class="ql-num">{UI.aed(L['ci_low'], compact=True)} – {UI.aed(L['ci_high'], compact=True)}</b><br>
  Wins in <b style="color:{P['t1']}" class="ql-num">{L['p_best']*100:.0f}%</b> of resamples ·
  <b style="color:{P['amber']}" class="ql-num">{int((~b.df['distinguishable']).sum()) - 1}</b> models statistically tied with it</div></div>
</div>""")

    UI.section("Management recommendation", "Generated from the live metrics — change the interest assumption or "
               "threshold policy in the sidebar and it rewrites itself.", eyebrow="For the CRO, CFO and Head of Lending")
    UI.callout(N.recommendation(b, ctx.deployed, ctx.art, 1 - float(ev["y_test"].mean())), color=P["teal"])
    UI.html(f'<div class="ql-muted" style="margin-top:10px">{policy_note(ctx)} Money saved uses the assignment\'s '
            'formula exactly: principal of caught defaulters − interest × principal of rejected repayers.</div>')


def aed_c(v):
    return UI.aed(v, compact=True)


METHOD_NOTE = ("The leaderboard compares all models using a common 0.50 classification threshold. The recommended "
               "deployment model is selected separately using training-only cross-validation and a business-oriented "
               "financial threshold. Both selections use the same financial payoff definition.")


CHAMPION_QUALIFIER = ("Selected using the common holdout benchmark for assignment comparison. Its reported result is a "
                      "model-ranking outcome, not an independent estimate of deployment performance.")


def deployment_figures(ev: dict, deployed: str, interest: float) -> dict:
    """Every figure on the deployment card, recomputed for the current interest assumption."""
    t = thresholds(ev, "business", interest)[deployed]
    curve = ev["threshold_curves"][deployed]
    k = int(np.argmin(np.abs(curve["grid"] - t)))
    n_tr, n_te = len(ev["y_train"]), len(ev["y_test"])
    cv_money = float(curve["caught"][k] - interest * curve["wrong"][k])           # training OOF, official sums
    p = predictions(ev, "business", interest)[deployed]
    ho = F.breakdown(ev["y_test"], p, ev["loan_test"], interest)
    ho_money = F.money_saved(ev["y_test"], p, ev["loan_test"], interest)           # official function
    nest = ev["nested_cv"]["summary"]
    return {"threshold": t, "cv_money": cv_money, "cv_per_1000": cv_money / n_tr * 1000, "n_train": n_tr,
            "holdout_money": ho_money, "holdout_per_1000": ho_money / n_te * 1000, "n_test": n_te,
            "tp": ho["n_tp"], "fp": ho["n_fp"], "n_def": int(ev["y_test"].sum()),
            "recall": ho["n_tp"] / max(int(ev["y_test"].sum()), 1),
            "nested_per_1000": nest["procedure_mean_money_per_1000"],
            "nested_times": int(nest["selection_counts"].get(deployed, 0)), "nested_folds": nest["outer_folds"]}


def model_selection_cards(ctx: Ctx) -> None:
    ev = ctx.ev
    bench = ctx.b if ctx.policy == "benchmark" else board(ev, ctx.interest, "benchmark")
    ch = bench.row(bench.champion)
    pct_i = f"{ctx.interest*100:.0f}%"
    left = UI.selection_card(
        "Holdout leaderboard champion", bench.champion,
        [("Money saved", UI.aed(ch["money_saved"], compact=True), f"vs approve-all · {pct_i} interest"),
         ("Threshold", "0.50", "common benchmark for all ten models"),
         ("Accuracy", UI.pct(ch["accuracy"], 2), f"{len(ev['y_test']):,} holdout loans"),
         ("Default recall", UI.pct(ch["recall"]), f"{int(ch['tp'])} of {int(ch['tp'] + ch['fn'])} defaulters caught")],
        "Highest financial result among models evaluated using the common benchmark threshold.",
        P["blue"], "A · benchmark", qualifier=CHAMPION_QUALIFIER)
    d = deployment_figures(ev, ctx.deployed, ctx.interest)
    right = UI.selection_card(
        "Recommended deployment model", ctx.deployed,
        [("Deployment threshold", f"{d['threshold']:.2f}", f"money-maximising at {pct_i} · training CV"),
         ("Training cross-validation", UI.aed(d["cv_money"], compact=True),
          f"{d['n_train']:,} out-of-fold loans · {UI.aed(d['cv_per_1000'], compact=True)} per 1,000"),
         ("Final holdout", UI.aed(d["holdout_money"], compact=True),
          f"{d['n_test']:,} loans · {UI.aed(d['holdout_per_1000'], compact=True)} per 1,000 · recall {UI.pct(d['recall'])}"),
         ("Nested CV (whole procedure)", UI.aed(d["nested_per_1000"], compact=True),
          f"per 1,000 loans at 20% · picked in {d['nested_times']} of {d['nested_folds']} outer folds")],
        f"Chosen on training cross-validation at 20% (highest out-of-fold money at its own business threshold) before "
        f"the holdout was opened; the threshold is re-optimised for the current interest. It catches {d['tp']} of "
        f"{d['n_def']} holdout defaulters while rejecting {d['fp']} good customers.",
        P["emerald"], "C · deployment")
    UI.section("Two model selections, two purposes", "Which model wins the benchmark comparison, and which one "
               "QuickLoan should actually deploy.", eyebrow="Model selection")
    UI.selection_cards(left, right, METHOD_NOTE)


# =========================================================================== 2. Leaderboard
SORTS = {"Money saved": ("money_saved", False), "Accuracy": ("accuracy", False), "Recall": ("recall", False),
         "Precision": ("precision", False), "F1-score": ("f1", False), "Model name": ("model", True)}


def leaderboard(ctx: Ctx) -> None:
    b = ctx.b
    UI.section("The Great Model Bake-Off", "All ten models scored on precisely the same 1,500 holdout loans. "
               "★ champion = highest money saved right now · ◉ focus = the model selected in the sidebar · "
               "◆ deployed = the model packaged in loan_model.joblib · 95% CI and P(best) come from a paired bootstrap "
               "of the holdout · ≈ = statistically tied with the champion.", eyebrow="Model leaderboard")
    with st.container(key="panel_lb_table"):
        c1, c2 = st.columns([3, 2])
        with c1:
            sort = st.segmented_control("Sort leaderboard by", list(SORTS), default="Money saved", key="lb_sort")
        with c2:
            UI.html(f'<div class="ql-muted" style="text-align:right;padding-top:30px">Interest {ctx.interest*100:.0f}% of principal · '
                    f'{ctx.policy_label}</div>')
        col, asc = SORTS[sort or "Money saved"]
        UI.leaderboard_table(b.df.sort_values(col, ascending=asc), b.champion, ctx.focus, ctx.deployed,
                             baseline=b.baseline_model)
        rej = b.baselines["reject_all"]
        bm = b.baseline_model
        UI.html(f'<div class="ql-muted" style="margin-top:10px">Reference — approve everyone: AED 0 · reject everyone: '
                f'<span style="color:{UI.tone(rej)}">{UI.aed(rej)}</span> · "Good rejected" = repayers the model would '
                f'have turned away (false positives). {policy_note(ctx)}</div>')
        UI.html(f'<div class="ql-muted" style="margin-top:6px"><b style="color:{P["t2"]}">Weak-learner baseline</b> '
                f'(brief Step 2, "feel the pain"): an untuned depth-3 tree at the 0.50 threshold catches '
                f'{int(bm["tp"])} of {int(bm["tp"] + bm["fn"])} defaulters (recall {UI.pct(bm["recall"])}) at '
                f'{UI.pct(bm["accuracy"], 2)} accuracy. It is shown for contrast and is not one of the ten ranked models; '
                'the tuned Decision Tree above is the single-model entry.</div>')
        UI.html(f'<div class="ql-muted" style="margin-top:6px"><b style="color:{P["t2"]}">Reading the ★ champion.</b> '
                f'{CHAMPION_QUALIFIER} The ◆ deployment model and its threshold are chosen on training data only '
                '(Executive Overview; Methodology → Models &amp; validation).</div>')

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(key="panel_lb_acc"):
            UI.chart_header("Accuracy — honest 0–100% scale", "Every bar looks the same. That is the point.")
            plot(CH.accuracy_bars(b.df, b.champion), "lb_acc")
    with c2:
        with st.container(key="panel_lb_rec"):
            UI.chart_header("Default recall — defaulters caught",
                            "Teal = most defaulters caught · label shows caught / total defaulters")
            plot(CH.recall_bars(b.df, b.champion), "lb_rec")
    with st.container(key="glow_lb_money"):
        UI.chart_header("Money saved vs approving everyone",
                        "Emerald = positive, coral = negative · white outline = champion, cyan outline = focus model")
        plot(CH.money_bars(b.df, b.champion, ctx.focus), "lb_money")
    with st.container(key="panel_lb_ci"):
        UI.chart_header("How certain is the ranking? — 95% intervals for money saved",
                        "Each line is the range money saved takes across 2,000 resamples of the 1,500 holdout loans "
                        "(same resample for every model). Shaded band = champion's interval · grey = statistically tied "
                        "with the champion · coral = reliably behind it · right column = share of resamples each model wins.")
        plot(CH.ci_forest(b.df, b.champion, ctx.focus, ctx.ev["bootstrap"]["B"]), "lb_ci")
    with st.container(key="panel_lb_scatter"):
        UI.chart_header("Accuracy vs recall — bubble size = money saved",
                        "Models sit within a sliver of accuracy but spread widely on recall; bigger bubbles save more. "
                        "Negative money saved is drawn at minimum size.")
        CH.set_family_lookup(C.FAMILY)
        plot(CH.scatter_acc_recall(b.df.reset_index(drop=True), b.champion), "lb_scatter")


# =========================================================================== 3. Financial performance
def financial(ctx: Ctx) -> None:
    b, ev, m = ctx.b, ctx.ev, ctx.focus
    r = b.row(m)
    total_def = b.baselines["total_default_principal"]
    UI.section(f"Where the money comes from — {m}",
               f"Official money-saved metric decomposed for the focus model at {ctx.interest*100:.0f}% foregone interest. "
               "Use the sidebar to change model, interest or threshold policy.", eyebrow="Financial performance")
    UI.tiles([
        UI.tile("Loans saved", UI.aed(r["loans_saved"], compact=True),
                f"{int(r['tp'])} defaulters correctly rejected — full principal avoided", P["emerald"]),
        UI.tile("Interest lost", "−" + UI.aed(r["interest_lost"], compact=True),
                f"{int(r['fp'])} good customers rejected × {ctx.interest*100:.0f}% of their principal", P["amber"]),
        UI.tile("Net money saved", UI.aed(r["money_saved"], compact=True),
                f"vs AED 0 for approving everyone · rank {int(b.df.index[b.df.model == m][0]) + 1} of 10",
                P["teal"] if r["money_saved"] >= 0 else P["neg"]),
        UI.tile("Missed defaults (exposure)", UI.aed(r["missed_default_principal"], compact=True),
                f"{int(r['fn'])} defaulters approved · reported separately, never subtracted twice", P["neg"]),
    ])
    st.write("")
    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1:
        with st.container(key="glow_fin_wf"):
            UI.chart_header("Money-saved waterfall", "Loans saved − interest lost = net money saved (the official metric)")
            plot(CH.waterfall(r.to_dict(), float(r["money_saved"])), "fin_wf")
    with c2:
        with st.container(key="panel_fin_split"):
            UI.chart_header("Defaulter principal: caught vs missed",
                            f"All {int(r['tp'] + r['fn'])} holdout defaulters, {UI.aed(total_def)} of principal. "
                            "Coral is the prize still on the table for better recall.")
            plot(CH.default_principal_split(r.to_dict(), total_def), "fin_split")
        with st.container(key="panel_fin_strat"):
            UI.chart_header("Versus simple strategies")
            items = [("Approve everyone", b.baselines["approve_all"]), ("Reject everyone", b.baselines["reject_all"]),
                     (f"{m} (focus)", float(r["money_saved"]))]
            if b.champion != m:
                items.append((f"{b.champion} (champion)", float(b.row(b.champion)["money_saved"])))
            plot(CH.strategy_bars(items), "fin_strat")

    UI.section("Interest sensitivity", "How money saved changes as the cost of rejecting a good customer rises. "
               "Higher foregone interest punishes models that reject aggressively."
               + (" Under the business policy each model's threshold is re-optimised at every interest level, so the "
                  "lines bend where a model changes its cut-off." if ctx.policy == "business" else ""),
               eyebrow="Scenario analysis")
    sens = sensitivity(ev, ctx.policy)
    CH.set_family_lookup(C.FAMILY)
    with st.container(key="panel_fin_sens"):
        UI.chart_header("Money saved by model across 5–50% foregone interest",
                        "Cyan = focus model · emerald = current champion · dashed amber = current assumption")
        plot(CH.sensitivity_lines(sens, m, b.champion, ctx.interest), "fin_sens")
        ranges = N.champion_ranges(sens)
        txt = " · ".join(f"<b>{lo*100:.0f}{'' if round(lo*100) == round(hi*100) else f'–{hi*100:.0f}'}%</b>: {UI.esc(n)}"
                         for lo, hi, n in ranges)
        UI.html(f'<div class="ql-muted" style="font-size:13px;margin-top:4px">Champion by interest range — {txt}</div>')

    UI.callout("Money saved is an <b>incremental</b> figure versus approving every applicant, under the assignment's "
               "simplified payoff: a defaulter loses the full principal, a rejected repayer costs only the foregone "
               "interest (a share of principal, not an APR). Recoveries, collateral, fees, operating and funding costs "
               "are excluded, so this is not audited accounting profit or an IFRS 9 expected-credit-loss figure. "
               "The ★ champion's figures come from ranking ten models on this same holdout: a model-ranking outcome, "
               "not an independent estimate of deployment performance.",
               title="How to read these numbers", color=P["amber"])


# =========================================================================== 4. Diagnostics
def diagnostics(ctx: Ctx) -> None:
    b, ev, m = ctx.b, ctx.ev, ctx.focus
    r = b.row(m)
    UI.section(f"Model diagnostics — {m}", f"Confusion matrix at {ctx.policy_label.lower()} "
               f"(threshold {r['threshold']:.2f}); curves use the model's raw scores.", eyebrow="Model diagnostics")
    c1, c2 = st.columns([1.3, 1], gap="medium")
    with c1:
        with st.container(key="panel_dg_cm"):
            UI.chart_header("Confusion matrix", "Positive class = default (1) · teal = correct decision, coral = error")
            plot(CH.confusion(int(r["tn"]), int(r["fp"]), int(r["fn"]), int(r["tp"])), "dg_cm")
    with c2:
        with st.container(key="panel_dg_metrics"):
            UI.chart_header("Classification metrics")
            rows = [("Accuracy", UI.pct(r["accuracy"], 2)), ("Default recall (sensitivity)", UI.pct(r["recall"])),
                    ("Precision", UI.pct(r["precision"])), ("F1-score", f"{r['f1']:.3f}"),
                    ("Specificity", UI.pct(r["specificity"])), ("Balanced accuracy", UI.pct(r["balanced_accuracy"])),
                    ("ROC-AUC", f"{r['roc_auc']:.3f}"), ("PR-AUC (average precision)", f"{r['pr_auc']:.3f}"),
                    ("Brier score (lower is better)", f"{r['brier']:.4f}")]
            UI.html('<table class="ql-table" style="min-width:0;width:100%">' + "".join(
                f'<tr><td style="text-align:left;color:{P["t2"]};white-space:normal">{k}</td><td><b>{v}</b></td></tr>'
                for k, v in rows) + "</table>")
            UI.html('<div class="ql-muted" style="margin-top:8px">TP = defaulter rejected · FP = repayer rejected · '
                    'FN = defaulter approved · TN = repayer approved.</div>')

    aucs = b.df.set_index("model")["roc_auc"].to_dict()
    aps = b.df.set_index("model")["pr_auc"].to_dict()
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(key="panel_dg_roc"):
            UI.chart_header("ROC curves — all ten models", "Focus model in cyan · legend shows ROC-AUC")
            plot(CH.roc_curves(ev["models"], m, aucs), "dg_roc")
    with c2:
        with st.container(key="panel_dg_pr"):
            UI.chart_header("Precision–recall curves", "More informative than ROC with a 12.5% default rate · legend shows PR-AUC")
            plot(CH.pr_curves(ev["models"], m, aps, float(ev["y_test"].mean())), "dg_pr")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(key="panel_dg_imp"):
            UI.chart_header("Permutation importance (raw inputs)",
                            "Measured on a validation fold carved from the training set — the test set is not used. "
                            "Associative, not causal.")
            plot(CH.importance_bars(ev["permutation"][m]), "dg_imp")
    with c2:
        with st.container(key="panel_dg_cal"):
            cal = ev["calibration"]
            if m == ctx.deployed and cal.get("curve_cal"):
                h, o = cal["holdout"], cal["candidates"]
                mth = cal["method"]
                verdict = (f"{mth.title()} calibration improved both training-OOF Brier and ECE, so it is used."
                           if cal["justified"] else
                           f"Neither calibrator improved both training-OOF Brier and ECE (best by Brier: {mth}, ECE "
                           f"{o['raw']['oof_ece']:.3f} → {o[mth]['oof_ece']:.3f}), so the raw score is kept.")
                prob = ("It is shown in the simulator as an estimated probability of default because its training-OOF "
                        f"ECE is within the {cal['ece_tolerance']*100:.0f}-point tolerance."
                        if cal["probability_justified"] else "It is not shown as a probability.")
                UI.chart_header("Calibration — raw score vs calibrated (holdout, deciles)",
                                f"{verdict} {prob} Holdout check: ECE raw {h['raw_ece']:.3f} vs {mth} {h['cal_ece']:.3f}; "
                                f"Brier {h['raw_brier']:.4f} vs {h['cal_brier']:.4f}.")
                plot(CH.calibration_compare(cal["curve_raw"], cal["curve_cal"], m), "dg_cal")
            else:
                UI.chart_header("Calibration (holdout, deciles)",
                                "Off the diagonal means scores are not trustworthy probabilities. Only the deployed model is "
                                "calibrated; select it as the focus model to compare raw and calibrated.")
                plot(CH.calibration_plot(ev["models"], m), "dg_cal")

    UI.section("All models side by side", "Threshold-free ranking quality (ROC-AUC, PR-AUC) next to the "
               "threshold-dependent metrics.")
    t = b.df[["model", "family", "accuracy", "balanced_accuracy", "recall", "specificity", "precision", "f1",
              "roc_auc", "pr_auc", "brier", "threshold"]].copy()
    table(t, hide_index=True, width="stretch", column_config={
        "model": "Model", "family": "Family",
        "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
        "balanced_accuracy": st.column_config.NumberColumn("Balanced acc.", format="percent"),
        "recall": st.column_config.NumberColumn("Recall", format="percent"),
        "specificity": st.column_config.NumberColumn("Specificity", format="percent"),
        "precision": st.column_config.NumberColumn("Precision", format="percent"),
        "f1": st.column_config.NumberColumn("F1", format="%.3f"),
        "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
        "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
        "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
        "threshold": st.column_config.NumberColumn("Threshold", format="%.2f")})

    UI.section("Segment diagnostics", f"{m} error rates by age band and home ownership on the holdout. "
               "Diagnostics only: small groups are noisy and this is not a fairness or discrimination test — see "
               "Methodology → Fairness check for the with/without-age comparison.")
    seg = E.segment_table(ev["X_test"], ev["y_test"], b.preds[m])       # current threshold policy and interest
    table(seg, hide_index=True, width="stretch", column_config={
        "dimension": "Dimension", "group": "Group", "n": "Loans", "defaulters": "Defaulters",
        "default_rate": st.column_config.NumberColumn("Default rate", format="percent"),
        "rejection_rate": st.column_config.NumberColumn("Rejection rate", format="percent"),
        "recall": st.column_config.NumberColumn("Recall", format="percent"),
        "false_rejection_rate": st.column_config.NumberColumn("Good customers rejected", format="percent")})


# =========================================================================== 5. Simulator
def simulator(ctx: Ctx) -> None:
    ev = ctx.ev
    rng = ev["observed_ranges"]
    UI.section("Loan decision simulator", "Enter an applicant. The saved pipeline applies the same feature "
               "engineering and encoding used in training, then compares the risk score with the decision threshold.",
               eyebrow="Underwriting")

    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        names = C.MODEL_ORDER
        sim_model = st.selectbox("Model", names, index=names.index(ctx.deployed), key="sim_model",
                                 format_func=lambda n: f"{n}  ◆ deployed" if n == ctx.deployed else n)
    with c2:
        biz_t = thresholds(ev, "business", ctx.interest)[sim_model]
        pct_i = ctx.interest * 100
        pol = st.radio("Decision threshold", ["business", "benchmark"], horizontal=True, key="sim_policy",
                       format_func=lambda k: (f"Business-optimised at {pct_i:.0f}% ({biz_t:.2f})" if k == "business"
                                              else "Benchmark (0.50)"),
                       help="The business threshold is re-optimised for the sidebar's interest assumption from "
                            "training out-of-fold predictions.")
    threshold = biz_t if pol == "business" else C.BENCHMARK_THRESHOLD

    with st.form("applicant", border=False):
        g1, g2 = st.columns(2, gap="medium")
        with g1:
            with st.container(key="panel_sim_g1"):
                UI.group_header(1, "Personal profile")
                a1, a2 = st.columns(2)
                age = a1.number_input("Age (years)", min_value=18, max_value=90, value=35, step=1)
                home = a2.selectbox("Home ownership", C.HOME_CATEGORIES, index=0)
            with st.container(key="panel_sim_g3"):
                UI.group_header(3, "Employment profile")
                a1, a2 = st.columns(2)
                emp = a1.number_input("Employment (years)", min_value=0.0, max_value=60.0, value=4.0, step=0.5)
                hist = a2.number_input("Credit history (months)", min_value=0, max_value=600, value=60, step=6)
        with g2:
            with st.container(key="panel_sim_g2"):
                UI.group_header(2, "Financial profile")
                a1, a2 = st.columns(2)
                inc = a1.number_input("Annual income (AED)", min_value=1_000, max_value=5_000_000, value=90_000, step=1_000)
                amt = a2.number_input("Loan amount (AED)", min_value=1_000, max_value=2_000_000, value=40_000, step=1_000)
                term = st.select_slider("Loan term (months)", options=C.LOAN_TERMS, value=36)
            with st.container(key="panel_sim_g4"):
                UI.group_header(4, "Existing obligations")
                a1, a2 = st.columns(2)
                n_loans = a1.number_input("Existing loans", min_value=0, max_value=20, value=1, step=1)
                pdef = a2.number_input("Past defaults", min_value=0, max_value=20, value=0, step=1)
                raw_ratio = amt / inc
                lti = min(round(raw_ratio, 2), C.DTI_CAP)
                capped = raw_ratio > C.DTI_CAP
                UI.html(f'<div class="ql-muted">Loan-to-income proxy (auto): <b style="color:{P["t1"]}" class="ql-num">'
                        f'{lti:.2f}</b>{" — capped at the observed maximum 3.00 (raw " + f"{raw_ratio:.2f})" if capped else ""}. '
                        'Dataset-specific <code>loan_amount ÷ annual_income</code>, not a conventional debt-to-income ratio.</div>')
        st.form_submit_button("Run underwriting decision", type="primary", width="stretch")

    applicant = dict(age=age, annual_income=float(inc), loan_amount=float(amt), loan_term_months=int(term),
                     employment_years=float(emp), credit_history_months=int(hist), num_existing_loans=int(n_loans),
                     past_defaults=int(pdef), debt_to_income=float(lti), home_ownership=home)
    warnings = []
    labels = {"age": "Age", "annual_income": "Annual income", "loan_amount": "Loan amount",
              "employment_years": "Employment years", "credit_history_months": "Credit history",
              "num_existing_loans": "Existing loans", "past_defaults": "Past defaults"}
    for k, lab in labels.items():
        lo, hi = rng[k]
        if not lo <= applicant[k] <= hi:
            warnings.append(f"{lab} {applicant[k]:,.0f} is outside the training range {lo:,.0f}–{hi:,.0f}.")
    if capped:
        warnings.append(f"Loan-to-income {raw_ratio:.2f} exceeds 3.00; the model sees 3.00 (as in the training data).")
    if emp > max(age - 16, 0):
        share = ev["audit"].get("implausible_employment", {}).get("share", 0)
        warnings.append(f"Employment years exceed age − 16. That is implausible ({share*100:.1f}% of dataset rows share "
                        "the issue, see Methodology) — check the inputs.")
    if home not in C.HOME_CATEGORIES:
        warnings.append("Unknown home-ownership category: encoded as all-zero, the model ignores it.")

    X = pd.DataFrame([applicant])[C.RAW_FEATURES]
    model = ev["fitted_models"][sim_model]
    score = float(model.predict_proba(X)[0, 1])
    reject = score >= threshold
    extra = []
    cal = ev["calibration"]
    if sim_model == ctx.deployed and cal.get("probability_justified"):
        src = cal["probability_source"]
        prob = score if src == "raw" else float(ev["calibrated_pipeline"].predict_proba(X)[0, 1])
        how = (f"raw score, already calibrated (training-OOF ECE {cal['candidates']['raw']['oof_ece']*100:.1f} pts)"
               if src == "raw" else f"{src} calibration")
        extra.append(("Estimated probability of default", f"{prob*100:.1f}%"))
        extra.append(("Probability basis", how))

    st.write("")
    c1, c2 = st.columns([1.1, 1], gap="medium")
    with c1:
        score_label = "Model score" if extra else "Risk score (uncalibrated)"
        UI.decision_panel(not reject, sim_model, score, threshold, score_label,
                          N.risk_interpretation(score, threshold, warnings),
                          f"business @ {ctx.interest*100:.0f}%" if pol == "business" else "benchmark", extra)
        if warnings:
            UI.html('<div style="height:12px"></div>')
            UI.callout("<br>".join("⚠ " + UI.esc(w) for w in warnings), title="Input warnings — inputs were not altered",
                       color=P["warn"])
    with c2:
        with st.container(key="panel_sim_gauge"):
            UI.chart_header("Risk score vs threshold", "Score ≥ threshold → reject")
            plot(_gauge(score, threshold, reject), "sim_gauge")
    st.write("")
    with st.container(key="panel_sim_reasons"):
        ex = explainer(sim_model)
        if ex is None:
            UI.chart_header("Why this decision? — reason codes",
                            f"Exact reason codes are available for the Decision Tree, Random Forest, Gradient Boosting, "
                            f"XGBoost and Logistic Regression. {sim_model} would need slow, sampling-based approximations, "
                            "so none are shown rather than unstable ones.")
        else:
            e = ex.explain(X)
            from quickloan.explain import reason_codes
            codes = reason_codes(e, reject)
            UI.chart_header("Why this decision? — reason codes",
                            ("Exact linear contributions (coefficient × distance from the training mean)"
                             if sim_model == "Logistic Regression" else "Exact Tree SHAP contributions")
                            + f" in {e.units}. Base value {e.base:+.3f} + contributions = model output {e.output:+.3f} "
                            f"(checked to {e.max_additivity_error:.0e}). Describes what moved this model's score — not causes.")
            r1, r2 = st.columns([1.5, 1], gap="medium")
            with r1:
                plot(CH.explanation_bars(e.table, e.units), "sim_reasons")
            with r2:
                head = "Main reasons for rejection" if reject else "Main factors supporting approval"
                body = "".join(f"<li>{UI.esc(c)}</li>" for c in codes) or "<li>No factor pushed in this direction.</li>"
                UI.callout(f'<ol style="margin:4px 0 0 18px;padding:0">{body}</ol>', title=head,
                           color=P["neg"] if reject else P["emerald"])
    UI.html('<div class="ql-muted" style="margin-top:12px">Educational prediction only — not a lending decision. '
            'The decision compares the raw model score with the threshold. An estimated probability of default is shown '
            'only for the deployed model, whose calibration was checked on training out-of-fold data; other models\' '
            'scores rank risk but are not presented as probabilities. Nothing here has been validated for fairness, '
            'affordability or regulatory compliance.</div>')


def _gauge(score, t, reject):
    col = P["neg"] if reject else P["emerald"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(valueformat=".3f", font=dict(size=40, color=col)),
        gauge=dict(axis=dict(range=[0, 1], tickformat=".1f", tickcolor=P["t3"], tickfont=dict(color=P["t2"])),
                   bar=dict(color=col, thickness=0.28), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, t], color="rgba(53,214,160,0.13)"),
                          dict(range=[t, 1], color="rgba(255,114,127,0.13)")],
                   threshold=dict(line=dict(color=P["amber"], width=3), thickness=0.9, value=t))))
    fig.add_annotation(x=0.5, y=-0.08, xref="paper", yref="paper", showarrow=False,
                       text=f"amber line = threshold {t:.2f} · green zone approve · coral zone reject",
                       font=dict(color=P["t3"], size=11.5))
    fig.update_layout(height=290, margin=dict(l=30, r=30, t=20, b=40))
    return fig


# =========================================================================== 6. Methodology & data
def paired_gaps(per_model: pd.DataFrame, ref: str) -> pd.DataFrame:
    """Outer-fold money per 1,000 of every model minus the reference model, paired by fold."""
    w = per_model.pivot(index="fold", columns="model", values="money_per_1000")
    d = w.sub(w[ref], axis=0)
    out = pd.DataFrame({"gap_mean": d.mean(), "gap_sd": d.std()})
    out["gap_t"] = out["gap_mean"] / (out["gap_sd"] / np.sqrt(len(w))).replace(0, np.nan)
    return out.rename_axis("model").reset_index()


def materiality_sentence(gap: pd.DataFrame, deployed: str) -> str:
    g = gap[gap["model"] != deployed].sort_values("gap_mean", ascending=False).iloc[0]
    if g["gap_mean"] <= 0:
        return f"No candidate beat {deployed} on average across the outer folds."
    if g["gap_t"] >= 2:
        return (f"{g['model']} beat {deployed} by AED {g['gap_mean']:,.0f} per 1,000 loans on average with a consistent "
                f"fold-to-fold advantage (t = {g['gap_t']:.1f}) — the selection rule should be revisited.")
    return (f"The best alternative, {g['model']}, beat {deployed} by only AED {g['gap_mean']:,.0f} per 1,000 loans on "
            f"average, against a fold-to-fold spread of AED {g['gap_sd']:,.0f} (t = {g['gap_t']:.1f}): no candidate is "
            f"materially better, so {deployed} is retained.")


def money_at(td: dict, t: float) -> float:
    c = td["curve"]
    return float(c.loc[np.isclose(c["threshold"], t), "money_saved"].iloc[0])


def band_sentence(td: dict) -> str:
    b1, b2 = td["band_99"], td["band_98"]
    one = (f"Only {b1[0]:.2f} is within 1% of the maximum" if b1[0] == b1[1]
           else f"Thresholds {b1[0]:.2f}–{b1[1]:.2f} stay within 1% of the maximum")
    return f"{one}; {b2[0]:.2f}–{b2[1]:.2f} stay within 2%."


def nested_cv_section(ev: dict, deployed: str) -> None:
    """Evaluations A / B / C, nested cross-validation of the selection procedure, and threshold stability."""
    n, td = ev["nested_cv"], ev["threshold_diagnostics"]
    s, f = n["summary"], n["folds"]
    bench_champ = ev["champion_050_20"]
    lb = ev["leaderboard_050"].set_index("model")
    bz = ev["leaderboard_business"].set_index("model")
    UI.chart_header("Three evaluations, three different questions",
                    "Each answers one question and none is reused for another; all use the official money-saved formula.")
    UI.html(
        '<div class="ql-muted" style="margin:0 0 14px;line-height:1.65">'
        f'<b>A · Benchmark leaderboard</b> — all ten models on the 1,500-loan holdout at the common 0.50 threshold. '
        f'Answers "which model ranks best like-for-like?" Champion: <b>{UI.esc(bench_champ)}</b> '
        f'({UI.aed(float(lb.loc[bench_champ, "money_saved"]))} at 20%) — a model-ranking outcome, not an independent '
        f'estimate of deployment performance.<br>'
        f'<b>B · Nested cross-validation</b> — the complete selection procedure (feature selection, tuning, inner '
        f'out-of-fold thresholds, model choice) repeated inside {s["outer_folds"]} outer folds of the 4,500 training '
        f'loans and scored on rows it never saw. Answers "how good is the <i>procedure</i>, and is it stable?" '
        f'Estimate: <b>{UI.aed(s["procedure_mean_money_per_1000"])}</b> per 1,000 loans '
        f'(sd {UI.aed(s["procedure_sd_money_per_1000"])}).<br>'
        f'<b>C · Final deployment holdout</b> — the procedure re-run once on all 4,500 training loans picks '
        f'<b>{UI.esc(deployed)}</b> at threshold {ev["threshold_diagnostics"]["chosen"]:.2f}; the holdout is opened '
        f'once to report it: <b>{UI.aed(float(bz.loc[deployed, "money_saved"]))}</b> '
        f'({UI.aed(float(bz.loc[deployed, "money_saved"]) / len(ev["y_test"]) * 1000)} per 1,000 loans).</div>')

    counts = ", ".join(f"{UI.esc(k)} {v}/{s['outer_folds']}" for k, v in s["selection_counts"].items())
    UI.chart_header("Nested cross-validation — is the selection procedure stable?",
                    f"Outer StratifiedKFold({s['outer_folds']}, shuffled, seed {s['outer_seed']}) over the "
                    f"{s['n_training_loans']:,} training loans; inner StratifiedKFold({s['inner_folds']}, seed "
                    f"{s['inner_seed']}) inside each outer-training part. The official holdout is never used.")
    tbl = f[["fold", "model", "threshold", "inner_cv_money_per_1000", "money_saved", "money_per_1000", "accuracy",
             "recall", "precision", "f1", "false_rejection_rate", "runner_up", "inner_margin_per_1000", "features"]]
    table(tbl, hide_index=True, width="stretch", column_config={
        "fold": "Outer fold", "model": "Selected model", "features": "Engineered features kept",
        "threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
        "inner_cv_money_per_1000": st.column_config.NumberColumn("Inner CV / 1,000", format="%,.0f"),
        "money_saved": st.column_config.NumberColumn("Outer-val money (AED)", format="%,.0f"),
        "money_per_1000": st.column_config.NumberColumn("Outer-val / 1,000", format="%,.0f"),
        "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
        "recall": st.column_config.NumberColumn("Recall", format="percent"),
        "precision": st.column_config.NumberColumn("Precision", format="percent"),
        "f1": st.column_config.NumberColumn("F1", format="%.3f"),
        "false_rejection_rate": st.column_config.NumberColumn("Good rejected", format="percent"),
        "runner_up": "Inner runner-up",
        "inner_margin_per_1000": st.column_config.NumberColumn("Inner margin / 1,000", format="%,.0f")})
    UI.html(
        f'<div class="ql-muted" style="margin:6px 0 14px">Selected per fold: <b>{counts}</b>. Thresholds ranged '
        f'{s["threshold_min"]:.2f}–{s["threshold_max"]:.2f} (mean {s["threshold_mean"]:.2f}). Mean outer recall '
        f'{UI.pct(s["procedure_mean_recall"])}, precision {UI.pct(s["procedure_mean_precision"])}, accuracy '
        f'{UI.pct(s["procedure_mean_accuracy"])}, good customers rejected {UI.pct(s["procedure_mean_false_rejection"])}. '
        f'Inner CV was optimistic by {UI.aed(s["optimism_per_1000"])} per 1,000 loans on average — the price of '
        f'choosing on the same rows that were scored.</div>')
    pm = n["per_model_summary"].reset_index()
    gap = paired_gaps(n["per_model"], deployed)
    pm = pm.merge(gap, on="model")[["model", "times_selected", "mean_money_per_1000", "sd_money_per_1000", "gap_mean",
                                    "gap_sd", "gap_t", "mean_inner_cv_per_1000", "mean_recall", "mean_precision",
                                    "mean_accuracy", "mean_threshold", "sd_threshold"]]
    pm["gap_t"] = pm["gap_t"].map(lambda t: "—" if pd.isna(t) else f"{t:+.1f}")       # undefined for the reference
    UI.chart_header("Every candidate inside the same outer folds",
                    "Each model tuned and thresholded on the inner folds only, then scored on the outer-validation rows. "
                    f"Shows whether a model chosen less often is materially better. Gap = model minus {UI.esc(deployed)}, "
                    "fold by fold (paired).")
    table(pm, hide_index=True, width="stretch", column_config={
        "model": "Model", "times_selected": "Times selected",
        "gap_mean": st.column_config.NumberColumn("Gap vs deployed / 1,000", format="%,.0f"),
        "gap_sd": st.column_config.NumberColumn("Gap sd", format="%,.0f"),
        "gap_t": st.column_config.TextColumn("Gap t"),
        "mean_money_per_1000": st.column_config.NumberColumn("Outer-val / 1,000 (mean)", format="%,.0f"),
        "sd_money_per_1000": st.column_config.NumberColumn("sd", format="%,.0f"),
        "mean_inner_cv_per_1000": st.column_config.NumberColumn("Inner CV / 1,000", format="%,.0f"),
        "mean_recall": st.column_config.NumberColumn("Recall", format="percent"),
        "mean_precision": st.column_config.NumberColumn("Precision", format="percent"),
        "mean_accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
        "mean_threshold": st.column_config.NumberColumn("Threshold (mean)", format="%.2f"),
        "sd_threshold": st.column_config.NumberColumn("Threshold sd", format="%.2f")})
    UI.html(f'<div class="ql-muted" style="margin:6px 0 14px">{UI.esc(materiality_sentence(gap, deployed))}</div>')

    a = td["at_chosen"]
    UI.chart_header(f"Threshold stability — {deployed} at {td['chosen']:.2f}",
                    "Training out-of-fold scores only, 20% interest.")
    UI.html(
        f'<div class="ql-muted" style="margin:0 0 18px">The chosen cut-off earns {UI.aed(a["money_saved"])} out-of-fold '
        f'(recall {UI.pct(a["recall"])}, precision {UI.pct(a["precision"])}, good customers rejected '
        f'{UI.pct(a["false_rejection_rate"])}). {UI.esc(band_sentence(td))} The five inner folds individually prefer '
        f'{", ".join(f"{t:.2f}" for t in td["per_fold"]["best_threshold"])} (mean {td["per_fold_threshold_mean"]:.2f}); '
        f'at the 0.50 benchmark the same scores keep only {UI.pct(money_at(td, 0.5) / td["max_money"], 0)} of the maximum, '
        'so moving away from 0.50 matters far more than the second decimal.</div>')


def methodology(ctx: Ctx) -> None:
    ev, art = ctx.ev, ctx.art
    au = ev["audit"]
    UI.section("Methodology and data", "How the numbers were produced, what they assume and where they stop being "
               "reliable.", eyebrow="Methodology & data")
    tabs = st.tabs(["Data quality", "Exploratory analysis", "Feature engineering", "Models & validation",
                    "Instructor benchmark", "Fairness check", "Assumptions & limitations", "Artifact contract"])

    with tabs[0]:
        UI.tiles([
            UI.tile("Rows × columns", f"{au['shape'][0]:,} × {au['shape'][1]}", "10 features + target, as expected", P["blue"]),
            UI.tile("Missing / duplicates", f"{au['missing']} / {au['duplicates']}", "no imputation or de-duplication needed", P["emerald"]),
            UI.tile("Default rate (full dataset)", UI.pct(au["default_rate"], 2),
                    f"{au['target_counts'][1]:,} defaulted · {au['target_counts'][0]:,} repaid · holdout "
                    f"{UI.pct(float(ev['y_test'].mean()), 2)}", P["neg"]),
            UI.tile("DTI inconsistencies", f"{len(au['dti_mismatch'])} rows", "kept as recorded for training", P["amber"]),
        ])
        st.write("")
        c1, c2 = st.columns([1, 1.3], gap="medium")
        with c1:
            with st.container(key="panel_md_sanity"):
                UI.chart_header("Validation checks")
                names = {"loan_amount_positive": "Loan amounts positive", "annual_income_positive": "Incomes positive",
                         "target_binary": "Target is 0 / 1", "home_ownership_known": "Home ownership in {rent, own, mortgage}",
                         "employment_le_age_minus_16": "Employment years ≤ age − 16",
                         "loan_term_in_allowed_set": "Term in {12, 24, 36, 48, 60}",
                         "non_negative_counts": "Counts and durations ≥ 0"}
                soft = {"employment_le_age_minus_16"}          # plausibility flag, not a hard failure
                def status(k, v):
                    if v:
                        return P["pos"], "✓ pass"
                    return (P["warn"], "⚠ flagged") if k in soft else (P["neg"], "✕ fail")
                UI.html('<table class="ql-table" style="min-width:0">' + "".join(
                    f'<tr><td style="text-align:left;color:{P["t2"]}">{names.get(k, k)}</td>'
                    f'<td style="color:{status(k, v)[0]}"><b>{status(k, v)[1]}</b></td></tr>'
                    for k, v in au["sanity"].items()) + "</table>")
                ie = au.get("implausible_employment")
                if ie and ie["n"]:
                    UI.html(f'<div class="ql-muted" style="margin-top:10px;color:{P["t2"]}"><b style="color:{P["warn"]}">'
                            f'⚠ {ie["n"]} rows ({ie["share"]*100:.1f}%)</b> report more years employed than years since '
                            f'age 16 (mostly applicants aged 21–29; default rate {ie["default_rate"]*100:.1f}%). The '
                            'dataset is synthetic and there is no evidence which field is wrong, so the rows are kept and '
                            'flagged rather than removed — removal would also break the assignment\'s reference split.</div>')
                cats = " · ".join(f"{k}: {v:,}" for k, v in au["categories"].items())
                UI.html(f'<div class="ql-muted" style="margin-top:8px">home_ownership levels — {cats}</div>')
        with c2:
            with st.container(key="panel_md_dti"):
                UI.chart_header("The debt_to_income issue",
                                "14 stored values differ from round(loan_amount / annual_income, 2). Eight are capped at "
                                "3.00; six are half-way cases stored rounded down. Training uses the values as recorded; "
                                "the simulator computes the same proxy and caps it at 3.00.")
                table(au["dti_mismatch"], width="stretch", height=300, column_config={
                    "loan_amount": st.column_config.NumberColumn("Loan (AED)", format="%,.0f"),
                    "annual_income": st.column_config.NumberColumn("Income (AED)", format="%,.0f"),
                    "debt_to_income": st.column_config.NumberColumn("Stored DTI", format="%.2f"),
                    "raw_ratio": st.column_config.NumberColumn("Raw ratio", format="%.4f"),
                    "reason": "Reason"})
        with st.container(key="panel_md_split"):
            UI.chart_header("Split reconciliation with the instructor reference",
                            "25% test · random_state = 1 · stratified on default — every value matches")
            labels = {"n_test": "Test loans", "n_def": "Defaulters", "n_good": "Repaid loans",
                      "total_def_loans": "Defaulted principal (AED)", "total_good_loans": "Repaid principal (AED)"}
            sc = pd.DataFrame([{"measure": labels.get(k, k), "reference": f"{v['reference']:,.0f}",
                                "this build": f"{v['actual']:,.0f}", "match": "✓ match" if v["match"] else "✕ differs"}
                               for k, v in ev["split_check"].items()])
            table(sc, hide_index=True, width="stretch")
        with st.container(key="panel_md_desc"):
            UI.chart_header("Feature distributions (full dataset)")
            table(au["describe"].round(2), width="stretch")

    with tabs[1]:
        eda = ev["eda"]
        overall = au["default_rate"]
        specs = [("past_defaults_band", "Past defaults"), ("dti_band", "Loan-to-income band"),
                 ("income_band", "Annual-income quintile"), ("home_ownership", "Home ownership"),
                 ("age_band", "Age band"), ("loan_term_months", "Loan term (months)")]
        for i in range(0, len(specs), 2):
            cols = st.columns(2, gap="medium")
            for col, (k, lab) in zip(cols, specs[i:i + 2]):
                with col, st.container(key=f"panel_eda_{k}"):
                    UI.chart_header(f"Default rate by {lab.lower()}",
                                    f"Full dataset (6,000 loans) · dotted line = portfolio default rate {overall*100:.2f}% · "
                                    "coral = well above it, emerald = well below")
                    plot(CH.default_rate_bars(eda[k], overall, lab), f"eda_{k}")
        with st.container(key="panel_eda_corr"):
            UI.chart_header("Correlation matrix (numeric features and target)",
                            "Debt-to-income, past defaults, loan amount and (negatively) income carry the linear signal; "
                            "age and tenure variables are nearly uncorrelated with default on their own.")
            plot(CH.corr_heatmap(eda["corr"]), "eda_corr")

    with tabs[2]:
        fe = ev["studies"]["feature_engineering"].copy()
        UI.callout(f"Each candidate was added on its own to a scaled logistic regression and a gradient-boosting model "
                   f"and scored with 5-fold CV average precision on the <b>training set only</b>. A feature is kept if it "
                   f"lifts mean PR-AUC by more than 0.002 — roughly the noise floor shown by the "
                   f"<code>credit_history_years</code> control, which is a pure rescaling and cannot add information. "
                   f"Retained: <b>{', '.join(ev['features']) or 'none'}</b>. None of these is an EMI — the data has no "
                   f"rate or amortisation schedule.", color=P["teal"])
        st.write("")
        meaning = {f"+ {k}": v[1] for k, v in CANDIDATES.items()}
        rows = []
        for r in fe.itertuples(index=False):
            dec = r.decision
            col = P["pos"] if dec == "RETAINED" else (P["t3"] if dec.startswith("rejected") else P["t2"])
            rows.append(f'<tr><td style="white-space:normal;max-width:240px"><b>{UI.esc(r.feature_set)}</b></td>'
                        f'<td style="white-space:normal;max-width:230px;color:{P["t2"]}"><code>{UI.esc(r.formula)}</code></td>'
                        f'<td style="white-space:normal;max-width:330px;text-align:left;color:{P["t2"]}">{UI.esc(meaning.get(r.feature_set, ""))}</td>'
                        f'<td>{r.LR_pr_auc:.4f}</td><td>{r.GB_pr_auc:.4f}</td>'
                        f'<td style="color:{UI.tone(r.delta_mean_pr_auc) if abs(r.delta_mean_pr_auc) > 0.002 else P["t2"]}">{r.delta_mean_pr_auc:+.4f}</td>'
                        f'<td style="color:{col}"><b>{UI.esc(dec)}</b></td></tr>')
        with st.container(key="panel_fe_table"):
            UI.html('<div class="ql-table-wrap"><table class="ql-table" style="min-width:980px"><thead><tr>'
                    '<th>Feature set</th><th>Formula</th><th style="text-align:left">Financial meaning</th><th>LR PR-AUC</th>'
                    '<th>GB PR-AUC</th><th>Δ mean PR-AUC</th><th>Decision</th></tr></thead><tbody>'
                    + "".join(rows) + "</tbody></table></div>")

    with tabs[3]:
        tun = ev["studies"]["tuning"].copy()
        tun["best_params"] = tun["best_params"].map(lambda d: ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in d.items()))
        UI.callout("Hyperparameters were chosen by 5-fold stratified CV on the 4,500 training loans, maximising average "
                   "precision (a threshold-free metric suited to the 12.5% minority class). Voting soft-averages the three "
                   "tuned single models; Stacking feeds out-of-fold probabilities of tuned Random Forest, XGBoost and "
                   "Logistic Regression into a logistic 'manager'. The test set was used once, for the final scores.",
                   color=P["teal"])
        st.write("")
        table(tun[["model", "best_params", "cv_pr_auc", "n_candidates", "seconds"]], hide_index=True, width="stretch",
                     column_config={"cv_pr_auc": st.column_config.NumberColumn("CV PR-AUC", format="%.4f"),
                                    "best_params": "Selected hyperparameters", "n_candidates": "Grid size",
                                    "seconds": "Search time (s)", "model": "Model"})
        UI.chart_header("Class weighting — evaluated, not applied blindly",
                        "Training-CV results at the 0.50 threshold. Re-weighting shifts the decision boundary much like "
                        "lowering the threshold; the build keeps unweighted models for the benchmark and moves the "
                        "threshold explicitly instead, which keeps scores interpretable and the comparison fair.")
        table(ev["studies"]["class_weight"], hide_index=True, width="stretch", column_config={
            "cv_accuracy": st.column_config.NumberColumn("CV accuracy", format="percent"),
            "cv_recall": st.column_config.NumberColumn("CV recall", format="percent"),
            "cv_precision": st.column_config.NumberColumn("CV precision", format="percent"),
            "cv_pr_auc": st.column_config.NumberColumn("CV PR-AUC", format="%.4f"),
            "cv_money_saved": st.column_config.NumberColumn("CV money saved (AED)", format="%,.0f")})
        cw = ev["studies"]["class_weight"]
        cvb = ev["studies"]["cv_summary"].set_index("model")["cv_money_saved_business"]
        bal = cw[cw["class_weight"] == "balanced"].set_index("model")["cv_money_saved"]
        wins = [m for m in bal.index if cvb[m] >= bal[m]]
        UI.html(f'<div class="ql-muted" style="margin:6px 0 18px">Evidence: for <b>{len(wins)} of {len(bal)}</b> models the '
                'explicit CV business threshold on the unweighted model saves at least as much out-of-fold money as '
                'class weighting at 0.50 (' + "; ".join(f"{UI.esc(m)}: {UI.aed(cvb[m], compact=True)} vs "
                f"{UI.aed(bal[m], compact=True)}" for m in bal.index) + ').</div>')
        UI.chart_header("Cross-validated performance, business thresholds and the deployment choice",
                        f"Business threshold = the cut-off maximising out-of-fold money saved at 20%. "
                        f"Deployed: <b>{UI.esc(ctx.deployed)}</b> — {UI.esc(art['selection_rule'])}")
        table(ev["studies"]["cv_summary"], hide_index=True, width="stretch", column_config={
            "cv_accuracy": st.column_config.NumberColumn("CV accuracy", format="percent"),
            "cv_recall": st.column_config.NumberColumn("CV recall", format="percent"),
            "cv_roc_auc": st.column_config.NumberColumn("CV ROC-AUC", format="%.3f"),
            "cv_pr_auc": st.column_config.NumberColumn("CV PR-AUC", format="%.3f"),
            "cv_brier": st.column_config.NumberColumn("CV Brier", format="%.4f"),
            "cv_money_saved_050": st.column_config.NumberColumn("CV money @0.50", format="%,.0f"),
            "business_threshold": st.column_config.NumberColumn("Business threshold", format="%.2f"),
            "cv_money_saved_business": st.column_config.NumberColumn("CV money @business", format="%,.0f")})
        nested_cv_section(ev, ctx.deployed)
        bz = ev["leaderboard_business"][["model", "accuracy", "recall", "fp", "money_saved"]]
        with st.container(key="panel_md_thr"):
            UI.chart_header("Business threshold by interest assumption",
                            "Each model's money-maximising cut-off, re-optimised on training out-of-fold scores for every "
                            "interest level the slider allows (focus model in cyan, dashed amber = current assumption). The "
                            "dotted line is the theoretical break-even i/(1+i), exact only for calibrated probabilities.")
            CH.set_family_lookup(C.FAMILY)
            plot(CH.threshold_by_interest(ev["threshold_curves"], ctx.focus, ctx.interest, F.break_even_threshold), "md_thr")
        UI.chart_header("Holdout results at business thresholds tuned for 20%",
                        "The exported business leaderboard (model_leaderboard_business.csv) — reported separately from "
                        "the 0.50 benchmark.")
        table(bz, hide_index=True, width="stretch", column_config={
            "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
            "recall": st.column_config.NumberColumn("Recall", format="percent"),
            "fp": "Good rejected", "money_saved": st.column_config.NumberColumn("Money saved (AED)", format="%,.0f")})

    with tabs[4]:
        b20 = board(ev, C.DEFAULT_INTEREST, "benchmark")
        ref = ev["reference_results"]
        ours = b20.df[["model", "accuracy", "recall", "money_saved"]].copy()
        ours["ref_accuracy"] = ours["model"].map(lambda m: ref[m]["acc"])
        ours["ref_recall"] = ours["model"].map(lambda m: ref[m]["recall"])
        ours["ref_money"] = ours["model"].map(lambda m: ref[m]["money"])
        ours["delta_money"] = ours["money_saved"] - ours["ref_money"]
        rc = N.rank_corr(ours["money_saved"].to_numpy(), ours["ref_money"].to_numpy())
        best_ref = max(ref, key=lambda m: ref[m]["money"])
        UI.callout(
            f"At 20% and the 0.50 threshold, this build's champion is <b>{UI.esc(b20.champion)}</b> with "
            f"{UI.aed(b20.row(b20.champion)['money_saved'])}; the instructor reference champion is <b>{best_ref}</b> with "
            f"{UI.aed(ref[best_ref]['money'])}. Spread here: {UI.aed(ours['money_saved'].max() - ours['money_saved'].min())} "
            f"(reference AED 2,605,920). Rank correlation of money saved with the reference: <b>{rc:.2f}</b>. "
            "The split is identical, so differences come from configuration: this build tunes every model by "
            "cross-validation, adds two engineered features, scales inputs for LR and KNN, and soft-votes. "
            "Results were not adjusted to match the reference.", title="Same split, independent models", color=P["violet"])
        st.write("")
        with st.container(key="panel_md_bench"):
            UI.chart_header("Money saved at 20%: this build vs instructor reference")
            plot(CH.benchmark_compare(ours, ref), "md_bench")
        table(ours, hide_index=True, width="stretch", column_config={
            "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
            "ref_accuracy": st.column_config.NumberColumn("Ref. accuracy", format="percent"),
            "recall": st.column_config.NumberColumn("Recall", format="percent"),
            "ref_recall": st.column_config.NumberColumn("Ref. recall", format="percent"),
            "money_saved": st.column_config.NumberColumn("Money saved", format="%,.0f"),
            "ref_money": st.column_config.NumberColumn("Ref. money", format="%,.0f"),
            "delta_money": st.column_config.NumberColumn("Δ vs ref.", format="%+,.0f")})

    with tabs[5]:
        fairness_tab(ctx)

    with tabs[6]:
        items = [
            ("Simplified payoff", "Default = full principal lost; rejected repayer = foregone interest only. No recoveries, "
             "collateral, fees, operating or funding costs. Not audited profit and not an expected-credit-loss model."),
            ("Interest assumption", "The slider is foregone interest as a share of principal over the loan's life, not an "
             "annual percentage rate. All money figures recompute from stored predictions with the official formula."),
            ("Class imbalance", "12.5% defaults. Accuracy is dominated by repayers, so recall, PR-AUC and money saved "
             "carry the decision."),
            ("Holdout limits", "One split of 1,500 loans with 188 defaulters. A paired bootstrap (2,000 resamples) "
             "puts a 95% interval on every model's money saved; several top models are statistically tied, so the "
             "ranking among them is not settled by this sample. The bootstrap varies the test sample only, not training."),
            ("Thresholds", "The 0.50 benchmark is a convention, not an optimum. Business thresholds are re-optimised for "
             "the selected interest on training out-of-fold scores; the exported business leaderboard uses 20%."),
            ("Calibration", "Decisions use raw scores against tuned thresholds. Only the deployed model is calibrated "
             "(method and justification decided on training out-of-fold data); every other score is a ranking "
             "score, labelled as such."),
            ("Extrapolation", "Inputs outside the observed ranges produce warnings. Tree models flat-line beyond the data; "
             "linear models extrapolate — neither is reliable there."),
            ("Feature importance", "Permutation importance shows association with predictions on a validation fold. It "
             "is not causal and is diluted between correlated inputs (loan amount, income, DTI)."),
            ("Interpretability", "The best-earning models are ensembles that need post-hoc explanation for adverse-action "
             "notices; the single tree and logistic regression are directly explainable."),
            ("Fairness", "Segment error rates and the with/without-age comparison (Fairness check tab) are diagnostics, "
             "not proof of fairness. Removing a variable does not remove its proxies. A real lender would need legal "
             "review, fairness testing, and model-risk and regulatory validation before any use. This dashboard is an "
             "educational tool, not a certified approval product."),
        ]
        for i in range(0, len(items), 2):
            cols = st.columns(2, gap="medium")
            for col, (t, d) in zip(cols, items[i:i + 2]):
                with col:
                    UI.callout(UI.esc(d), title=t, color=P["amber"])
                    st.write("")

    with tabs[7]:
        tr = art["training"]
        UI.callout(f"<b>loan_model.joblib</b> holds the deployed pipeline (<b>{UI.esc(art['display_name'])}</b>) "
                   "including feature engineering and one-hot encoding, so the dashboard passes the raw ten columns and "
                   "never rebuilds dummy columns. Schema <code>" + UI.esc(art["schema_version"]) + "</code>. "
                   f"Created {UI.esc(tr['created_utc'])} · dataset sha256 <code>{tr['dataset_sha256'][:16]}…</code>",
                   color=P["teal"])
        st.write("")
        contract = pd.DataFrame([
            ("pipeline", "fitted sklearn pipeline · raw 10-column DataFrame → predict_proba"),
            ("raw_features", ", ".join(art["raw_features"])),
            ("model / columns", "starter-compatible wrapper · accepts raw inputs or the instructor app.py's one-hot "
                                "columns (" + ", ".join(art["columns"][-3:]) + ") · identical predictions"),
            ("portability", "project classes pickled by value: loads next to the instructor's app.py and loans.csv "
                            "without this project's code (pinned library versions still required)"),
            ("categorical_levels", str(art["categorical_levels"])),
            ("engineered_features", ", ".join(art["engineered_features"]) or "none"),
            ("thresholds", f"benchmark {art['thresholds']['benchmark']:.2f} · business {art['thresholds']['business']:.2f} at 20% "
                           f"· business_by_interest for 5–50% ({art['thresholds']['business_rule']})"),
            ("selected_on", f"{art['selected_on']} — {art['selection_rule']}"),
            ("selection_methodology", art["selection_methodology"]["assessment"]),
            ("metrics.nested_cv", f"{art['metrics']['nested_cv']['outer_folds']} outer × "
                                  f"{art['metrics']['nested_cv']['inner_folds']} inner folds · procedure estimate "
                                  f"{UI.aed(art['metrics']['nested_cv']['procedure_mean_money_per_1000'])} per 1,000 loans · "
                                  "selections " + ", ".join(f"{k} {v}" for k, v in
                                                            art["metrics"]["nested_cv"]["selection_counts"].items())),
            ("positive class", "default = 1 (reject when score ≥ threshold)"),
            ("calibrated_pipeline", (f"CalibratedClassifierCV ({art['calibration']['method']}) — justified on training OOF"
                                     if art.get("calibrated_pipeline") is not None else
                                     "none — neither calibrator improved both training-OOF Brier and ECE, so the raw score is kept")),
            ("loan_model_leaderboard_champion.joblib", f"same contract · {ev['champion_050_20']} · selected_on = holdout "
                                                       "(the brief's 'best model'; a model-ranking outcome, not an "
                                                       "independent estimate of deployment performance)"),
            ("interest_assumption", f"{art['interest_assumption']:.2f}"),
            ("score_type", "decisions compare the raw score with the threshold · shown as a probability: "
                           + ("yes — " + ("raw score, training-OOF ECE within tolerance" if art["score_type"].get("probability_source") == "raw"
                                          else f"{art['score_type'].get('probability_source')} calibration")
                              if art["score_type"].get("probability_available") else "no (ranking score only)")),
            ("training", f"{tr['n_train']:,} train / {tr['n_test']:,} test · test_size {tr['test_size']} · "
                         f"random_state {tr['random_state']} · {tr['cv_folds']}-fold CV"),
            ("versions", ", ".join(f"{k} {v}" for k, v in tr["versions"].items())),
        ], columns=["key", "value"])
        with st.container(key="panel_contract"):
            UI.html('<table class="ql-table" style="min-width:0">' + "".join(
                f'<tr><td style="width:210px"><code>{UI.esc(k)}</code></td>'
                f'<td style="white-space:normal;text-align:left;color:{P["t2"]}">{UI.esc(v)}</td></tr>'
                for k, v in contract.itertuples(index=False)) + "</table>")


# =========================================================================== fairness tab
def fairness_tab(ctx: Ctx) -> None:
    ev = ctx.ev
    fair = ev["fairness"]
    corr = ev["eda"]["corr"]["age"].drop(["age", "default"]).abs()
    UI.callout(
        "Each studied model was rebuilt <b>without age</b> (and without the age-derived engineered feature), re-tuned "
        "for its business threshold on training out-of-fold scores, refitted, and scored on the same holdout. Comparing "
        "the two shows what using age buys in money and how rejections shift across age bands. In this dataset age is "
        f"almost uncorrelated with the other inputs (largest |r| = {corr.max():.2f}, with {corr.idxmax()}), so removing it "
        "also removes most of its indirect influence — real portfolios often contain proxies, so this need not hold. "
        "A diagnostic, not a fairness certification. This study is fixed at 20% interest (it involves refitting "
        "models), so it does not follow the sidebar slider.", title="Does the model need age?", color=P["violet"])
    st.write("")
    names = list(fair)
    pick = st.segmented_control("Model", names, default=names[0], key="fair_model",
                                format_func=lambda n: f"{n} ◆ deployed" if n == ctx.deployed else f"{n} ★ leaderboard champion")
    pick = pick or names[0]
    f = fair[pick]
    t = f["table"]
    cols = ["variant", "policy", "threshold", "accuracy", "recall", "roc_auc", "fp", "money_saved"]
    view = t[cols].copy()
    view["variant"] = view["variant"].map({"with_age": "With age", "without_age": "Without age"})
    table(view, hide_index=True, width="stretch", column_config={
        "variant": "Variant", "policy": "Threshold policy",
        "threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
        "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
        "recall": st.column_config.NumberColumn("Recall", format="percent"),
        "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
        "fp": "Good rejected",
        "money_saved": st.column_config.NumberColumn("Money saved at 20% (AED)", format="%,.0f")})
    biz = t[t["policy"] == "CV business"].set_index("variant")
    b050 = t[t["policy"] == "0.50 benchmark"].set_index("variant")
    d_biz = biz.loc["with_age", "money_saved"] - biz.loc["without_age", "money_saved"]
    d_050 = b050.loc["with_age", "money_saved"] - b050.loc["without_age", "money_saved"]
    word = lambda d: (f"<b>{UI.aed(abs(d))}</b> {'more' if d > 0 else 'less'} with age" if abs(d) >= 1 else "no difference")
    UI.html(f'<div class="ql-muted" style="margin:8px 0 16px;color:{P["t2"]}">Holdout money saved: {word(d_biz)} at the CV '
            f'business threshold; {word(d_050)} at 0.50. Single-holdout differences of this size should be read against '
            'the bootstrap intervals on the leaderboard page.</div>')
    seg = f["segments"]
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(key="panel_fair_rej"):
            UI.chart_header("Rejection rate by age band",
                            "Share of applicants in each band the model would reject, with vs without age · each "
                            "variant at its own CV business threshold tuned at 20%")
            plot(CH.fairness_age_bars(seg[("with_age", "CV business")], seg[("without_age", "CV business")],
                                      "rejection_rate", "Rejection rate (%)"), "fair_rej")
    with c2:
        with st.container(key="panel_fair_frr"):
            UI.chart_header("Good customers rejected by age band",
                            "Share of repayers in each band turned away (false-rejection rate) · same thresholds as left")
            plot(CH.fairness_age_bars(seg[("with_age", "CV business")], seg[("without_age", "CV business")],
                                      "false_rejection_rate", "False-rejection rate (%)"), "fair_frr")
    sw, so = seg[("with_age", "CV business")], seg[("without_age", "CV business")]
    gap_w = sw["false_rejection_rate"].max() - sw["false_rejection_rate"].min()
    gap_o = so["false_rejection_rate"].max() - so["false_rejection_rate"].min()
    d_rej = (so["rejection_rate"].to_numpy() - sw["rejection_rate"].to_numpy()) * 100
    t_with = float(biz.loc["with_age", "threshold"]); t_wo = float(biz.loc["without_age", "threshold"])
    UI.html(f'<div class="ql-muted" style="margin-top:8px">Without age, rejection rates change by {d_rej.min():+.1f} to '
            f'{d_rej.max():+.1f} points across the bands. The business threshold moves from {t_with:.2f} to {t_wo:.2f}, '
            f'which shifts every band in the same direction — so compare the <i>pattern</i> across bands, not the levels. '
            f'Spread of false-rejection rates across bands: {gap_w*100:.1f} points with age vs {gap_o*100:.1f} without. '
            f'Bands hold {sw["n"].min():,}–{sw["n"].max():,} holdout loans and {sw["defaulters"].min()}–'
            f'{sw["defaulters"].max()} defaulters each, so differences of a few points are within sampling noise.</div>')
