"""Dynamic, metric-driven text. Nothing here is a fixed conclusion: every sentence is built from the
current board (interest assumption + threshold policy)."""
from __future__ import annotations

import numpy as np

from quickloan.config import INTERPRETABILITY

from .components import aed, esc, pct


def recommendation(b, deployed: str, art: dict, repay_share: float, uncertainty: bool = True) -> str:
    df = b.df.set_index("model")
    lead = b.champion
    L = df.loc[lead]
    acc_top = df["accuracy"].idxmax()
    A = df.loc[acc_top]
    second = b.df.iloc[1]
    interp, interp_why = INTERPRETABILITY[lead]
    n_def = int(L["tp"] + L["fn"])

    # most interpretable alternative and what it costs
    hi = [m for m in df.index if INTERPRETABILITY[m][0] == "High"]
    best_hi = max(hi, key=lambda m: df.loc[m, "money_saved"])
    gap_hi = L["money_saved"] - df.loc[best_hi, "money_saved"]

    p = []
    p.append(f"<b>{esc(lead)} leads the holdout</b> on current assumptions. At a foregone-interest assumption of "
             f"<b>{b.interest*100:.0f}%</b> it saves <b>{aed(L['money_saved'])}</b> on the 1,500-loan holdout versus "
             f"approving everyone — {aed(L['money_saved'] - second['money_saved'])} more than the runner-up, "
             f"{esc(second['model'])}.")
    if uncertainty and "ci_low" in df:
        ties = [m for m in df.index if m != lead and not df.loc[m, "distinguishable"]]
        p.append(f"<b>How sure is that lead?</b> On a paired bootstrap of the holdout (the same resampled loans for "
                 f"every model) the champion's 95% "
                 f"interval is {aed(L['ci_low'], compact=True)} – {aed(L['ci_high'], compact=True)} and it comes out on top "
                 f"in <b>{L['p_best']*100:.0f}%</b> of resamples. "
                 + (f"<b>{len(ties)}</b> other model{'s are' if len(ties) != 1 else ' is'} statistically indistinguishable from it "
                    f"(the 95% interval of the gap includes zero): {esc(', '.join(ties))}. Treat the top of the ranking as a "
                    f"group, not a single winner." if ties else
                    "Every other model is reliably behind it (the 95% interval of each gap excludes zero)."))
    p.append(f"<b>Why it leads:</b> it catches <b>{int(L['tp'])} of {n_def}</b> defaulters (recall {pct(L['recall'])}), "
             f"keeping {aed(L['loans_saved'], compact=True)} of principal out of default, while rejecting "
             f"{int(L['fp'])} good customers at an interest cost of {aed(L['interest_lost'], compact=True)}. "
             f"Accuracy is {pct(L['accuracy'], 2)} and precision {pct(L['precision'])}.")
    if acc_top != lead:
        gap = aed(L["money_saved"] - A["money_saved"])
        if int(A["tp"]) < int(L["tp"]):
            why = f"because it catches {int(A['tp'])} defaulters instead of {int(L['tp'])}"
        elif int(A["tp"]) == int(L["tp"]):
            why = (f"because, while it catches the same {int(L['tp'])} defaulters by count, they are smaller loans "
                   f"({aed(A['loans_saved'], compact=True)} of principal saved vs {aed(L['loans_saved'], compact=True)})")
        else:
            why = (f"even though it catches {int(A['tp'])} defaulters, because the ones it misses are larger loans "
                   f"or it rejects costlier good customers")
        p.append(f"<b>Accuracy would have picked the wrong model.</b> {esc(acc_top)} has the highest accuracy "
                 f"({pct(A['accuracy'], 2)}) but saves {aed(A['money_saved'])} — {gap} less, {why}. "
                 f"Accuracy counts every loan equally; money saved weights each decision by its loan size. With "
                 f"{pct(repay_share, 1)} of holdout loans repaid, accuracy mostly measures how well a model approves good "
                 f"customers, not how well it stops losses.")
    else:
        p.append(f"<b>Here the most accurate model is also the most profitable</b> — but accuracy across all ten models "
                 f"spans only {(df['accuracy'].max()-df['accuracy'].min())*100:.1f} points while money saved spans "
                 f"{aed(df['money_saved'].max()-df['money_saved'].min())}, so accuracy alone could not have told them apart.")
    p.append(f"<b>Trade-off accepted:</b> interpretability is <b>{interp.lower()}</b> ({interp_why}). "
             + (f"The best fully transparent option, {esc(best_hi)}, would forgo {aed(gap_hi)} of savings — "
                f"the price of explainability on this holdout." if best_hi != lead else
                "It is also the most transparent leading option."))
    p.append("<b>Limitations:</b> one 1,500-loan holdout (only 188 defaulters, so a handful of loans moves the ranking); "
             "full loss assumed on default with no recoveries, fees or funding costs; only the deployed model's score is "
             "presented as a probability (its calibration was checked on training data); the fairness check is diagnostic "
             "and regulatory validation has not been performed.")
    return "<br><br>".join(p)


def champion_ranges(sens) -> list[tuple[float, float, str]]:
    """Contiguous interest ranges for which each model is champion."""
    best = sens.sort_values(["interest", "money_saved", "model"], ascending=[True, False, True]) \
               .groupby("interest").head(1)[["interest", "model"]].to_numpy()
    out, start, cur = [], best[0][0], best[0][1]
    for i in range(1, len(best)):
        if best[i][1] != cur:
            out.append((start, best[i - 1][0], cur))
            start, cur = best[i][0], best[i][1]
    out.append((start, best[-1][0], cur))
    return out


def risk_interpretation(score: float, threshold: float, warnings: list[str]) -> str:
    margin = score - threshold
    if margin >= 0.15:
        s = "Score is well above the rejection threshold — a clear high-risk profile."
    elif margin >= 0:
        s = "Score is just above the threshold — a borderline rejection; small input changes could flip it."
    elif margin > -0.08:
        s = "Score is just below the threshold — a borderline approval worth a manual review."
    else:
        s = "Score is comfortably below the threshold — a low-risk profile for this model."
    if warnings:
        s += " Some inputs are outside the training data, so treat this result with extra caution."
    return s


def spread_numbers(b):
    acc = b.df["accuracy"]
    money = b.df["money_saved"]
    return float(acc.max() - acc.min()), float(money.max() - money.min())


def rank_corr(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
