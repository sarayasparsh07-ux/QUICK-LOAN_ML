"""Generate QuickLoan_Model_Bakeoff.ipynb (then execute it with nbclient / jupyter nbconvert --execute)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

# ----------------------------------------------------------------------------------------------- 1-2
md("""
# 💰 QuickLoan — The Great Model Bake-Off
### Credit-risk modelling, ensemble learning and money-based model selection

**1. Introduction.** QuickLoan is a (fictional) UAE fintech issuing small personal loans. Every application ends in
*approve* or *reject*. This notebook builds ten classifiers — three single models and seven ensembles covering
**Voting, Bagging, Boosting and Stacking** — scores them all on one common holdout, and recommends which one to deploy
**in money terms, not accuracy terms**.

**2. Business objective.** Mistakes are asymmetric:

| | Model approves (0) | Model rejects (1) |
|---|---|---|
| **Actually repays** | +20 % × loan (interest earned) | 0 (interest lost) |
| **Actually defaults** | −100 % × loan (principal lost) | 0 (loss avoided) |

*Money saved* = how much better a model does than approving everyone
= principal of caught defaulters − 20 % × principal of rejected good customers.

**Workflow and leakage rules.** All modelling code lives in the `quickloan` package and is shared with
`train_models.py`, so the notebook and the script produce identical artifacts. The test set (25 %, `random_state=1`,
stratified) is opened **once**, in Section 17. Feature engineering, class weighting, hyperparameters, thresholds,
feature importance, calibration and the deployment choice are all decided on the 4,500 training loans with 5-fold
stratified CV.

**Beyond the brief's core:** an untuned weak-learner baseline (Step 2), bootstrap confidence intervals on every
model's money saved, business thresholds re-optimised for any interest assumption, a with/without-age fairness check,
a calibrated deployment model, and exact reason codes for individual decisions (Sections 19–20).
""")
# ---- headline figures: written into the summary AND asserted against computed results in the final cell
HEADLINE = {"champion": "Stacking", "champion_money": 4_891_400, "acc_spread_pts": 0.9, "money_spread": 2_256_440,
            "p_best_pct": 44, "n_tied": 4, "deployed": "Gradient Boosting", "deployed_threshold": 0.16,
            "deployed_money": 5_943_420, "deployed_recall_pct": 54.3, "age_gain": 28_800,
            "nested_per_1000": 4_133_827, "nested_sd_per_1000": 295_127, "nested_deployed_folds": 3}
H = HEADLINE
md(f"""
## Executive summary
1. **Accuracy hides the money.** All ten models fall within **{H['acc_spread_pts']} accuracy points**, yet money saved
   ranges over **AED {H['money_spread']:,}** on the same 1,500 test loans.
2. **Leaderboard champion: {H['champion']}**, saving **AED {H['champion_money']:,}** versus approving everyone
   (0.50 threshold, 20 % interest) — but it wins only {H['p_best_pct']} % of bootstrap resamples and {H['n_tied']} other
   models are statistically tied with it ([§19c](#s19c)). It was selected using the common holdout benchmark for
   assignment comparison, so its figure is a model-ranking outcome, not an independent estimate of deployment
   performance.
3. **Recommended deployment: {H['deployed']} at a {H['deployed_threshold']:.2f} threshold**, chosen on training
   cross-validation before the test set was opened. It saves **AED {H['deployed_money']:,}** on the test set
   (recall {H['deployed_recall_pct']} %) — the threshold is the biggest money lever ([§17a](#s17), [§21](#s21)).
   Items 2 and 3 are different evaluations — **A**, all models at a common 0.50 threshold, and **C**, the deployment
   policy chosen on training data — and both use the same money-saved formula ([§17](#s17)).
4. **The selection procedure itself is validated.** Nested cross-validation (5 outer × 5 inner folds, training data
   only — evaluation **B**) re-ran the whole selection: it chose {H['deployed']} in {H['nested_deployed_folds']} of 5
   folds and earned **AED {H['nested_per_1000']:,} per 1,000 loans** (sd {H['nested_sd_per_1000']:,}) on rows it never
   saw; no alternative was materially better ([§17c](#s17c)).
5. **Age is not needed:** without it the deployed model saves AED {H['age_gain']:,} *more*, within noise ([§20d](#s20d)).
6. **Every decision is explainable** with exact reason codes, and the deployed model's score is well calibrated
   ([§20e](#s20e), [§20f](#s20f)). The final cell re-checks every figure in this summary.
""")
md("""
## Rubric map (30 marks + 3 stretch)
| # | Criterion | Marks | Where to find it |
|---|---|---:|---|
| 1 | Clean load, encode, train/test split (no leakage) | 3 | [§3 Load](#s3) · [§4 Validation](#s4) · [§8 Split](#s8) · [§7 Encoding pipeline](#s7) |
| 2 | Weak single models built + evaluated | 4 | [§9a–9c](#s9a) · [§17d metrics](#s17d) · [§19a untuned weak learner](#s19a) |
| 3 | Voting correctly built | 3 | [§10](#s10) |
| 4 | Bagging / Random Forest correctly built | 3 | [§11](#s11) · [§12](#s12) |
| 5 | Boosting (incl. XGBoost) correctly built | 4 | [§13 AdaBoost](#s13) · [§14 Gradient Boosting](#s14) · [§15 XGBoost](#s15) |
| 6 | Stacking correctly built | 4 | [§16](#s16) |
| 7 | Leaderboard with accuracy, recall **and** money saved | 4 | [§19](#s19) · [§19b charts](#s19b) · [§19c confidence intervals](#s19c) |
| 8 | Verdict on business grounds, accuracy-vs-money insight | 3 | [§21](#s21) |
| 9 | Code quality & interpretation | 2 | `quickloan/` package + 200 automated tests · a "Reading it" note under every result |
| ★ | Stretch: dashboard with decision + money cards | +3 | `app.py` · `loan_model.joblib` (also runs in the instructor's starter `app.py`) · [§22](#s22) |

The order is the brief's Steps 1–5. The only deliberate reordering: the split ([§8](#s8)) comes *before* feature
engineering ([§6](#s6)), so no modelling decision can see the test set.
""")
code("""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from IPython.display import display, Markdown
from quickloan import config as C, data as D, features as FE, finance as F, models as M, evaluation as E, training as T, artifacts as A

P = C.PALETTE
mpl.rcParams.update({
    "figure.facecolor": P["bg1"], "axes.facecolor": P["bg2"], "savefig.facecolor": P["bg1"],
    "axes.edgecolor": "#2A3A55", "axes.labelcolor": P["t2"], "axes.titlecolor": P["t1"], "text.color": P["t2"],
    "xtick.color": P["t2"], "ytick.color": P["t2"], "grid.color": "#22324B", "axes.grid": True, "grid.alpha": .6,
    "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.labelsize": 11, "legend.frameon": False,
    "font.family": "DejaVu Sans", "figure.dpi": 110,
})
GOOD, BAD = P["emerald"], P["neg"]
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
pd.set_option("display.width", 160)
print("XGBoost available:", M.HAS_XGB, "| model label:", M.XGB_LABEL)
""")

# ----------------------------------------------------------------------------------------------- 3
md("## 3. Data inspection")
code("""
df = D.load_loans()            # resolves loans.csv / loans(1).csv and runs hard validation
print("Shape:", df.shape)
display(df.head())
display(df.dtypes.to_frame("dtype").T)
print("Target counts:", df[C.TARGET].value_counts().to_dict(), f"| default rate {df[C.TARGET].mean():.2%}")
""")

# ----------------------------------------------------------------------------------------------- 4
md("""
## 4. Data-quality validation
Hard checks (schema, missing values, binary target, positive amounts) are enforced by `D.validate`; the audit below
adds duplicates, ranges, percentiles, IQR outliers and logical-consistency checks.
""")
code("""
audit = D.quality_audit(df)
print(f"Missing values: {audit['missing']} | duplicate rows: {audit['duplicates']}")
print("home_ownership levels:", audit["categories"])
display(pd.Series(audit["sanity"], name="passes").to_frame())
display(audit["describe"].round(2))
display(pd.Series(audit["iqr_outliers"], name="IQR outliers (1.5×)").to_frame().T)
ie = audit["implausible_employment"]
print(f"Plausibility flag - employment_years > age - 16: {ie['n']} rows ({ie['share']:.1%}), "
      f"default rate {ie['default_rate']:.1%}, oldest such applicant {ie['max_age']}. Decision: {ie['decision']}")
display(df.loc[df.employment_years > df.age - 16, ["age", "employment_years"]].assign(age_band=lambda d: pd.cut(d.age, [20, 25, 29, 39, 64])).age_band.value_counts().sort_index().to_frame("flagged rows"))
""")
md("""
**Reading the audit.** No missing values or duplicates; amounts are positive, the target is binary, loan terms are
one of 12/24/36/48/60 and counts are non-negative. IQR "outliers" are the long right tails of income and loan size —
legitimate large borrowers, not errors.

**One plausibility flag:** some applicants report more years employed than years since age 16 (see below — almost all
aged 21–29, e.g. a 21-year-old with 16 years of employment). The data is synthetic and there is no evidence which field
is wrong, so these rows are **kept and flagged, not removed**: deleting them would be an unjustified intervention and
would also change the assignment's reference split. The simulator warns when a user enters such a combination.

### 4.2 The `debt_to_income` inconsistency
""")
code("""
mm = audit["dti_mismatch"]
print(f"{len(mm)} rows where stored debt_to_income != round(loan_amount / annual_income, 2)")
display(mm)
print(mm["reason"].value_counts())
""")
md("""
**Decision.** Eight values are capped at the dataset maximum 3.00 and six are half-way ratios (x.xx5) stored rounded
down. Both are recording conventions, not corrupt data, so **training uses the values exactly as recorded**. The
dashboard's simulator re-creates the same quantity as a documented *loan-to-income proxy*
(`loan_amount / annual_income`, rounded to 2 dp, capped at 3.00). It is **not** a conventional debt-to-income ratio,
which would use total monthly debt service.
""")

# ----------------------------------------------------------------------------------------------- 5
md("## 5. Exploratory data analysis")
code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
vc = df[C.TARGET].value_counts().sort_index()
ax[0].bar(["Repaid (0)", "Defaulted (1)"], vc.values, color=[GOOD, BAD])
for i, v in enumerate(vc.values): ax[0].text(i, v + 60, f"{v:,}\\n({v/len(df):.1%})", ha="center", color=P["t1"])
ax[0].set(title="Target: default vs repayment", ylabel="Number of loans", ylim=(0, vc.max() * 1.2))
ho = df.groupby("home_ownership")[C.TARGET].agg(["size", "mean"]).sort_values("mean")
ax[1].bar(ho.index, ho["mean"], color=[P["blue"], P["violet"], P["amber"]])
for i, (n, r) in enumerate(zip(ho["size"], ho["mean"])): ax[1].text(i, r + .004, f"{r:.1%}\\nn={n:,}", ha="center", color=P["t1"])
ax[1].axhline(df[C.TARGET].mean(), ls=":", color=P["t3"]); ax[1].set(title="Default rate by home ownership", ylabel="Default rate", ylim=(0, .19))
ax[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1)); plt.tight_layout(); plt.show()
""")
code("""
cols = [("annual_income", "Annual income (AED)"), ("loan_amount", "Loan amount (AED)"), ("age", "Age (years)"),
        ("employment_years", "Employment (years)"), ("credit_history_months", "Credit history (months)"),
        ("debt_to_income", "Loan-to-income (debt_to_income)")]
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for a, (c, lab) in zip(axes.ravel(), cols):
    bins = np.histogram_bin_edges(df[c], bins=35)
    a.hist(df.loc[df[C.TARGET] == 0, c], bins=bins, density=True, alpha=.55, color=GOOD, label="Repaid")
    a.hist(df.loc[df[C.TARGET] == 1, c], bins=bins, density=True, alpha=.55, color=BAD, label="Defaulted")
    a.set(title=lab, xlabel=lab, ylabel="Density")
axes[0, 0].legend(); plt.suptitle("Feature distributions by outcome (density, so the 12.5% minority is visible)", color=P["t1"]); plt.tight_layout(); plt.show()
""")
code("""
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for a, (c, lab) in zip(axes, [("annual_income", "Annual income (AED)"), ("loan_amount", "Loan amount (AED)"),
                              ("debt_to_income", "Loan-to-income"), ("employment_years", "Employment (years)")]):
    bp = a.boxplot([df.loc[df[C.TARGET] == 0, c], df.loc[df[C.TARGET] == 1, c]], tick_labels=["Repaid", "Defaulted"],
                   patch_artist=True, widths=.55, medianprops=dict(color=P["t1"]), flierprops=dict(markersize=2, alpha=.3, markeredgecolor=P["t3"]))
    for patch, col in zip(bp["boxes"], [GOOD, BAD]): patch.set(facecolor=col, alpha=.55, edgecolor=col)
    a.set(title=lab, ylabel=lab)
plt.tight_layout(); plt.show()
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for a, c, lab in [(axes[0], "past_defaults", "Past defaults"), (axes[1], "num_existing_loans", "Existing loans"),
                  (axes[2], "loan_term_months", "Loan term (months)")]:
    g = df.groupby(c)[C.TARGET].agg(["size", "mean"])
    a2 = a.twinx(); a2.grid(False)
    a.bar(g.index.astype(str), g["size"], color=P["blue"], alpha=.45, label="Loans")
    a2.plot(g.index.astype(str), g["mean"], "o-", color=BAD, lw=2, label="Default rate")
    for x, (n, r) in zip(g.index.astype(str), zip(g["size"], g["mean"])): a2.annotate(f"{r:.0%}", (x, r), textcoords="offset points", xytext=(0, 7), ha="center", color=P["t1"], fontsize=9)
    a.set(title=f"{lab}: volume and default rate", xlabel=lab, ylabel="Number of loans"); a2.set_ylabel("Default rate", color=BAD)
    a2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1)); a2.set_ylim(0, max(g["mean"].max() * 1.25, .2))
plt.tight_layout(); plt.show()
print(df.groupby("past_defaults")[C.TARGET].agg(["size", "mean"]).T)
""")
code("""
d = df.assign(lti_band=pd.cut(df["debt_to_income"], [0, .25, .5, .75, 1, 1.5, 3.01], labels=["≤0.25", ".25–.5", ".5–.75", ".75–1", "1–1.5", ">1.5"]),
              pd_band=df["past_defaults"].clip(upper=2).map({0: "0", 1: "1", 2: "2+"}))
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
s = df.sample(2500, random_state=1)
ax[0].scatter(s["annual_income"], s["loan_amount"], c=np.where(s[C.TARGET] == 1, BAD, GOOD), s=9, alpha=.55)
ax[0].set(title="Income vs loan size (sample of 2,500; coral = defaulted)", xlabel="Annual income (AED)", ylabel="Loan amount (AED)")
pv = d.pivot_table(index="pd_band", columns="lti_band", values=C.TARGET, aggfunc="mean", observed=True)
im = ax[1].imshow(pv.values, cmap=mpl.colors.LinearSegmentedColormap.from_list("q", [P["bg3"], P["amber"], BAD]), aspect="auto", vmin=0, vmax=.7)
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]):
        n = ((d["pd_band"] == pv.index[i]) & (d["lti_band"] == pv.columns[j])).sum()
        ax[1].text(j, i, f"{pv.values[i, j]:.0%}\\nn={n}", ha="center", va="center", color=P["t1"], fontsize=9)
ax[1].set(xticks=range(pv.shape[1]), xticklabels=pv.columns, yticks=range(pv.shape[0]), yticklabels=pv.index,
          title="Interaction: default rate by past defaults × loan-to-income", xlabel="Loan-to-income band", ylabel="Past defaults")
ax[1].grid(False); plt.colorbar(im, ax=ax[1], format=mpl.ticker.PercentFormatter(1)); plt.tight_layout(); plt.show()
""")
code("""
corr = df.drop(columns="home_ownership").corr()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap=mpl.colors.LinearSegmentedColormap.from_list("d", [BAD, P["bg2"], P["teal"]]), vmin=-1, vmax=1)
ax.set(xticks=range(len(corr)), yticks=range(len(corr)), title="Correlation matrix (Pearson)")
ax.set_xticklabels(corr.columns, rotation=40, ha="right"); ax.set_yticklabels(corr.index); ax.grid(False)
for i in range(len(corr)):
    for j in range(len(corr)): ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=P["t1"])
plt.colorbar(im, ax=ax, fraction=.046); plt.tight_layout(); plt.show()
display(corr[C.TARGET].drop(C.TARGET).sort_values().to_frame("corr with default"))
""")
code("""
# Segmented financial view: where the principal at risk sits
seg = d.groupby("lti_band", observed=True).agg(loans=("loan_amount", "size"), principal=("loan_amount", "sum"),
          default_principal=("loan_amount", lambda s: s[d.loc[s.index, C.TARGET] == 1].sum()))
seg["share_of_all_default_principal"] = seg["default_principal"] / seg["default_principal"].sum()
seg["loss_rate_on_principal"] = seg["default_principal"] / seg["principal"]
display(seg.style.format({"principal": "AED {:,.0f}", "default_principal": "AED {:,.0f}",
        "share_of_all_default_principal": "{:.1%}", "loss_rate_on_principal": "{:.1%}"}))
""")
md("""
**EDA conclusions.**
* 12.55 % of loans default — a clear minority class, so accuracy is dominated by repayers.
* **Past defaults** is the strongest single signal: any prior default roughly triples the default rate (≈8 % → ≈27 %).
* **Loan-to-income** (`debt_to_income`) has the highest linear correlation with default (≈0.35); income is protective
  and larger loans riskier. High loan-to-income bands carry a disproportionate share of defaulted principal, which is
  exactly what the money metric rewards catching.
* Renters default more than owners and mortgagors; age, term and credit-history length are weak on their own.
* The interaction heatmap shows risk compounding: past defaults **and** a high loan-to-income ratio together.
""")

# ----------------------------------------------------------------------------------------------- 8 (split before FE)
md("""
## 8. Train/test split *(done before feature engineering so no decision can see the test set)*
Assignment settings: 25 % test, `random_state=1`, stratified on `default`. Test loan amounts are taken from `X_test`
itself, so they stay aligned with every prediction by construction.
""")
code("""
split = D.make_split(df)
check = D.reconcile_split(split)
display(pd.DataFrame(check).T)
assert all(v["match"] for v in check.values()), "Split does not reconcile with the instructor reference"
print(f"Train {len(split.y_train):,} ({split.y_train.mean():.2%} default) | Test {len(split.y_test):,} ({split.y_test.mean():.2%} default)")
""")
md("Every reference value (1,500 test loans, 188 defaulters, AED 12,271,900 defaulted principal, AED 62,945,000 repaid principal) reconciles exactly.")

# ----------------------------------------------------------------------------------------------- 6
md("""
## 6. Feature engineering
Candidates are deterministic row-wise functions of the applicant's own inputs (no fitting, no target information).
They run inside the model pipeline, so the saved artifact computes them identically at prediction time.
A principal burden is **not** an EMI: the data has no interest rate or amortisation schedule.
""")
code("""
display(pd.DataFrame([(k, f, m) for k, (f, m) in FE.CANDIDATES.items()], columns=["feature", "formula", "financial meaning"]))
eng = FE.add_features(split.X_train, list(FE.CANDIDATES))
new = list(FE.CANDIDATES)
display(eng[new].describe().T.round(3))
red = eng[new + ["loan_amount", "annual_income", "debt_to_income", "credit_history_months", "past_defaults", "employment_years"]].corr().loc[new]
display(red.round(2).style.background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1).set_caption("Redundancy with existing variables (training set)"))
""")
md("""
**Validation protocol.** Each feature is added *on its own* to a scaled logistic regression and to gradient boosting
and scored by 5-fold CV average precision (PR-AUC) on the training set. A feature is kept only if it lifts mean PR-AUC
by more than **0.002**; `credit_history_years` is a pure rescaling and acts as a control for the noise floor. The
retained set is then re-checked in combination.
""")
code("""
fe_table, FEATURES = T.feature_engineering_study(split, verbose=False)
display(fe_table.style.format({c: "{:.4f}" for c in ["LR_pr_auc", "GB_pr_auc", "LR_roc_auc", "GB_roc_auc"]} | {"delta_mean_pr_auc": "{:+.4f}"}))
print("Retained engineered features:", FEATURES)
""")
md("""
**Outcome.** `has_past_default` gives the largest lift (prior default is a step change, not a linear count effect) and
`employment_share_of_adult_life` adds a career-stability signal; together they pass the combined test and are kept.
The principal-burden ratios, log income and the rescaled control fall inside the noise band and are **rejected**:
they are largely redundant with `debt_to_income`, `loan_amount` and `annual_income`.
""")

# ----------------------------------------------------------------------------------------------- 7
md("""
## 7. Preprocessing architecture
`FeatureEngineer → ColumnTransformer(OneHotEncoder(handle_unknown="ignore") on home_ownership + StandardScaler or passthrough) → estimator`.
Scaling is applied for Logistic Regression and KNN (distance- and penalty-based); trees get raw values. Each pipeline
is fitted on training data only — inside CV this happens fold by fold — and an unseen home-ownership value is encoded as
all zeros instead of crashing.
""")
code("""
pipe = M.make_pipeline("Logistic Regression", FEATURES)
display(pipe)
pre = pipe[:-1].fit(split.X_train, split.y_train)
print("Model input columns:", list(pre.get_feature_names_out()))
""")

# ----------------------------------------------------------------------------------------------- 9-16
md("""
## 9–16. Model development
Each directly tuned model gets a **small, transparent grid** searched with 5-fold stratified CV, scored by average
precision. Class weights stay at their default here; their effect is studied separately in Section 17.
""")
code("""
TUNED, PARAMS, ROWS, GRIDS = {}, {}, [], {}
def tune(name):
    est, prm, row = T.tune_one(name, split, FEATURES, verbose=False)
    GRIDS[name] = row.pop("_grid"); TUNED[name], PARAMS[name] = est, prm; ROWS.append(row)
    print(f"{name}: best CV PR-AUC {row['cv_pr_auc']:.4f} from {row['n_candidates']} candidates in {row['seconds']}s")
    print("  selected:", {k.replace('clf__', ''): v for k, v in prm.items()})
    display(GRIDS[name].head(5))
""")
for title, name, text in [
    ("### 9a. Decision Tree (single)", "Decision Tree",
     "Complexity controls: `max_depth`, `min_samples_leaf`, `min_samples_split`. **Interpretable** — a short list of "
     "if-then rules — but high-variance: small data changes can reshape the tree. Tuning makes this tree much stronger "
     "than the brief's 'weak' learner, so Section 19 also reports an **untuned depth-3 tree** as the weak-learner "
     "baseline (shown beside, not inside, the ten-model leaderboard)."),
    ("### 9b. Logistic Regression (single)", "Logistic Regression",
     "Scaled inputs, L2 penalty tuned through `C`. Coefficients are directly explainable, but it can only draw a linear "
     "boundary in the (engineered) feature space, so interactions such as past-default × loan-to-income are missed."),
    ("### 9c. KNN (single)", "KNN",
     "Distances are meaningless unless features share a scale — income in AED would otherwise swamp everything — so "
     "inputs are standardised. Searched: `n_neighbors`, `weights`, distance metric `p` (1 = Manhattan, 2 = Euclidean)."),
]:
    md(f"{title}\n{text}")
    code(f'tune("{name}")')
md("""
### 10. Voting (ensemble)
A **soft** vote averages the predicted default probabilities of the three tuned single models, so a confident model
counts for more than an unsure one. Mixing a tree, a linear model and a neighbourhood model gives genuinely different
errors to average out.
""")
code("""
TUNED["Voting"] = M.make_voting(TUNED); PARAMS["Voting"] = T.ENSEMBLE_PARAMS["Voting"]
TUNED["Voting"]
""")
for title, name, text in [
    ("### 11. Bagging (ensemble)", "Bagging",
     "200 decision trees, each trained on a bootstrap sample of the rows; their probabilities are averaged. Deep trees "
     "have low bias but high variance — averaging many de-synchronised copies cancels much of that variance."),
    ("### 12. Random Forest (ensemble, bagging family)", "Random Forest",
     "Bagging plus a random subset of features at every split (`max_features`), which de-correlates the trees further "
     "so the average reduces variance more than plain bagging can."),
    ("### 13. AdaBoost (ensemble, boosting)", "AdaBoost",
     "Shallow trees trained **in sequence**; after each round the loans it got wrong are up-weighted, so the next tree "
     "concentrates on them. Boosting attacks bias — many weak learners become a strong one."),
    ("### 14. Gradient Boosting (ensemble, boosting)", "Gradient Boosting",
     "Each new tree fits the residual errors (gradient of the log-loss) of the ensemble so far; `learning_rate` shrinks "
     "each step and `subsample=0.8` adds stochastic regularisation."),
    ("### 15. XGBoost (ensemble, boosting)", "XGBoost",
     "Regularised gradient boosting (`reg_lambda`), histogram tree method, row and column subsampling (0.8/0.8), "
     "`eval_metric='logloss'`, fixed `random_state` for reproducibility. If XGBoost were unavailable the code would fall "
     "back to `HistGradientBoostingClassifier` and label it as a substitute, never as XGBoost."),
]:
    md(f"{title}\n{text}")
    code(f'tune("{name}")')
md("""
### 16. Stacking (ensemble)
Strong, diverse bases — tuned **Random Forest, XGBoost and Logistic Regression** — feed their default probabilities to
a logistic-regression **manager**. With `cv=5` the manager is trained only on *out-of-fold* base predictions, so it
learns how far to trust each base model on data that base model never saw, preventing leakage.
""")
code("""
TUNED["Stacking"] = M.make_stacking(TUNED); PARAMS["Stacking"] = T.ENSEMBLE_PARAMS["Stacking"]
TUNED = {n: TUNED[n] for n in C.MODEL_ORDER}
tuning_table = pd.DataFrame(ROWS)
display(tuning_table)
TUNED["Stacking"]
""")

# ----------------------------------------------------------------------------------------------- 17
md("""
## 17. Model evaluation
Three evaluations answer three different questions and are never mixed:

| | Evaluation | Data | Threshold | Question it answers |
|---|---|---|---|---|
| **A** | Benchmark leaderboard ([§19](#s19)) | 1,500-loan holdout | 0.50 for every model | Which model ranks best like-for-like? (A model-ranking outcome, not an independent estimate of deployment performance.) |
| **B** | Nested cross-validation ([§17c](#s17c)) | 4,500 training loans only | chosen inside each outer fold | How good and how stable is the *selection procedure*? |
| **C** | Final deployment holdout ([§17d](#s17d)) | 1,500-loan holdout, opened once | training-CV business threshold | What does the model we would deploy actually earn? |

All three use the official money-saved formula.

### 17a. Training-set cross-validation (decides thresholds and deployment)
Out-of-fold probabilities for all ten models give honest training-set estimates. They are also used to (i) study class
weighting and (ii) pick each model's **business threshold** — the cut-off that maximises out-of-fold money saved at
20 %. The 0.50 benchmark stays the primary comparison.
""")
code("""
cw_table = T.class_weight_study(split, TUNED, verbose=False)
display(cw_table.style.format({"cv_accuracy": "{:.2%}", "cv_recall": "{:.2%}", "cv_precision": "{:.2%}", "cv_pr_auc": "{:.4f}", "cv_money_saved": "AED {:,.0f}"}))
""")
md("""
**Class weighting.** `class_weight="balanced"` (or `scale_pos_weight` for XGBoost) pushes recall up sharply at the
0.50 cut-off by rejecting far more applicants. It is mathematically similar to lowering the threshold, but distorts
the scores and changes every model's operating point differently, which would make the 0.50 comparison
apples-to-oranges. The build therefore keeps unweighted models and moves the threshold **explicitly** (17a → business
thresholds), which keeps the benchmark clean and the trade-off visible.
""")
code("""
OOF = T.oof_scores(split, TUNED, verbose=False)
cvs = T.cv_summary(split, OOF)
display(cvs.style.format({"cv_accuracy": "{:.2%}", "cv_recall": "{:.2%}", "cv_roc_auc": "{:.3f}", "cv_pr_auc": "{:.3f}", "cv_brier": "{:.4f}",
        "cv_money_saved_050": "AED {:,.0f}", "business_threshold": "{:.2f}", "cv_money_saved_business": "AED {:,.0f}"}))
DEPLOYED = T.choose_deployment(cvs)                  # the one rule shared with train_models.py and nested CV
print("Deployment choice (training CV only):", DEPLOYED)
""")
md("""
Note the business thresholds sit far below 0.50. That is expected: with this payoff it pays to reject when
P(default) × loan > P(repay) × 0.20 × loan, i.e. when P(default) > 0.20 / 1.20 ≈ 0.17 for a calibrated score.

**The optimal threshold depends on the interest assumption.** The dashboard's interest slider runs from 5 % to 50 %,
so instead of one threshold tuned at 20 % every model gets a threshold *curve*: for each candidate cut-off, the
out-of-fold principal of caught defaulters and of rejected repayers. Money saved at any interest is then
`caught − interest × rejected`, the exact same sums as the official function, so re-optimising is instant and still
uses training data only. The assertion inside `threshold_curves` checks the fast path reproduces the official search.
""")
code("""
CURVES = T.threshold_curves(split, OOF, cvs)          # asserts equality with the official search at 20 %
grid_i = np.round(np.arange(0.05, 0.501, 0.01), 2)
thr_tab = pd.DataFrame({n: [F.threshold_at(c, i) for i in grid_i] for n, c in CURVES.items()}, index=grid_i)
fig, ax = plt.subplots(figsize=(12, 4.6))
for n in thr_tab:
    ax.step(thr_tab.index, thr_tab[n], where="post", lw=2.6 if n == DEPLOYED else 1.1, alpha=1 if n == DEPLOYED else .55,
            color=P["cyan"] if n == DEPLOYED else C.FAMILY_COLOR[C.FAMILY[n]], label=n)
ax.plot(grid_i, [F.break_even_threshold(i) for i in grid_i], ":", color=P["t1"], label="break-even i/(1+i)")
ax.set(title="Money-maximising threshold vs interest assumption (training out-of-fold)", xlabel="Foregone interest (% of principal)",
       ylabel="Threshold"); ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.legend(ncol=2, fontsize=8.5, bbox_to_anchor=(1.01, 1), loc="upper left"); plt.tight_layout(); plt.show()
display(thr_tab.loc[[0.05, 0.10, 0.20, 0.30, 0.40, 0.50]].T.rename(columns=lambda c: f"{c:.0%}").style.format("{:.2f}")
        .set_caption("Business threshold by interest assumption"))
""")
md("""
Thresholds rise with the interest assumption — rejecting a good customer costs more, so a model should only reject
when it is more confident — and they track the break-even line, which is what a sensible money-maximising rule should do.

### 17b. How stable is the deployment threshold?
Still training data only: the deployed model's out-of-fold money, recall, precision and false-rejection rate across
every candidate cut-off, the band of thresholds within 1 % / 2 % of the maximum, and the cut-off each of the five
inner folds would prefer on its own.
""")
code("""
TD = T.threshold_diagnostics(split, OOF[DEPLOYED], float(cvs.set_index("model").loc[DEPLOYED, "business_threshold"]))
cur = TD["curve"]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
ax[0].plot(cur.threshold, cur.money_saved / 1e6, color=P["teal"], lw=2.2)
ax[0].axvspan(*TD["band_99"], color=P["teal"], alpha=.15, label="within 1 % of max")
ax[0].axvline(TD["chosen"], color=P["amber"], ls="--", label=f"chosen {TD['chosen']:.2f}")
ax[0].axvline(0.5, color=P["t3"], ls=":", label="0.50 benchmark")
ax[0].set(title=f"{DEPLOYED}: out-of-fold money saved by threshold", xlabel="Threshold", ylabel="AED million"); ax[0].legend()
for col, c, lab in [("recall", GOOD, "recall"), ("precision", P["blue"], "precision"), ("false_rejection_rate", BAD, "good customers rejected")]:
    ax[1].plot(cur.threshold, cur[col], color=c, lw=2, label=lab)
ax[1].axvline(TD["chosen"], color=P["amber"], ls="--")
ax[1].set(title="Trade-offs around the chosen cut-off", xlabel="Threshold", ylabel="Rate"); ax[1].legend()
ax[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1)); plt.tight_layout(); plt.show()
print(f"Chosen {TD['chosen']:.2f}; 99 % band {TD['band_99'][0]:.2f}–{TD['band_99'][1]:.2f}; 98 % band {TD['band_98'][0]:.2f}–{TD['band_98'][1]:.2f}")
display(TD["per_fold"].style.format({"best_threshold": "{:.2f}", "money_at_best": "AED {:,.0f}", "money_at_chosen": "AED {:,.0f}"}).hide(axis="index"))
share_050 = cur.set_index("threshold").loc[0.50, "money_saved"] / TD["max_money"]
print(f"At 0.50 the same out-of-fold scores keep only {share_050:.0%} of the maximum money saved.")
""")
md("""
**Reading it.** The peak is sharp at the top (the 1 % band is narrow) but the curve stays within about 2 % of the
maximum over a wider band of cut-offs, and every inner fold prefers a threshold far below 0.50. The decision that
matters is moving away from 0.50; the second decimal moves the result by a few percent at most.

### 17c. Nested cross-validation — evaluation B
Section 17a picks the deployment model *and* its threshold on the same out-of-fold scores, so its winning figure is
optimistic. Nested CV measures the **whole procedure** without that optimism: the 4,500 training loans are split into
5 outer folds (`StratifiedKFold`, shuffled, seed 2). For each outer fold, `T.select_deployment` — the same function
that made the choice above — runs on the outer-training part only: feature selection, tuning, inner 5-fold
out-of-fold scores (seed 1), business thresholds and the model choice. The chosen pipeline is refitted on the
outer-training part and scored once on the outer-validation part at its inner threshold. The official holdout is not
touched. All ten inner-tuned models are also scored on each outer-validation part, to see whether a model the rule
picks less often is materially better.

*Computational compromise:* the hyperparameter grids, five inner folds and the 0.01 threshold grid are exactly those
of the final selection; only the number of outer folds (5) limits the cost (about 20 minutes on 2 CPU cores).
""")
code("""
NESTED = T.nested_cv(split, verbose=True)
NS, NF = NESTED["summary"], NESTED["folds"]
display(NF[["fold", "model", "features", "threshold", "inner_cv_money_per_1000", "money_saved", "money_per_1000", "accuracy",
            "recall", "precision", "f1", "false_rejection_rate", "runner_up"]]
        .style.format({"threshold": "{:.2f}", "inner_cv_money_per_1000": "{:,.0f}", "money_saved": "AED {:,.0f}", "money_per_1000": "{:,.0f}",
                       "accuracy": "{:.2%}", "recall": "{:.2%}", "precision": "{:.2%}", "f1": "{:.3f}", "false_rejection_rate": "{:.2%}"})
        .hide(axis="index").set_caption("Selected model per outer fold, scored on its untouched outer-validation part"))
display(NESTED["per_model_summary"].style.format({c: "{:,.0f}" for c in ["mean_money_per_1000", "sd_money_per_1000", "mean_inner_cv_per_1000"]}
        | {c: "{:.2%}" for c in ["mean_recall", "mean_precision", "mean_accuracy"]} | {"mean_threshold": "{:.2f}", "sd_threshold": "{:.2f}"})
        .set_caption("Every candidate inside the same outer folds (each at its own inner threshold)"))
print("Selection counts:", NS["selection_counts"])
print(f"Procedure estimate: AED {NS['procedure_mean_money_per_1000']:,.0f} per 1,000 loans (sd {NS['procedure_sd_money_per_1000']:,.0f}); "
      f"inner CV said {NS['procedure_mean_inner_cv_per_1000']:,.0f} -> optimism {NS['optimism_per_1000']:,.0f} per 1,000")
print(f"Thresholds {NS['threshold_min']:.2f}-{NS['threshold_max']:.2f}; mean recall {NS['procedure_mean_recall']:.1%}, "
      f"precision {NS['procedure_mean_precision']:.1%}, good customers rejected {NS['procedure_mean_false_rejection']:.1%}")
# Is a model the rule picks less often materially better? Paired by outer fold, relative to the deployed model.
W_ = NESTED["per_model"].pivot(index="fold", columns="model", values="money_per_1000")
GAP = W_.sub(W_[DEPLOYED], axis=0).agg(["mean", "std"]).T
GAP["t"] = GAP["mean"] / (GAP["std"] / np.sqrt(len(W_)))
display(GAP.drop(index=DEPLOYED).sort_values("mean", ascending=False).style.format({"mean": "{:+,.0f}", "std": "{:,.0f}", "t": "{:+.1f}"})
        .set_caption(f"Outer-fold money per 1,000 loans: each model minus {DEPLOYED} (paired)"))
best_alt = GAP.drop(index=DEPLOYED)["mean"].idxmax()
print(f"Best alternative: {best_alt}, {GAP.loc[best_alt, 'mean']:+,.0f} per 1,000 (sd {GAP.loc[best_alt, 'std']:,.0f}, "
      f"t = {GAP.loc[best_alt, 't']:+.1f}) ->", "materially better" if GAP.loc[best_alt, "t"] >= 2 else f"not materially better; {DEPLOYED} retained")
""")
md("""
**Reading it.** The table shows how often the rule lands on each model, how far the selected model's thresholds move,
and what the procedure really earns on rows it never saw — the honest figure to quote for the deployment *process*. The
per-model table checks the rule is not leaving money on the table: if a model chosen less often were materially better
on the outer folds, the rule would need revisiting. The final deployment choice is the same procedure re-run on all
4,500 training loans (Section 17a); nested CV assesses it, it does not pick it.

### 17d. Final fit and the one-time holdout evaluation (evaluation C)
""")
code("""
FITTED = T.fit_all(split, TUNED, verbose=False)
RES = T.holdout_results(split, FITTED, cvs)
print("predict() agrees with 0.50 threshold on predict_proba:", {n: round(r["predict_agreement"], 4) for n, r in RES.items()})
y, loan = split.y_test.to_numpy(), split.loan_test
metrics = pd.DataFrame([E.full_row(n, r["family"], y, r["pred_050"], r["scores"], loan, 0.20) for n, r in RES.items()]).set_index("model")
display(metrics[["family", "accuracy", "recall", "precision", "f1", "specificity", "balanced_accuracy", "roc_auc", "pr_auc", "brier"]]
        .style.format({c: "{:.2%}" for c in ["accuracy", "recall", "precision", "specificity", "balanced_accuracy"]} | {"f1": "{:.3f}", "roc_auc": "{:.3f}", "pr_auc": "{:.3f}", "brier": "{:.4f}"}))
""")
code("""
fig, axes = plt.subplots(2, 5, figsize=(17, 6.4))
for a, (n, r) in zip(axes.ravel(), metrics.iterrows()):
    cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
    a.imshow(np.array([[1, -1], [-1, 1]]) * cm / cm.sum(1, keepdims=True), cmap=mpl.colors.LinearSegmentedColormap.from_list("c", [BAD, P["bg3"], P["teal"]]), vmin=-1, vmax=1)
    for i in range(2):
        for j in range(2): a.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=P["t1"], fontsize=13, fontweight="bold")
    a.set(title=n, xticks=[0, 1], yticks=[0, 1], xticklabels=["Approve", "Reject"], yticklabels=["Repaid", "Default"]); a.grid(False)
fig.suptitle("Confusion matrices on the holdout at threshold 0.50 (rows = actual, columns = decision)", color=P["t1"]); plt.tight_layout(); plt.show()
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for n, r in RES.items():
    c = r["curves"]; lw = 2.4 if n == DEPLOYED else 1.1
    ax[0].plot(c["roc"]["fpr"], c["roc"]["tpr"], lw=lw, label=f"{n} ({metrics.loc[n, 'roc_auc']:.3f})", color=C.FAMILY_COLOR[r["family"]] if n != DEPLOYED else P["cyan"])
    ax[1].plot(c["pr"]["recall"], c["pr"]["precision"], lw=lw, label=f"{n} ({metrics.loc[n, 'pr_auc']:.3f})", color=C.FAMILY_COLOR[r["family"]] if n != DEPLOYED else P["cyan"])
ax[0].plot([0, 1], [0, 1], ":", color=P["t3"]); ax[1].axhline(y.mean(), ls=":", color=P["t3"])
ax[0].set(title="ROC curves (legend: ROC-AUC)", xlabel="False positive rate", ylabel="True positive rate (recall)")
ax[1].set(title="Precision-recall curves (legend: PR-AUC)", xlabel="Recall", ylabel="Precision")
for a in ax: a.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------------------------------------- 18
md("""
## 18. Financial calculations
The official function (logic unchanged from the brief) and its decomposition. *Missed-default principal* is a separate
risk-exposure measure and is **not** subtracted a second time.
""")
code("""
import inspect; print(inspect.getsource(F.money_saved))
# hand-checked example: +10,000 caught, -10,000 (0.2 × 50,000) wrongly rejected, +5,000 caught -> 5,000
assert F.money_saved([1, 1, 0, 0, 1], [1, 0, 1, 0, 1], [10_000, 20_000, 50_000, 30_000, 5_000]) == 5_000
base = F.baselines(y, loan, 0.20)
print({k: f"AED {v:,.0f}" for k, v in base.items()})
""")
code("""
fin = pd.DataFrame([{"model": n, **F.breakdown(y, r["pred_050"], loan, 0.20)} for n, r in RES.items()]).set_index("model")
display(fin[["n_tp", "loans_saved", "n_fp", "interest_lost", "net_money_saved", "n_fn", "missed_default_principal"]]
        .sort_values("net_money_saved", ascending=False).style.format({c: "AED {:,.0f}" for c in ["loans_saved", "interest_lost", "net_money_saved", "missed_default_principal"]}))
""")

# ----------------------------------------------------------------------------------------------- 19
md("""
## 19. The leaderboard 🏆 (holdout · threshold 0.50 · interest 20 %)
The same extension stage `train_models.py` runs (`T.extend`) adds the weak-learner baseline, paired bootstrap
intervals, the interest-dependent threshold curves, the no-age fairness variants and the calibration study.

*Reading the champion.* The champion is selected using the common holdout benchmark for assignment comparison. Its
reported result is a model-ranking outcome, not an independent estimate of deployment performance: picking the best
of ten scores on the same 1,500 loans favours whichever model happened to fit this sample best. The deployment
model's figure (§17d) and the nested-CV estimate (§17c) are the deployment-performance evidence.
""")
code("""
LB = T.leaderboard(split, RES)
LB_BIZ = T.leaderboard(split, RES, which="pred_business")
CHAMPION = LB.iloc[0]["model"]
RUN = {"df": df, "audit": audit, "split": split, "split_check": check, "features": FEATURES, "fe_table": fe_table,
       "tuned": TUNED, "params": PARAMS, "tuning_table": tuning_table, "cw_table": cw_table, "oof": OOF, "cv_summary": cvs,
       "fitted": FITTED, "results": RES, "leaderboard": LB, "leaderboard_business": LB_BIZ, "deployed": DEPLOYED,
       "champion": CHAMPION, "data_path": str(D.resolve_data_path())}
T.extend(RUN, verbose=False)
RUN["nested_cv"] = NESTED                                 # evaluation B from §17c, exported with the artifacts
assert RUN["threshold_diagnostics"]["chosen"] == TD["chosen"]
LB = RUN["leaderboard"]                                   # now with bootstrap columns
show = LB[["rank", "model", "family", "accuracy", "recall", "precision", "f1", "tp", "fp", "money_saved",
           "money_ci_low", "money_ci_high", "p_best", "distinguishable"]]
display(show.style.format({"accuracy": "{:.2%}", "recall": "{:.2%}", "precision": "{:.2%}", "f1": "{:.3f}", "p_best": "{:.1%}",
        "money_saved": "AED {:,.0f}", "money_ci_low": "{:,.0f}", "money_ci_high": "{:,.0f}"})
        .background_gradient(subset=["money_saved"], cmap="Greens").hide(axis="index"))
acc_spread = LB.accuracy.max() - LB.accuracy.min(); money_spread = LB.money_saved.max() - LB.money_saved.min()
print(f"Accuracy spread: {acc_spread*100:.2f} points | Money-saved spread: AED {money_spread:,.0f}")
""")
md("### 19a. The weak-learner baseline (brief Step 2 — shown beside, not inside, the ten)")
code("""
(bname, bl), = RUN["baselines"].items()
brow = E.full_row(bname, "Baseline", y, bl["pred_050"], bl["scores"], loan, 0.20)
dt = LB.set_index("model").loc["Decision Tree"]
cols_ = ["accuracy", "recall", "precision", "tp", "fp", "money_saved"]
display(pd.DataFrame([{"model": bname, **{k: brow[k] for k in cols_}},
                      {"model": "Decision Tree (tuned)", **{k: dt[k] for k in cols_}}])
        .style.format({"accuracy": "{:.2%}", "recall": "{:.2%}", "precision": "{:.2%}", "money_saved": "AED {:,.0f}"}).hide(axis="index"))
print(f"Tuning the tree is worth AED {dt.money_saved - brow['money_saved']:,.0f}; the best ensemble adds a further "
      f"AED {LB.money_saved.max() - dt.money_saved:,.0f} on this holdout.")
""")
md("### 19b. Leaderboard charts")
code("""
order = LB.sort_values("money_saved")
cols = [C.FAMILY_COLOR[f] for f in order.family]
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(wspace=.55))
ax[0].barh(order.model, order.accuracy, color=cols); ax[0].set(xlim=(0, 1), title="Accuracy (honest 0–100% scale)", xlabel="Accuracy")
ax[1].barh(order.model, order.recall, color=cols); ax[1].set(title="Default recall", xlabel="Share of defaulters caught", xlim=(0, order.recall.max() * 1.3))
ax[2].barh(order.model, order.money_saved, color=[GOOD if v > 0 else BAD for v in order.money_saved]); ax[2].set(title="Money saved vs approve-all", xlabel="AED")
for a, col, fmt in [(ax[0], "accuracy", "{:.1%}"), (ax[1], "recall", "{:.1%}"), (ax[2], "money_saved", "{:,.0f}")]:
    for i, v in enumerate(order[col]): a.text(v, i, " " + fmt.format(v), va="center", color=P["t1"], fontsize=8.5)
    a.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1) if col != "money_saved" else mpl.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
ax[2].set_xlim(0, order.money_saved.max() * 1.32)
plt.show()
""")
code("""
fig, ax = plt.subplots(figsize=(10, 5.5))
for _, r in LB.iterrows():
    ax.scatter(r.accuracy, r.recall, s=60 + 900 * max(r.money_saved, 0) / LB.money_saved.max(), color=C.FAMILY_COLOR[r.family], alpha=.8, edgecolor=P["t1"] if r["rank"] == 1 else "none", lw=2)
    ax.annotate(r.model, (r.accuracy, r.recall), textcoords="offset points", xytext=(8, 6), fontsize=9, color=P["t2"])
ax.set(title="Accuracy vs recall (bubble = money saved)", xlabel="Accuracy", ylabel="Default recall")
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=1)); ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
plt.show()
""")
md("""
### 19c. How certain is the ranking? — paired bootstrap of the holdout
2,000 resamples of the 1,500 test loans (with replacement); every model is scored on the *same* resample, so the gap
between two models is measured on identical loans. Models are not refitted — this isolates the uncertainty that
comes from evaluating on one finite sample.
""")
code("""
order = LB.sort_values("money_saved")
fig, ax = plt.subplots(figsize=(11, 5.2))
ch = LB.iloc[0]
ax.axvspan(ch.money_ci_low, ch.money_ci_high, color=GOOD, alpha=.08)
for i, r in enumerate(order.itertuples()):
    col = GOOD if r.model == CHAMPION else (P["neutral"] if not r.distinguishable else BAD)
    ax.plot([r.money_ci_low, r.money_ci_high], [i, i], color=col, lw=2.4)
    ax.plot(r.money_saved, i, "D" if r.model == CHAMPION else "o", color=col, ms=8)
    ax.text(order.money_ci_high.max() * 1.02, i, f"P(best) {r.p_best:.0%}", va="center", color=P["t2"], fontsize=9)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order.model)
ax.set(title="Money saved with 95% paired-bootstrap intervals (grey = tied with champion, coral = reliably behind)", xlabel="AED")
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M")); ax.set_xlim(right=order.money_ci_high.max() * 1.16)
plt.tight_layout(); plt.show()
ties = LB[(LB.model != CHAMPION) & (~LB.distinguishable)].model.tolist()
print(f"Statistically tied with {CHAMPION} (95% CI of the paired gap includes 0): {ties}")
""")
md("""
**Reading it.** The intervals are wide (roughly ±AED 1.3M) because only 188 loans default and each defaulted loan is
worth tens of thousands of dirhams. Several top models are statistically tied with the champion; only the weaker tail
is reliably behind. The robust conclusions are therefore *which group* of models to consider and *how to set the
threshold* — not the exact order at the top.
""")
code("""
display(Markdown("**Separate report — holdout at each model's CV-chosen business threshold (20 %)** (not the primary benchmark):"))
display(LB_BIZ[["rank", "model", "accuracy", "recall", "fp", "money_saved"]].assign(threshold=LB_BIZ.model.map(cvs.set_index("model").business_threshold))
        .style.format({"accuracy": "{:.2%}", "recall": "{:.2%}", "money_saved": "AED {:,.0f}", "threshold": "{:.2f}"}).hide(axis="index"))
""")

# ----------------------------------------------------------------------------------------------- 20
md("""
## 20. Comparative financial analysis
### 20a. Interest-rate sensitivity — fixed 0.50 threshold vs thresholds re-optimised per interest
""")
code("""
grid = np.round(np.arange(0.05, 0.501, 0.01), 2)
sens = pd.DataFrame({n: [F.money_saved(y, r["pred_050"], loan, i) for i in grid] for n, r in RES.items()}, index=grid)
sens_biz = pd.DataFrame({n: [F.money_saved(y, (r["scores"] >= F.threshold_at(RUN["threshold_curves"][n], i)).astype(int), loan, i)
                             for i in grid] for n, r in RES.items()}, index=grid)
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
for ax, tab, title in [(axes[0], sens, "0.50 benchmark threshold"), (axes[1], sens_biz, "Business thresholds re-optimised per interest (training OOF)")]:
    lead = tab.mean().idxmax()
    for n in tab: ax.plot(tab.index, tab[n], lw=2.4 if n == lead else 1.1, color=C.FAMILY_COLOR[C.FAMILY[n]], alpha=1 if n == lead else .55, label=n)
    ax.axvline(.2, ls="--", color=P["amber"]); ax.axhline(0, ls=":", color=P["t3"])
    ax.set(title=title, xlabel="Foregone interest (% of principal)", ylabel="Money saved (AED)")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1)); ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
axes[1].legend(ncol=1, fontsize=8.5, bbox_to_anchor=(1.01, 1), loc="upper left"); plt.tight_layout(); plt.show()
for label, tab in [("0.50", sens), ("business", sens_biz)]:
    champ = tab.idxmax(axis=1)
    print(f"Champion by interest ({label}):", " | ".join(champ.groupby((champ != champ.shift()).cumsum())
          .agg(lambda s: (f"{s.index.min():.0%}" if s.index.min() == s.index.max() else f"{s.index.min():.0%}–{s.index.max():.0%}") + f": {s.iloc[0]}")))
""")
md("### 20b. Instructor benchmark comparison (same split, independent models — not adjusted to match)")
code("""
ref = pd.DataFrame(C.REFERENCE_RESULTS).T.rename(columns={"acc": "ref_accuracy", "recall": "ref_recall", "money": "ref_money"})
cmp_ = LB.set_index("model")[["accuracy", "recall", "money_saved"]].join(ref)
cmp_["delta_money"] = cmp_.money_saved - cmp_.ref_money
display(cmp_.style.format({"accuracy": "{:.2%}", "ref_accuracy": "{:.2%}", "recall": "{:.2%}", "ref_recall": "{:.2%}", "money_saved": "{:,.0f}", "ref_money": "{:,.0f}", "delta_money": "{:+,.0f}"}))
print(f"Reference: best money {ref.ref_money.max():,.0f} ({ref.ref_money.idxmax()}), spread {ref.ref_money.max()-ref.ref_money.min():,.0f}")
print(f"This build: best money {LB.money_saved.max():,.0f} ({LB.model[0]}), spread {money_spread:,.0f}")
""")
md("""
**Why results differ legitimately.** The split is identical (Section 8), so differences come from configuration: this
build tunes every model by CV instead of using defaults, adds two engineered features, scales LR/KNN inputs, uses
soft voting and a stacking design of its own. Both builds reproduce the same lesson — accuracy within about one point,
money saved spread by millions — and, per Section 19c, the gap between the two champions is well inside holdout noise.

### 20c. Permutation importance (training-internal validation fold — the test set is not used)
""")
code("""
PERM = T.permutation_study(split, TUNED, verbose=False)
RUN["permutation"] = PERM
imp = pd.DataFrame({n: PERM[n].set_index("feature")["importance_mean"] for n in [CHAMPION, DEPLOYED, "Logistic Regression"]})
display(imp.sort_values(imp.columns[0], ascending=False).style.format("{:+.4f}").background_gradient(cmap="Greens"))
""")
md("""
Importance is the drop in validation ROC-AUC when one raw input is shuffled. It is associative, not causal, and is
shared out between correlated inputs (loan amount, income, debt_to_income).

### 20d. Fairness check — does the model need age?
Each studied model (deployed and leaderboard champion) is rebuilt **without age** and without the age-derived
engineered feature, its business threshold is re-tuned on training OOF scores, and it is scored on the same holdout.
""")
code("""
for name, f in RUN["fairness"].items():
    display(Markdown(f"**{name}**"))
    display(f["table"][["variant", "policy", "threshold", "accuracy", "recall", "roc_auc", "fp", "money_saved"]]
            .style.format({"threshold": "{:.2f}", "accuracy": "{:.2%}", "recall": "{:.2%}", "roc_auc": "{:.3f}", "money_saved": "AED {:,.0f}"}).hide(axis="index"))
seg_ = RUN["fairness"][DEPLOYED]["segments"]
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for a, metric, title in [(ax[0], "rejection_rate", "Rejection rate by age band"), (ax[1], "false_rejection_rate", "Good customers rejected by age band")]:
    w, o = seg_[("with_age", "CV business")], seg_[("without_age", "CV business")]
    xs = np.arange(len(w)); a.bar(xs - .2, w[metric], .4, color=P["blue"], label="With age"); a.bar(xs + .2, o[metric], .4, color=P["teal"], label="Without age")
    a.set_xticks(xs); a.set_xticklabels(w["group"]); a.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    a.set(title=f"{title} — {DEPLOYED}, CV business threshold", xlabel="Age band", ylabel="Rate"); a.legend()
plt.tight_layout(); plt.show()
print("Largest |correlation| of age with any other input:", df.drop(columns="home_ownership").corr()["age"].drop(["age", "default"]).abs().max().round(3))
w_, o_ = seg_[("with_age", "CV business")].set_index("group"), seg_[("without_age", "CV business")].set_index("group")
t_with = RES[DEPLOYED]["business_threshold"]; t_wo = RUN["fairness"][DEPLOYED]["threshold_without_age"]
print(f"Business threshold: {t_with:.2f} with age vs {t_wo:.2f} without")
display(((o_[["rejection_rate", "false_rejection_rate", "recall"]] - w_[["rejection_rate", "false_rejection_rate", "recall"]]) * 100)
        .round(1).rename(columns=lambda c: f"change in {c} (pts)").style.format("{:+.1f}").set_caption(f"{DEPLOYED}: without age minus with age"))
""")
md("""
**Reading it.** Removing age costs nothing in money here — the no-age variants save slightly *more* on this holdout,
well within the bootstrap noise of Section 19c. Rejection and false-rejection rates fall by a few points in **every**
age band without age; that is mostly because the no-age model's tuned business threshold is higher (printed above), so
it rejects fewer applicants overall — a near-uniform shift rather than one that singles out an age band. In this
synthetic data age is almost uncorrelated with the other inputs, so excluding it also removes most of its indirect
influence. That supports dropping age from a production model, but it is a diagnostic, not a fairness certification:
real portfolios contain proxies, and legal and regulatory review is still required.

### 20e. Calibration of the deployed model
""")
code("""
cal = RUN["calibration"]
display(pd.DataFrame(cal["candidates"]).T.rename(columns={"oof_brier": "training-OOF Brier", "oof_ece": "training-OOF ECE"}).style.format("{:.5f}"))
print("Rule:", cal["rule"])
print(f"Chosen calibrator by Brier: {cal['method']} | calibration justified: {cal['justified']} | probability source: "
      f"{cal['probability_source']} | probability display justified: {cal['probability_justified']}")
print("Holdout check:", {k: round(v, 5) for k, v in cal["holdout"].items()})
fig, ax = plt.subplots(figsize=(6.5, 5))
mx = 0.65
ax.plot([0, mx], [0, mx], ":", color=P["t3"], label="perfect")
ax.plot(cal["curve_raw"]["mean_pred"], cal["curve_raw"]["frac_pos"], "o-", color=P["neutral"], label=f"{DEPLOYED} raw")
ax.plot(cal["curve_cal"]["mean_pred"], cal["curve_cal"]["frac_pos"], "o-", color=P["cyan"], label=f"{cal['method']} calibrated")
ax.set(title="Reliability on the holdout (deciles)", xlabel="Mean predicted", ylabel="Observed default rate", xlim=(0, mx), ylim=(0, mx)); ax.legend()
plt.tight_layout(); plt.show()
""")
md("""
**Reading it.** Gradient boosting with log-loss already produces well-calibrated scores, so neither calibrator improves
*both* Brier and ECE on training out-of-fold data and the raw score is kept. Because its training-OOF ECE is within
the 2-point tolerance (and the holdout confirms it), the simulator presents the deployed model's score as an
estimated probability of default; for every other model it remains a ranking score.

### 20f. Reason codes for individual decisions
Exact explanations only: Tree SHAP for tree models, coefficient × (value − training mean) for logistic regression.
Base value + contributions reproduces the model's own output, checked on every call.
""")
code("""
from quickloan import explain as X
ex = X.ReasonExplainer(DEPLOYED, FITTED[DEPLOYED], split.X_train)
thr = cvs.set_index("model").loc[DEPLOYED, "business_threshold"]
s_dep = RES[DEPLOYED]["scores"]
for label, i in [("highest-risk holdout applicant", int(np.argmax(s_dep))), ("lowest-risk holdout applicant", int(np.argmin(s_dep)))]:
    e = ex.explain(split.X_test.iloc[[i]])
    reject = s_dep[i] >= thr
    display(Markdown(f"**{label}** — score {s_dep[i]:.3f} vs threshold {thr:.2f} → **{'REJECT' if reject else 'APPROVE'}** "
                     f"(additivity error {e.max_additivity_error:.1e}, units: {e.units})"))
    display(e.table.head(6).style.format({"contribution": "{:+.3f}"}).hide(axis="index"))
    print("  reason codes:", X.reason_codes(e, reject))
""")

# ----------------------------------------------------------------------------------------------- 21
md("## 21. Business recommendation (generated from the numbers above)")
code("""
L, A_ = LB.iloc[0], LB.loc[LB.accuracy.idxmax()]
dep_biz = LB_BIZ.set_index("model").loc[DEPLOYED]
ties = LB[(LB.model != CHAMPION) & (~LB.distinguishable)].model.tolist()
fair = RUN["fairness"][DEPLOYED]["table"].set_index(["variant", "policy"])["money_saved"]
age_gap = fair[("with_age", "CV business")] - fair[("without_age", "CV business")]
repaid = 1 - y.mean()
dep_thr = cvs.set_index("model").loc[DEPLOYED, "business_threshold"]
lines = [
    "**Recommendation to QuickLoan leadership.**",
    "",
    f"* **Leaderboard champion: {L.model}** — saves **AED {L.money_saved:,.0f}** on the 1,500 test loans versus approving "
    f"everyone at 20 % interest and the 0.50 threshold, catching **{int(L.tp)} of {int(y.sum())} defaulters** (recall "
    f"{L.recall:.1%}) while rejecting {int(L.fp)} good customers (AED {L.interest_lost:,.0f} of foregone interest). Accuracy "
    f"{L.accuracy:.2%}. It is saved as `loan_model_leaderboard_champion.joblib`, as the brief asks.",
    f"* **But the lead is not decisive.** Its 95% bootstrap interval is AED {L.money_ci_low:,.0f} – {L.money_ci_high:,.0f} and "
    f"it wins only **{L.p_best:.0%}** of resamples; {len(ties)} other models ({', '.join(ties)}) are statistically tied with it.",
    f"* **Accuracy would not have found it.** All ten models sit within **{acc_spread*100:.1f} accuracy points**, but money "
    f"saved ranges over **AED {money_spread:,.0f}**. The most accurate model ({A_.model}, {A_.accuracy:.2%}) saves "
    f"AED {A_.money_saved:,.0f}. With {repaid:.1%} of holdout loans repaid, accuracy mostly rewards approving good customers.",
    f"* **What to deploy — and why it differs from the champion.** The champion was chosen *using* the holdout, so its "
    f"holdout figure is optimistic. The leakage-free choice, made on training cross-validation before the holdout was "
    f"opened, is **{DEPLOYED}** with a money-maximising threshold of {dep_thr:.2f} at 20 % (`loan_model.joblib`). On the "
    f"untouched holdout that policy saves **AED {dep_biz.money_saved:,.0f}** (recall {dep_biz.recall:.1%}, {int(dep_biz.fp)} "
    f"good customers rejected) — {'more than' if dep_biz.money_saved > LB.money_saved.max() else 'compared with'} "
    f"the best model at 0.50 (AED {LB.money_saved.max():,.0f}). The threshold is the biggest money lever and is "
    f"re-optimised for other interest assumptions from the same training curves.",
    f"* **The procedure is checked, not just the winner.** Nested cross-validation (5 outer × 5 inner folds, training "
    f"data only) re-ran the whole selection {NS['outer_folds']} times: it chose "
    + ", ".join(f"{k} {v}×" for k, v in NS["selection_counts"].items()) +
    f", with thresholds {NS['threshold_min']:.2f}–{NS['threshold_max']:.2f}, and earned **AED "
    f"{NS['procedure_mean_money_per_1000']:,.0f} per 1,000 loans** on rows it never saw (sd "
    f"{NS['procedure_sd_money_per_1000']:,.0f}); the deployed policy's holdout figure is AED "
    f"{dep_biz.money_saved / len(y) * 1000:,.0f} per 1,000.",
    f"* **Age is not needed.** Without age the deployed model saves AED {abs(age_gap):,.0f} "
    f"{'less' if age_gap > 0 else 'more'} at its business threshold — inside holdout noise — so the production model "
    f"should exclude it, subject to legal review.",
    f"* **Trade-off accepted.** Higher recall costs some good customers and interpretability. Exact reason codes (Tree "
    f"SHAP) make each {DEPLOYED} decision explainable, and its raw score is already well calibrated, so it can be read "
    f"as a probability of default.",
    "* **Before production:** validate on a later time period, model recoveries and funding costs, and run full fairness, "
    "compliance and model-risk review. This is an educational model, not an approval system.",
]
display(Markdown("\\n".join(lines)))
""")

# ----------------------------------------------------------------------------------------------- 22
md("""
## 22. Model export
Calls `train_models.export(RUN)` — the exact function the script uses — then reloads every artifact independently.
Artifact contract: see `quickloan/artifacts.py` and the README.
""")
code("""
import train_models
lb_csv, biz_csv = train_models.export(RUN)
dep, champ = A.load_model_artifact(C.MODEL_ARTIFACT), A.load_model_artifact(C.CHAMPION_ARTIFACT)
assert np.allclose(dep["pipeline"].predict_proba(split.X_test)[:, 1], RES[DEPLOYED]["scores"])
assert np.allclose(champ["pipeline"].predict_proba(split.X_test)[:, 1], RES[CHAMPION]["scores"])
evr = A.load_eval_artifact()
for name, art_ in [(C.MODEL_ARTIFACT.name, dep), (C.CHAMPION_ARTIFACT.name, champ)]:
    print(f"{name}: {art_['model_name']} · selected_on={art_['selected_on']} · business threshold @20% "
          f"{art_['thresholds']['business']:.2f} · @50% {art_['thresholds']['business_by_interest'][50]:.2f} · "
          f"probability available: {art_['score_type']['probability_available']}")
print("Evaluation artifact keys:", len(evr))
display(lb_csv[["rank", "model", "money_saved", "money_ci_low", "money_ci_high", "p_best"]].head(5))
""")

md("""
## Summary check
Re-computes every headline figure quoted in the executive summary at the top and fails loudly if any differs.
""")
code("""
H = """ + repr(HEADLINE) + """
L = LB.iloc[0]
dep_row = LB_BIZ.set_index("model").loc[DEPLOYED]
fair_t = RUN["fairness"][DEPLOYED]["table"].set_index(["variant", "policy"])["money_saved"]
checks = {
    "champion": (L.model, H["champion"]),
    "champion_money": (int(round(L.money_saved)), H["champion_money"]),
    "acc_spread_pts": (round(acc_spread * 100, 1), H["acc_spread_pts"]),
    "money_spread": (int(round(money_spread)), H["money_spread"]),
    "p_best_pct": (int(round(L.p_best * 100)), H["p_best_pct"]),
    "n_tied": (int(((LB.model != L.model) & (~LB.distinguishable)).sum()), H["n_tied"]),
    "deployed": (DEPLOYED, H["deployed"]),
    "deployed_threshold": (round(float(cvs.set_index("model").loc[DEPLOYED, "business_threshold"]), 2), H["deployed_threshold"]),
    "deployed_money": (int(round(dep_row.money_saved)), H["deployed_money"]),
    "deployed_recall_pct": (round(dep_row.recall * 100, 1), H["deployed_recall_pct"]),
    "age_gain": (int(round(fair_t[("without_age", "CV business")] - fair_t[("with_age", "CV business")])), H["age_gain"]),
    "nested_per_1000": (int(round(NS["procedure_mean_money_per_1000"])), H["nested_per_1000"]),
    "nested_sd_per_1000": (int(round(NS["procedure_sd_money_per_1000"])), H["nested_sd_per_1000"]),
    "nested_deployed_folds": (int(NS["selection_counts"].get(DEPLOYED, 0)), H["nested_deployed_folds"]),
    "nested_no_material_alternative": (bool(GAP.drop(index=DEPLOYED)["t"].max() < 2), True),
}
bad = {k: v for k, v in checks.items() if v[0] != v[1]}
display(pd.DataFrame(checks, index=["computed", "summary"]).T)
assert not bad, f"Executive summary out of date: {bad}"
print(f"All {len(checks)} executive-summary figures match the computed results.")
""")

# ---- anchors so the rubric map links work (Jupyter, JupyterLab, VS Code)
import re as _re
_ids = set()
for _c in cells:
    if _c.cell_type != "markdown":
        continue
    _lines = _c.source.split("\n")
    for _i, _l in enumerate(_lines):
        _m = _re.match(r"^#{2,3} (\d+[a-z]?)(?:[.–]|–\d+\.)", _l)
        if _m:
            _id = "s" + _m.group(1)
            if _id not in _ids:
                _lines[_i] = f'<a id="{_id}"></a>\n\n' + _l
                _ids.add(_id)
    _c.source = "\n".join(_lines)
_links = set(_re.findall(r"\(#(s[0-9a-z]+)\)", "\n".join(c.source for c in cells if c.cell_type == "markdown")))
assert _links <= _ids, f"rubric/summary links without a target: {sorted(_links - _ids)}"

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = Path(__file__).resolve().parent.parent / "QuickLoan_Model_Bakeoff.ipynb"
nbf.write(nb, out)
print("wrote", out)
