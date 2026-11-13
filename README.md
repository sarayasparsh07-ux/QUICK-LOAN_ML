# QuickLoan | Credit Risk Intelligence

An educational credit-risk project for QuickLoan, a fictional UAE personal-loan fintech. Ten classifiers — three single
models and seven ensembles covering **Voting, Bagging, Boosting and Stacking** — are trained on `loans.csv`, scored on
one common holdout, and compared on **money saved**, not just accuracy — with bootstrap confidence intervals, business
thresholds re-optimised for any interest assumption, a with/without-age fairness check, a calibration study and exact
reason codes for individual decisions. A Streamlit dashboard turns the results into an interactive analytics product.

> Educational analytics only. Not connected to live lending, not a certified approval system, and not validated for
> fairness, affordability or regulatory compliance.

---

## Installation

**Python 3.13 (64-bit) is required.** The saved artifacts were built on **Python 3.13.18**, and that is what their
metadata records. Two facts about the 3.13 releases decide how to install:

- **Python 3.13.17** (21 Oct 2026) is the **last 3.13 release with Windows installers** (`python-3.13.17-amd64.exe`),
  published as an exception after 3.13 entered its security-only phase.
- **Python 3.13.18** (25 Oct 2026) is a **source-only security release**. python.org provides no Windows or macOS
  installer for it.

So Windows users install 3.13.17 (Route A). Only users who need exact patch-level parity with the build environment
need 3.13.18 (Route B). The saved models depend on the *library* versions, which are pinned exactly in
`requirements.txt`. They do not depend on the Python patch release, and the dashboard's version check compares
libraries only.

### Route A — recommended (Windows): Python 3.13.17, 64-bit

1. Download the **Windows installer (64-bit)** (`python-3.13.17-amd64.exe`, not the 32-bit or ARM64 one) for Python
   3.13.17 from <https://www.python.org/downloads/release/python-31317/> and run it. Tick **"Add python.exe to PATH"**.
2. Open PowerShell or Command Prompt in the project folder and run:

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip check
```

`python -m pip check` must print `No broken requirements found.` and `python --version` inside the activated
environment should show `Python 3.13.17`. A clean install takes about a minute and downloads only ready-made wheels.

On macOS or Linux, use any Python 3.13 (3.13.17 from python.org for macOS, or your package manager):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip check
```

**What was verified for Route A.** A fresh Python 3.13.17 virtual environment was tested with an unmodified CPython
3.13.17 interpreter (the conda-forge build, Linux x86-64) and the exact commands above: every pin installed, `pip check` was clean,
PyArrow 26.0.0 was present, all three artifacts loaded and reproduced the stored predictions and money figures, the
full test suite passed, and `streamlit run app.py` served every page without errors. For Windows specifically, all 92
packages resolve to ready-made `win_amd64` / `py3-none-any` wheels for Python 3.13. The artifacts contain no
operating-system-specific objects such as file paths. The install was not run on a physical Windows machine.

### Route B — advanced: exact build-environment reproduction (Python 3.13.18)

For patch-level parity with the environment that generated the artifacts, use a Python **3.13.18** interpreter. On
Windows this means building it from the python.org source release
(<https://www.python.org/downloads/release/python-31318/>). On macOS/Linux, a distribution that ships 3.13.18
(pyenv, conda-forge `python=3.13.18`, or your package manager) also works. Then create the virtual environment and
install `requirements.txt` exactly as in Route A. This route is optional: it reproduces the build environment but is
not needed to run, test or grade the project. The artifacts were built, and the full test suite verified, on 3.13.18
(Linux x86-64).

### Environment summary

| Package | Version | Why it is needed |
|---|---|---|
| Python | 3.13.17 (Route A, tested) · 3.13.18 (build environment, Route B) | runtime |
| pandas | 3.0.5 | data handling; text columns are Arrow-backed when pyarrow is installed |
| numpy · scipy | 2.5.3 · 1.18.1 | numerics (scipy is scikit-learn's backend) |
| scikit-learn | 1.9.1 | pipelines, nine of the ten models, CV, metrics |
| xgboost | 3.4.1 | the XGBoost model (and inside Stacking) |
| joblib · cloudpickle | 1.6.0 · 3.1.2 | saving / loading the artifacts (self-contained model files) |
| **pyarrow** | **26.0.0** | **required** — see below |
| shap | 0.54.1 | exact Tree SHAP reason codes in the simulator |
| streamlit · plotly | 1.66.1 · 7.1.0 | dashboard and interactive charts |
| matplotlib · ipython · ipykernel · nbformat · nbclient · nbconvert | see `requirements.txt` | executing the notebook |
| pytest | 9.1.1 | automated tests |

**Why PyArrow is pinned.** It is a genuine runtime dependency, not an optional extra: (1) with pyarrow installed,
pandas 3 stores text columns (`home_ownership`, model names) as Arrow strings, and
`artifacts/evaluation_results.joblib` contains such DataFrames — without pyarrow it cannot be unpickled at all;
(2) Streamlit requires pyarrow to render every table (`st.dataframe`). The two model files
(`loan_model.joblib`, `loan_model_leaderboard_champion.joblib`) do not need it. Version 26.0.0 is the one that created
the evaluation artifact.

### Artifact requirements

The dashboard needs three files, all shipped and built from `loans.csv` by `train_models.py`:
`loan_model.joblib` (deployed model), `loan_model_leaderboard_champion.joblib` (0.50 leaderboard champion) and
`artifacts/evaluation_results.joblib` (holdout predictions, OOF scores, studies, nested CV). Nothing else on disk is
read at runtime except the bundled font and logo files. If a file is missing or was produced by a different version,
the dashboard shows a fix-it screen instead of crashing.

**Loading an artifact yourself.** Both model files are self-contained: the project's own classes are stored inside them,
so they load with only the pinned libraries, even outside this folder:

```python
import joblib, pandas as pd
art = joblib.load("loan_model.joblib")                       # dict: pipeline, thresholds, metrics, metadata ...
X = pd.read_csv("loans.csv")[art["raw_features"]]            # the 10 raw columns, in this order
score = art["pipeline"].predict_proba(X)[:, 1]               # probability of default (positive class = default = 1)
reject = score >= art["thresholds"]["business"]              # deployment threshold (0.16 at 20 % interest)
```

`artifacts/evaluation_results.joblib` needs PyArrow and the project's `quickloan` package, so load it from the project
folder with `quickloan.artifacts.load_eval_artifact()`.

## Running

All commands run from the project folder inside the activated virtual environment.

```bash
streamlit run app.py            # the dashboard (artifacts are already built) → http://localhost:8501
python train_models.py          # optional: rebuild every artifact from loans.csv (~27 min on 2 CPU cores,
                                #           of which ~20 min is nested cross-validation)
```

### Testing

```bash
python -m compileall -q .       # byte-compiles every source file; no output means no syntax errors
python -m pytest -q             # the full automated suite (~2 min)
```

The suite has **202 tests**: data validation and the official split, the money-saved formula and its edge cases,
feature engineering and preprocessing, all ten models and their stored predictions, threshold selection, nested cross-validation and
leakage, artifact loading and contract, the instructor's starter app (run in a separate process), and Streamlit
AppTest integration tests of every dashboard page, the overview cards, the scatter labels and the simulator. Tests that
read the stored artifacts are marked to *skip* (not pass) if the artifacts are missing. With the shipped artifacts,
nothing should be skipped: pytest's summary line must read `202 passed`. Use `python -m pytest -q -rs` to list any
skips and their reasons.

`loans.csv` (or `loans(1).csv`) must sit in the project folder. To execute the notebook end-to-end (same run time
as `train_models.py`, and it rewrites the same artifacts):

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 QuickLoan_Model_Bakeoff.ipynb
```

The notebook and `train_models.py` call the same `quickloan` functions with fixed seeds and produce identical artifacts.
The notebook opens with an **executive summary** and a **rubric map** linking each of the brief's marking criteria to its
section; its last cell re-computes every headline figure in the summary and fails if any is out of date.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `py -3.13` not found (Windows) | Install Python 3.13.17 (64-bit) from python.org (tick "Add python.exe to PATH"), then reopen the terminal. There is no Windows installer for 3.13.18. |
| `.venv\Scripts\activate` blocked in PowerShell | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt. |
| pip tries to *build* numpy / pandas / pyarrow from source | The interpreter is not Python 3.13 64-bit. Check `python --version`; recreate the venv with 3.13. |
| `ModuleNotFoundError: pyarrow` or "evaluation_results.joblib could not be loaded" | pyarrow is missing — `pip install -r requirements.txt` inside the activated venv. |
| Dashboard shows a version warning | Installed versions differ from the training ones. Reinstall the pins, or run `python train_models.py` to rebuild for your environment. |
| Dashboard shows "artifacts missing" | Run `python train_models.py` from the project folder (needs `loans.csv`). |
| `python -m compileall -q .` is slow | It also compiles the libraries inside `.venv`. That is harmless, or use `python -m compileall -q app.py train_models.py quickloan ui tests tools`. |

## Using the instructor's starter `app.py`

`loan_model.joblib` works unchanged in the starter dashboard supplied with the brief. Put the starter `app.py`,
`loan_model.joblib` and `loans.csv` in one folder and run `streamlit run app.py` — this project's code does not need to
be present. Two things make that work:

- `model` is a wrapper that accepts the starter app's one-hot `home_ownership_<level>` columns as well as raw inputs,
  with identical predictions; `columns` is the starter's encoded schema.
- The project's own classes are stored inside the model file ("pickled by value"), so it loads without the
  `quickloan` package. The pinned library versions are still required.

The starter app uses the standard 0.50 cut-off, so its money card shows the 0.50 figure for the model (AED 4,700,260
for the deployed Gradient Boosting; AED 4,891,400 if you rename the champion artifact to `loan_model.joblib`).
`tests/test_starter_compat.py` runs the unmodified starter file (`tests/fixtures/instructor_starter_app.py`) exactly
this way, in an isolated Python process, for both artifacts.

## Project layout

```
app.py                                  Streamlit entry point (page navigation only)
train_models.py                         Reproduces training, evaluation and artifact export from loans.csv
QuickLoan_Model_Bakeoff.ipynb           Executed notebook: data → EDA → features → 10 models → leaderboard →
                                        uncertainty → fairness → calibration → reason codes → verdict → export
loan_model.joblib                       Deployed pipeline, chosen on training CV (leakage-free)
loan_model_leaderboard_champion.joblib  Holdout leaderboard winner (the brief's "best model"), same contract
model_leaderboard.csv                   Ten models on the holdout at 0.50 / 20 %, with 95 % bootstrap CIs and P(best)
model_leaderboard_business.csv          Same models at their CV business thresholds tuned at 20 % (separate report)
artifacts/evaluation_results.joblib     Holdout predictions, OOF scores, threshold curves, studies — what the dashboard plots
quickloan/                              Modelling package
  config.py      constants, schema, paths, colour tokens  data.py       load, validate, audit, split
  features.py    leakage-free engineered features (+drop) finance.py    official money-saved metric, breakdowns,
  models.py      the ten pipelines and their grids                      threshold curves, bootstrap
  evaluation.py  metrics, curves, ECE, segment diagnostics training.py  workflow, selection procedure, nested CV
  explain.py     exact reason codes (Tree SHAP / linear)   artifacts.py  save / load / validate the contract
  compat.py      starter-app compatibility wrapper
ui/                                     Dashboard: shell (chrome + controls), theme, components, charts, state,
                                        narrative, pages
static/fonts/                           Inter variable font (SIL OFL), served locally — no external font request
assets/                                 logo.svg and icon.svg (sidebar logo and collapsed-sidebar icon)
docs/screenshots/                       Captures of the running dashboard
tests/                                  test_finance.py · test_artifacts.py · test_enhancements.py ·
                                        test_nested_cv.py (selection procedure, leakage) ·
                                        test_dashboard.py · test_selection_cards.py (Streamlit AppTest) ·
                                        test_starter_compat.py · page_harness.py ·
                                        fixtures/instructor_starter_app.py (unmodified)
requirements.txt                        Exact library pins (Python 3.13)
pytest.ini                              Points pytest at tests/
tools/build_notebook.py                 Generates the notebook from source
.streamlit/config.toml                  Dark theme, bundled Inter font, palette
```

## Dataset

6,000 loans × 10 features + `default` (1 = defaulted, 0 = repaid). Default rate **12.55 % on the full dataset** and
**12.53 % on the holdout** (188 / 1,500); the dashboard labels which one each figure uses. No missing values or
duplicates; one plausibility flag (below). **`debt_to_income`:** 14 stored values differ from
`round(loan_amount / annual_income, 2)` — 8 are capped at 3.00 and 6 are half-way cases stored rounded down. Training uses
the values as recorded. The simulator computes the same quantity as a documented *loan-to-income proxy* (rounded to
2 dp, capped at 3.00); it is not a conventional debt-to-income ratio.

## Method

| Step | What happens | Data used |
|---|---|---|
| Split | 25 % test, `random_state=1`, stratified. Reconciles exactly with the instructor reference (1,500 / 188 / AED 12,271,900 / AED 62,945,000). | — |
| Feature engineering | Six candidates tested one at a time with LR and GB; kept if mean CV PR-AUC rises > 0.002. Retained: `has_past_default`, `employment_share_of_adult_life`. | training CV |
| Preprocessing | `FeatureEngineer → OneHotEncoder(handle_unknown="ignore") + StandardScaler (LR, KNN) / passthrough (trees)` inside every pipeline. | fitted on train |
| Tuning | Small grids, 5-fold stratified CV, average precision. | training CV |
| Class weighting | Studied (balanced vs none) but not applied; the threshold is moved explicitly instead. | training CV |
| Thresholds | Primary leaderboard at 0.50. Business threshold per model = maximiser of out-of-fold money saved, stored as a curve so it is re-optimised for every interest from 5 % to 50 %. | training OOF |
| Deployment choice | Highest CV money saved at its business threshold (`training.choose_deployment`; ties by name). The holdout plays no part. | training OOF |
| Nested CV of the procedure | The whole selection procedure above (`training.select_deployment`) repeated inside 5 outer folds (StratifiedKFold, shuffled, seed 2) with 5 inner folds (seed 1), scored on outer-validation rows it never saw. Assesses the procedure; does not pick the model. | training only |
| Threshold stability | Deployed model's OOF money / recall / precision / false-rejection curve, the 99 % and 98 % near-optimal bands, and each inner fold's own optimum. | training OOF |
| Calibration | Sigmoid and isotonic compared with the raw score; used only if both Brier and ECE improve. | training OOF |
| Fairness check | Deployed and champion models rebuilt without age (and the age-derived feature), thresholds re-tuned. | training CV, then holdout |
| Importance | Permutation importance on a validation fold carved from the training set. | training only |
| Evaluation | Every model scored once on the same 1,500 test loans. | test (once) |
| Uncertainty | Paired bootstrap of the holdout, 2,000 resamples, no refitting. | test predictions |

**Models:** Decision Tree, Logistic Regression, KNN · Voting (soft: DT + LR + KNN) · Bagging (200 trees) · Random Forest ·
AdaBoost · Gradient Boosting · XGBoost · Stacking (RF + XGBoost + LR → logistic meta-model, 5-fold OOF). Plus an
**untuned depth-3 tree** as the brief's weak-learner baseline, reported beside — not inside — the ten.

## Financial metric

The assignment's `money_saved` function is used unchanged (`quickloan/finance.py`):

```
money saved = principal of correctly rejected defaulters − interest × principal of wrongly rejected repayers
```

It is measured against approving everyone (which scores AED 0). *Missed-default principal* is reported as a separate
risk-exposure figure and never subtracted a second time. The dashboard's interest slider (5–50 %, step 1 %) is the
foregone interest as a share of principal, **not an APR**; every money figure, the champion and the recommendation are
recomputed from stored holdout predictions with the official function.

Assumptions: full principal loss on default; no recoveries, collateral, fees, operating or funding costs. This is not
audited accounting profit or an expected-credit-loss model.

## Results

**Primary leaderboard — holdout, threshold 0.50, interest 20 %** (`model_leaderboard.csv`)

| # | Model | Family | Accuracy | Recall | Good rejected | Money saved | 95 % CI | P(best) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **Stacking** | Stacking | 89.73 % | 33.5 % | 29 | **AED 4,891,400** | 3.64–6.33M | 44 % |
| 2 | Decision Tree ≈ | Single | 89.53 % | 33.5 % | 32 | AED 4,808,180 | 3.56–6.22M | 34 % |
| 3 | Bagging ≈ | Bagging | 89.80 % | 33.5 % | 28 | AED 4,792,000 | 3.56–6.16M | 14 % |
| 4 | Gradient Boosting ≈ | Boosting | 89.73 % | 31.9 % | 26 | AED 4,700,260 | 3.50–6.10M | 4 % |
| 5 | XGBoost ≈ | Boosting | 89.80 % | 33.0 % | 27 | AED 4,691,620 | 3.49–6.03M | 4 % |
| 6 | AdaBoost | Boosting | 89.67 % | 29.8 % | 23 | AED 4,364,280 | 3.21–5.71M | 0 % |
| 7 | Random Forest | Bagging | 89.40 % | 29.8 % | 27 | AED 4,303,780 | 3.13–5.67M | 0 % |
| 8 | Voting | Voting | 89.47 % | 25.5 % | 18 | AED 3,927,220 | 2.84–5.23M | 0 % |
| 9 | Logistic Regression | Single | 89.00 % | 20.2 % | 15 | AED 3,308,220 | 2.24–4.50M | 0 % |
| 10 | KNN | Single | 88.87 % | 16.0 % | 9 | AED 2,634,960 | 1.70–3.71M | 0 % |
| – | *Shallow tree (depth 3, untuned) — baseline* | Baseline | 89.47 % | 27.7 % | 22 | AED 3,926,160 | not ranked | – |

≈ = statistically tied with the champion: the 95 % paired-bootstrap interval of the gap includes zero.
Reference strategies: approve everyone AED 0 · reject everyone −AED 317,100.

**The lesson:** accuracy spans **0.9 points**; money saved spans **AED 2.26M**. Bagging and XGBoost are the most
accurate (89.80 %) yet save less than Stacking, which catches the same number of defaulters but larger loans.

**How certain is the ranking.** Paired bootstrap, 2,000 resamples of the 1,500 holdout loans. The champion
wins only 44 % of resamples, and four models are statistically tied with it, so the top five should be read as a
group. Only AdaBoost, Random Forest, Voting, Logistic Regression and KNN are reliably behind.

**Weak-learner baseline.** The untuned depth-3 tree catches 52 defaulters and saves AED 3.93M. Tuning the
tree adds AED 0.88M, and the best ensemble adds a further AED 0.08M — inside the noise.

**Instructor benchmark:** reference champion XGBoost AED 4,826,940, spread AED 2,605,920. This build's champion
(Stacking, AED 4,891,400) and spread (AED 2,256,440) differ legitimately — same split, independently tuned models —
and were not adjusted to match. The gap between the two champions is far inside the bootstrap interval.

**Business thresholds.** Tuned on training out-of-fold scores, they sit at 0.14–0.30 at 20 % interest,
near the break-even 0.20 / 1.20 ≈ 0.17, and **rise with the interest assumption** (Gradient Boosting: 0.16 at 20 %,
0.23 at 35 %, 0.26 at 50 %). The dashboard re-optimises them live for the slider; `loan_model.joblib` stores the full
5–50 % table. At 20 % every model saves more than at 0.50; the best is AdaBoost (t = 0.30) at AED 6,231,400.

**Champion vs deployed model.** Two different questions, two artifacts:

| Artifact | Model | Chosen on | Holdout money saved (20 %) |
|---|---|---|---:|
| `loan_model_leaderboard_champion.joblib` | Stacking | the holdout leaderboard at 0.50 (the brief's "best model") | AED 4,891,400 at 0.50 |
| `loan_model.joblib` | Gradient Boosting, threshold 0.16 | training cross-validation, before the holdout was opened | **AED 5,943,420** |

The champion is selected using the common holdout benchmark for assignment comparison. Its reported result is a
model-ranking outcome, not an independent estimate of deployment performance: choosing the best of ten scores on the
same 1,500 loans builds selection optimism into the winning figure. The deployment model's holdout figure and the
nested-CV estimate below are the deployment-performance evidence. The dashboard's Executive Overview, Model
Leaderboard, Financial Performance and Methodology pages, and the notebook (§17, §19), use this same wording.

The deployed policy catches 102 of 188 defaulters (recall 54.3 %) and rejects 142 good customers, at 84.8 % accuracy —
more money with *lower* accuracy. The threshold is the biggest money lever in this problem.

**Three evaluations, kept separate.** All use the official money-saved formula; none is reused to answer another's
question.

| | Evaluation | Data | Threshold | Result (20 % interest) |
|---|---|---|---|---|
| **A** | Benchmark leaderboard | 1,500-loan holdout | 0.50 for all ten models | Champion **Stacking**, AED 4,891,400 (AED 3.26M per 1,000 loans) — a model-ranking outcome |
| **B** | Nested cross-validation of the selection procedure | 4,500 training loans only | chosen inside each outer fold | **AED 4.13M per 1,000 loans** (sd AED 0.30M across 5 outer folds) |
| **C** | Final deployment holdout | 1,500-loan holdout, opened once | 0.16, from training CV | **Gradient Boosting**, AED 5,943,420 (AED 3.96M per 1,000 loans) |

**Nested cross-validation (evaluation B).** The complete selection procedure — engineered-feature selection, tuning,
inner 5-fold out-of-fold scores, business thresholds and the model choice (`training.select_deployment`) — was re-run
inside each of 5 outer folds of the training set and scored once on the outer-validation rows it never saw.

| Outer fold | Selected | Threshold | Inner-CV estimate / 1,000 | Outer-validation money (900 loans) | / 1,000 | Accuracy | Recall | Precision | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | XGBoost | 0.24 | 4.33M | AED 3,810,120 | 4.23M | 88.9 % | 47.8 % | 56.8 % | 0.519 |
| 2 | Gradient Boosting | 0.16 | 4.48M | AED 3,517,120 | 3.91M | 86.1 % | 51.3 % | 45.3 % | 0.481 |
| 3 | Random Forest | 0.17 | 4.47M | AED 3,742,980 | 4.16M | 83.4 % | 54.0 % | 38.6 % | 0.450 |
| 4 | Gradient Boosting | 0.19 | 4.54M | AED 3,428,240 | 3.81M | 84.6 % | 51.3 % | 40.8 % | 0.455 |
| 5 | Gradient Boosting | 0.20 | 4.26M | AED 4,103,760 | 4.56M | 87.6 % | 59.3 % | 50.4 % | 0.545 |

- **Stability.** Gradient Boosting was chosen in 3 of 5 folds (XGBoost and Random Forest once each); thresholds ranged
  0.16–0.24 (mean 0.19), always far below 0.50. Fold 1 was a near-tie (inner margin AED 517 per 1,000 loans). The
  engineered features kept also varied between folds — feature selection is the least stable step.
- **Honest estimate.** The procedure earned AED 4.13M per 1,000 loans on unseen rows, versus AED 4.42M that the inner
  CV promised: about AED 0.28M per 1,000 (6 %) of selection optimism. On the same outer folds the selected models at
  0.50 would have saved AED 14.59M in total against AED 18.60M at their business thresholds (+27 %).
- **No better alternative was ignored.** Scoring all ten inner-tuned models on every outer fold, the best alternative
  (Random Forest) beat Gradient Boosting by only AED 34K per 1,000 loans on average, against a fold-to-fold spread of
  AED 191K (paired t = 0.4); Voting, AdaBoost and XGBoost are similar. No candidate is materially better, so the
  deployment choice is unchanged.
- **Final selection.** The same procedure re-run on all 4,500 training loans selects Gradient Boosting at 0.16 — the
  existing `loan_model.joblib`, which is therefore unchanged in model and threshold (only its metadata gained the
  nested-CV results). The holdout figure in C (AED 3.96M per 1,000) sits inside the nested-CV range.

**Threshold stability (training out-of-fold, Gradient Boosting).** Only 0.16 is within 1 % of the maximum OOF money,
but every cut-off from 0.16 to 0.23 stays within 2 %; the five inner folds individually prefer 0.17, 0.19, 0.16, 0.16
and 0.23. At 0.50 the same scores keep only 73 % of the maximum, so moving away from 0.50 matters far more than the
second decimal.

*Computational compromise:* nested CV uses exactly the final grids, five inner folds and the 0.01 threshold grid; only
the number of outer folds (5) bounds the cost, which is about 20 minutes on 2 CPU cores.

**Fairness check.** Without age (and the age-derived engineered feature), Gradient Boosting saves AED 28,800
*more* at its business threshold and Stacking AED 42,820 more — well inside the noise — and the false-rejection rate
falls in every age band, mostly because the no-age model's tuned threshold is higher (0.22 vs 0.16). Age is almost
uncorrelated with the other inputs here (largest |r| = 0.02), so removing it also removes most of its indirect influence.
It is a diagnostic, not a fairness certification.

**Calibration.** For Gradient Boosting, neither sigmoid nor isotonic calibration improves *both* training-OOF
Brier and ECE, so the raw score is kept. Its training-OOF ECE is 1.2 points (within the 2-point tolerance; holdout ECE
0.6 points), so the simulator shows it as an estimated probability of default for the deployed model only.

**Reason codes.** Exact Tree SHAP (Decision Tree, Random Forest, Gradient Boosting, XGBoost) and exact linear
contributions (Logistic Regression), each checked for additivity (error < 1e-4) on every call. KNN, Voting, Bagging,
AdaBoost and Stacking would need sampling-based approximations, so the simulator says so instead of guessing.

**Data-quality flag:** 278 rows (4.6 %) report more years employed than years since age 16 (mostly ages 21–29). The
data is synthetic and the faulty field is unknown, so the rows are kept and flagged rather than removed.

## Dashboard

Six pages, all driven by three sidebar controls (foregone interest, threshold policy, focus model):

1. **Executive Overview** — hero, six KPI cards, two model-selection cards (**Holdout leaderboard champion** at the common 0.50 threshold, with its qualification as a model-ranking outcome, and **Recommended deployment model** with its training-CV threshold, CV, nested-CV and holdout money — all recomputed live for the interest slider) with the methodology note, money-saved chart, accuracy-vs-money spread, holdout uncertainty and a dynamic management recommendation.
2. **Model Leaderboard** — sortable table with 95 % CI, P(best) and tie markers (★ champion, ◉ focus, ◆ deployed), the baseline row, accuracy / recall / money-saved bars, the confidence-interval chart, and the accuracy-vs-recall bubble chart (labels placed collision-free; points keep their true coordinates).
3. **Financial Performance** — loans saved, interest lost, net, missed-default exposure; waterfall; caught-vs-missed principal; strategy comparison; 5–50 % sensitivity with champion ranges (thresholds re-optimised per interest under the business policy).
4. **Model Diagnostics** — confusion matrix, ROC and PR curves, permutation importance, raw-vs-calibrated reliability, a full metric table, and segment diagnostics for the current scenario.
5. **Loan Decision Simulator** — grouped applicant form, any of the ten models, the business threshold for the current interest or 0.50, a risk-score gauge, an estimated probability of default (deployed model), exact reason codes, and out-of-range warnings (inputs are never altered).
6. **Methodology and Data** — data quality, DTI issue, split reconciliation, EDA, feature engineering, tuning, class weighting, CV deployment choice, the three evaluations A / B / C, nested cross-validation per fold and per model, threshold stability, threshold-by-interest chart, instructor benchmark, **Fairness check**, limitations, artifact contract.

Missing or incompatible artifacts show an in-app message telling you to run `python train_models.py`. The Inter font
is bundled in `static/fonts/`, so the dashboard looks the same offline.

## Automated tests

`tests/test_dashboard.py` drives the real app with Streamlit's `AppTest` (no browser needed), through
`tests/page_harness.py`, which runs the production `ui/shell.py`. It checks that:

- every page renders without exceptions under both threshold policies, and `app.py` itself runs;
- the slider (5 / 20 / 37 / 50 %) and policy produce the correct champion and money, recomputed independently with the
  official `money_saved` and `best_threshold`;
- the focus model and sort control drive the right content;
- the simulator rejects a risky applicant with warnings, reason codes and a probability, and explains why reason codes
  are unavailable for unsupported models;
- missing and starter-style (incompatible) artifacts show the fix-it screen;
- the two Executive Overview selection cards show the right models and figures for every interest / policy combination
  (card 1 stays the 0.50 champion even under the business policy) and the exact methodology note
  (`tests/test_selection_cards.py`), and the accuracy-vs-recall labels never overlap a label or bubble at three plot widths.

`tests/test_nested_cv.py` runs the real nested-CV code on a reduced sample and checks that outer folds partition the
training rows, the selection never sees outer-validation or holdout rows, thresholds come from inner out-of-fold scores
only, preprocessing is fitted on outer-training rows only, results are reproducible and aggregates are correct; it also
checks the stored full run.

`QUICKLOAN_MODEL_PATH` / `QUICKLOAN_EVAL_PATH` override the artifact locations, which the failure-screen tests use.

## Artifact contract — `loan_model.joblib` and `loan_model_leaderboard_champion.joblib`

A dict with `schema_version` (`quickloan-artifact/1.2`), `model_name`, `family`, `display_name`, **`selected_on`**
(`training_cv` or `holdout`), `selection_rule`, `pipeline` (fitted; raw 10-column DataFrame → `predict_proba`),
**`model`** (starter-compatible wrapper: raw or one-hot input, identical predictions), `starter_compatible`,
`raw_features` (raw input schema), **`columns`** (the starter app's encoded schema: 9 numeric +
`home_ownership_{mortgage,own,rent}`), `categorical_levels`, `engineered_features`,
`observed_ranges`, `thresholds` (`benchmark` 0.50, `business` at 20 %, **`business_by_interest`** {5: t, …, 50: t},
`business_rule`), `interest_assumption`, `score_type` (decision basis, `probability_available`, `probability_source`),
**`calibrated_pipeline`** (or `None`), **`calibration`** (method, justification, OOF and holdout Brier/ECE), `metrics`
(`cv`, `test_benchmark`, `test_business`, **`nested_cv`** summary, **`threshold_stability`**),
**`selection_methodology`** (procedure, assessment, final selection) and `training` (split settings, row counts, library versions, dataset
SHA-256, timestamp, hyperparameters; versions include pandas, numpy, scipy, scikit-learn, xgboost, joblib, cloudpickle,
pyarrow, shap, streamlit). The positive class is `default = 1`; inputs are the 10 raw columns in `raw_features` order. `quickloan.artifacts.load_model_artifact()` validates all of it. Both files are
gzip-compressed and self-contained (written by `quickloan.artifacts.save_portable`); plain `joblib.load` reads them.

```python
import joblib, pandas as pd
art = joblib.load("loan_model.joblib")
row = pd.DataFrame([{...10 raw fields...}])[art["raw_features"]]
score = art["pipeline"].predict_proba(row)[0, 1]
reject = score >= art["thresholds"]["business_by_interest"][20]     # or any interest 5–50
```

## Reproducibility

Fixed `random_state=1` everywhere (split, CV folds, every estimator); the nested-CV outer folds use seed 2 so they
differ from the inner folds. Running `train_models.py` and executing the
notebook give identical scores for all ten models (maximum difference 0.0) and identical leaderboard CSVs. The
artifacts record the dataset hash and library versions, and every library is pinned in `requirements.txt` (Python
3.13). Model files are tied to the scikit-learn / XGBoost versions that created them: the dashboard warns when the
installed versions differ, and `python train_models.py` rebuilds everything for the current environment.

## Limitations

- **Small holdout.** One 1,500-loan holdout with 188 defaulters. The bootstrap shows the top five models are
  statistically tied, and it varies the test sample only, not training.
- **Champion selection optimism.** The leaderboard champion is picked on that same holdout, so its figure is a
  model-ranking outcome, not an independent estimate of deployment performance.
- **Noisy nested CV.** Nested cross-validation covers training variation, but its five outer folds of 900 loans
  (113 defaulters each) are themselves noisy (sd AED 0.30M per 1,000 loans).
- **Simplified payoff.** Full loss on default, foregone interest as a flat share of principal, no recoveries, fees,
  operating or funding costs.
- **Scores and explanations.** Only the deployed model's score is presented as a probability. Predictions are
  unreliable outside the observed input ranges. Permutation importance and reason codes are associative, not causal.
- **Fairness.** The fairness check is a diagnostic: removing a variable does not remove its proxies. A real lender
  would need legal, fairness, model-risk and regulatory validation.
- **Environment verification.** The Windows route (Python 3.13.17) was verified with a CPython 3.13.17 interpreter
  on Linux plus a check that every package has a Windows wheel. It was not executed on a physical Windows
  machine.

## Screenshots

`docs/screenshots/` holds captures of the running dashboard: the overview with both model-selection cards, the
leaderboard with confidence intervals, ranking uncertainty and the accuracy-vs-recall chart, financial performance,
diagnostics with calibration, the simulator with reason codes, the fairness check, the threshold-by-interest chart,
the feature-engineering study, nested cross-validation (A / B / C and per fold), and the per-candidate comparison with
threshold stability.
