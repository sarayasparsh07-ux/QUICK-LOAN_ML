"""Reproduce the full QuickLoan bake-off from loans.csv and export every artifact.

    python train_models.py                 # uses ./loans.csv (or loans(1).csv)
    python train_models.py --data path/to/loans.csv

Outputs
    loan_model.joblib                        deployed pipeline (chosen on training CV) + artifact contract
    loan_model_leaderboard_champion.joblib   holdout leaderboard winner, same contract (the brief's "best model")
    model_leaderboard.csv                    all ten models on the common holdout at 0.50, with bootstrap CIs
    model_leaderboard_business.csv           same models at their CV-chosen business thresholds (separate report)
    artifacts/evaluation_results.joblib      predictions, curves and studies the dashboard reads
"""
from __future__ import annotations

import argparse

import pandas as pd

from quickloan import artifacts, config as C, training

LB_COLS = ["rank", "model", "family", "accuracy", "recall", "precision", "f1", "specificity",
           "balanced_accuracy", "roc_auc", "pr_auc", "brier", "tp", "fp", "fn", "tn",
           "loans_saved", "interest_lost", "money_saved", "missed_default_principal"]
CI_COLS = ["money_ci_low", "money_ci_high", "p_best", "gap_ci_low", "gap_ci_high", "distinguishable"]


def export(out: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write every artifact. Called by main() and by the notebook's export section."""
    # model artifacts are self-contained (load without this project's code - e.g. next to the starter app.py)
    artifacts.save_portable(training.build_model_artifact(out, "deployed"), C.MODEL_ARTIFACT)
    artifacts.save_portable(training.build_model_artifact(out, "champion"), C.CHAMPION_ARTIFACT)
    artifacts.save(training.build_eval_results(out), C.EVAL_ARTIFACT)

    lb = out["leaderboard"][LB_COLS + CI_COLS].copy()
    lb.insert(3, "threshold", C.BENCHMARK_THRESHOLD)
    lb["interest_assumption"] = C.DEFAULT_INTEREST
    lb["bootstrap_resamples"] = out["bootstrap"]["B"]
    lb.to_csv(C.LEADERBOARD_CSV, index=False, float_format="%.6f")

    biz = out["leaderboard_business"][LB_COLS].copy()
    thr = out["cv_summary"].set_index("model")["business_threshold"]
    biz.insert(3, "threshold", biz["model"].map(thr))
    biz["interest_assumption"] = C.DEFAULT_INTEREST
    biz.to_csv(C.ROOT / "model_leaderboard_business.csv", index=False, float_format="%.6f")
    return lb, biz


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="path to loans.csv")
    args = ap.parse_args(argv)

    out = training.run(args.data)
    lb, _ = export(out)

    with pd.option_context("display.width", 180, "display.max_columns", 20):
        print("\nHoldout leaderboard (threshold 0.50, interest 20 %, 95 % bootstrap CI)")
        print(lb[["rank", "model", "accuracy", "recall", "money_saved", "money_ci_low", "money_ci_high", "p_best"]]
              .to_string(index=False, formatters={c: "AED {:,.0f}".format for c in
                                                   ["money_saved", "money_ci_low", "money_ci_high"]}
                         | {"p_best": "{:.1%}".format}))
    print(f"\nDeployed model: {out['deployed']}  ->  {C.MODEL_ARTIFACT.name}")
    print(f"Leaderboard champion: {out['champion']}  ->  {C.CHAMPION_ARTIFACT.name}")
    cal = out["calibration"]
    print(f"Calibration ({cal['method']}): justified={cal['justified']}  holdout ECE "
          f"{cal['holdout']['raw_ece']:.4f} -> {cal['holdout']['cal_ece']:.4f}")


if __name__ == "__main__":
    main()
