"""
One-command reproduction of every experiment, table and figure.

    python -m experiments.run_experiments

  1. experiments.sweep              staged hyperparameter search (validation only)
  2. experiments.final_experiments  fresh test seeds: CKKS vs plaintext, ablation, attacks
  3. experiments.tables             results/summary/*.csv (derived from results/raw/runs.jsonl)
  4. experiments.make_plots         results/figures/*.png (derived from the same store)
  5. DP budget for the Gaussian sanitisation actually applied (results/summary/dp_budget.csv)

Runs already present in results/raw/runs.jsonl are re-used, so the command
is resumable.  Delete results/raw/runs.jsonl to recompute from scratch.
"""
import csv
import os
import sys

import config
from experiments import final_experiments, make_plots, sweep, tables
from experiments import results_io as rio
from privacy.dp_accountant import compute_dp_budget


def dp_budget():
    cfg = tables.selected_cfg()
    rs = tables._runs_for(cfg, "test", tables._final_seeds())
    rounds = [r["full_defense_rounds"] for r in rs]
    per_run = max(rounds) if rounds else 0
    b = compute_dp_budget(rounds_with_full_defense=per_run, sigma=cfg.noise_scale, sensitivity=cfg.clip_threshold)
    row = {**b.__dict__, "note": "rounds_composed = max Full-Defense rounds in any single final run "
           "(each run is one training job); basic composition."}
    with open(os.path.join(config.SUMMARY_DIR, "dp_budget.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    print("DP budget:", row)


if __name__ == "__main__":
    sys.argv = sys.argv[:1]
    sweep.stage1(); sweep.stage2(); sweep.stage3(); sweep.defense_sensitivity()
    final_experiments.main()
    tables.write_all()
    dp_budget()
    make_plots.main()
