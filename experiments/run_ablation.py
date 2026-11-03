"""Component ablation (A-F) on the TEST split for the configuration selected
on validation.  All runs are stored in results/raw/runs.jsonl (no circular
references: each run record holds a fresh per-round history list) and the
table is derived from that store.

Usage:  python -m experiments.run_ablation [--seeds 10]
Output: results/summary/ablation.csv
"""
import argparse
import os
from dataclasses import replace

import config
from experiments import results_io as rio
from experiments import tables as T
from experiments.final_experiments import ABLATIONS, FINAL_SEEDS, selected_config
from experiments.pipeline_runner import run_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(FINAL_SEEDS))
    a = ap.parse_args()
    base = selected_config()
    for name, kw in ABLATIONS.items():
        for s in FINAL_SEEDS[: a.seeds]:
            rio.get_or_run(replace(base, **kw), s, "test", run_config, "final")
        print("done:", name, flush=True)
    df = T.ablation()
    df.to_csv(os.path.join(config.SUMMARY_DIR, "ablation.csv"), index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
