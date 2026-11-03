"""
Staged hyperparameter search -- VALIDATION SPLIT ONLY (test never touched).

Selection protocol (fixed before any result was seen):
  Stage 1  lr x local_epochs x knn_k grid, 40 rounds, val seeds 1000-1002.
           Validation metrics are recorded every round, so each run also
           yields the result for rounds in ROUND_CHOICES (stopping earlier is
           exactly identical to a shorter run: nothing depends on the total).
  Stage 2  one-factor variations (knn_k, hidden, dropout) around the
           stage-1 winner, same seeds.
  Stage 3  top-N (config, rounds) candidates re-run on FRESH val seeds
           2000-2004; the winner is chosen on those seeds only.
  Defense  anomaly_threshold / clip_threshold / noise_scale sensitivity
           around the winner (reported only; defaults are kept for the final
           model so the security configuration is not weakened for accuracy).

Score = mean validation accuracy of the CKKS arm (tie-break: mean F1).
Every configuration is also run as plaintext FedAvg with identical seeds.

Usage:  python -m experiments.sweep            (all stages)
        python -m experiments.sweep --stage 1  (one stage)
"""
import argparse
import itertools
import json
import os
import time
from dataclasses import replace

import numpy as np

import config
from experiments.pipeline_runner import RunConfig, original_config, run_config
from experiments import results_io as rio

S1_SEEDS = [1000, 1001, 1002]
S3_SEEDS = [2000, 2001, 2002, 2003, 2004]
ROUND_CHOICES = [8, 15, 20, 30, 40]
MAX_ROUNDS = 40
LRS = [0.003, 0.005, 0.01, 0.02, 0.05]
EPOCHS = [3, 5, 8, 10]
KS = [3, 6]
N_FINALISTS = 6


def _run_both(cfg: RunConfig, seeds, stage):
    for s in seeds:
        for ck in (True, False):
            rio.get_or_run(replace(cfg, use_ckks=ck), s, "val", run_config, stage)


def _hp(cfg: RunConfig):
    return {k: getattr(cfg, k) for k in rio.HP_FIELDS}


def score_table(seeds, base_filter=None):
    """Mean val metrics of every stored val config at every round choice (CKKS arm)."""
    runs = [r for r in rio.load_runs() if r["holdout"] == "val" and r["seed"] in seeds
            and r["config"]["use_ckks"] and r["config"]["use_multi_agent"]
            and r["config"]["attack_type"] == "scale_gaussian"
            and r["config"]["malicious_ids"] == list(config.MALICIOUS_CLIENT_IDS)]
    groups = {}
    for r in runs:
        hp = {k: r["config"][k] for k in rio.HP_FIELDS if k != "rounds"}
        groups.setdefault(json.dumps(hp, sort_keys=True), []).append(r)
    rows = []
    for hpk, rs in groups.items():
        if len({r["seed"] for r in rs}) < len(seeds):
            continue
        hp = json.loads(hpk)
        if base_filter and not base_filter(hp):
            continue
        for R in ROUND_CHOICES:
            if R > rs[0]["config"]["rounds"]:
                continue
            ms = rio.mean_std(rs, R)
            rows.append({**hp, "rounds": R, **ms})
    rows.sort(key=lambda x: (-x["accuracy_mean"], -x["f1_mean"]))
    return rows


def defaults_filter(hp):
    return (hp["anomaly_threshold"] == config.ANOMALY_THRESHOLD and hp["clip_threshold"] == config.CLIP_THRESHOLD
            and hp["noise_scale"] == config.NOISE_SCALE)


def stage1():
    combos = list(itertools.product(LRS, EPOCHS, KS))
    for i, (lr, ep, k) in enumerate(combos, 1):
        t = time.time()
        _run_both(original_config(rounds=MAX_ROUNDS, lr=lr, local_epochs=ep, knn_k=k), S1_SEEDS, "sweep_stage1")
        print(f"[stage1 {i}/{len(combos)}] lr={lr} ep={ep} k={k} ({time.time()-t:.0f}s)", flush=True)


def best_stage1():
    return score_table(S1_SEEDS, defaults_filter)[0]


def stage2():
    b = best_stage1()
    print("stage-1 best:", {k: b[k] for k in ["lr", "local_epochs", "knn_k", "rounds", "accuracy_mean"]}, flush=True)
    base = RunConfig(rounds=MAX_ROUNDS, lr=b["lr"], local_epochs=b["local_epochs"], knn_k=b["knn_k"],
                     hidden=b["hidden"], dropout=b["dropout"])
    variants = [replace(base, knn_k=k) for k in (2, 4, 8)]
    variants += [replace(base, hidden=h) for h in (16, 64)]
    variants += [replace(base, dropout=d) for d in (0.0, 0.5)]
    for i, v in enumerate(variants, 1):
        _run_both(v, S1_SEEDS, "sweep_stage2")
        print(f"[stage2 {i}/{len(variants)}] k={v.knn_k} hidden={v.hidden} dropout={v.dropout}", flush=True)


def finalists():
    rows = score_table(S1_SEEDS, defaults_filter)
    seen, out = set(), []
    for r in rows:  # one (best) round choice per distinct hyperparameter set
        key = json.dumps({k: r[k] for k in rio.HP_FIELDS if k != "rounds"}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(r)
        if len(out) == N_FINALISTS:
            break
    return out


def stage3():
    cands = finalists()
    # Always include the original project configuration as a reference point.
    for c in cands:
        cfg = RunConfig(**{k: c[k] for k in rio.HP_FIELDS if k != "rounds"}, rounds=MAX_ROUNDS)
        _run_both(cfg, S3_SEEDS, "sweep_stage3")
        print(f"[stage3] lr={c['lr']} ep={c['local_epochs']} k={c['knn_k']} h={c['hidden']} d={c['dropout']}", flush=True)
    rows = []
    for c in cands:
        hp = {k: c[k] for k in rio.HP_FIELDS if k != "rounds"}
        rs = rio.select(rio.load_runs(), holdout="val", seeds=S3_SEEDS, use_ckks=True,
                        rounds=MAX_ROUNDS, use_multi_agent=True, attack_type="scale_gaussian", **hp)
        ms = rio.mean_std(rs, c["rounds"])
        rows.append({**hp, "rounds": c["rounds"], "stage1_accuracy_mean": c["accuracy_mean"], **ms})
    rows.sort(key=lambda x: (-x["accuracy_mean"], -x["f1_mean"]))
    winner = rows[0]
    sel = {
        "selected_config": {k: winner[k] for k in rio.HP_FIELDS},
        "selection_metric": "mean validation accuracy (CKKS arm) on fresh seeds, tie-break F1",
        "selection_seeds": S3_SEEDS,
        "stage1_seeds": S1_SEEDS,
        "finalists": rows,
        "note": "Selected on VALIDATION data only. Test split untouched during selection.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(rio.SELECTION_PATH, "w") as f:
        json.dump(sel, f, indent=2)
    print("SELECTED:", sel["selected_config"], "val acc", round(winner["accuracy_mean"], 4), flush=True)


def defense_sensitivity():
    sel = rio.load_selection()["selected_config"]
    base = RunConfig(**sel)
    variants = [replace(base, anomaly_threshold=a) for a in (0.3, 0.5, 0.9)]
    variants += [replace(base, clip_threshold=c) for c in (0.5, 2.0, 5.0)]
    variants += [replace(base, noise_scale=n) for n in (0.001, 0.05, 0.1)]
    for v in variants:
        _run_both(v, S1_SEEDS, "sweep_defense")
        print(f"[defense] thr={v.anomaly_threshold} clip={v.clip_threshold} noise={v.noise_scale}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "1", "2", "3", "defense"])
    a = ap.parse_args()
    if a.stage in ("all", "1"):
        stage1()
    if a.stage in ("all", "2"):
        stage2()
    if a.stage in ("all", "3"):
        stage3()
    if a.stage in ("all", "defense"):
        defense_sensitivity()
