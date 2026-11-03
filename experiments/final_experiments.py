"""
Final experiments on the TEST split with FRESH seeds (never used in the
hyperparameter search), using the configuration selected on validation by
experiments/sweep.py (results/summary/selected_config.json).

  1. Baseline (original project config) -- CKKS vs plaintext
  2. Selected config                      -- CKKS vs plaintext
  3. Component ablation of the selected config
  4. Every implemented attack, defense ON vs OFF (+ no-attacker reference)

All runs go to results/raw/runs.jsonl (identical runs are re-used, never
re-typed).  Usage:  python -m experiments.final_experiments [--seeds 10]
"""
import argparse
from dataclasses import replace

import config
from attacks.attack_types import ALL_ATTACKS
from experiments import results_io as rio
from experiments.pipeline_runner import RunConfig, original_config, run_config

FINAL_SEEDS = list(range(100, 110))

ABLATIONS = {
    "A. Full Model (CKKS)": {},
    "B. Without CKKS (plaintext FedAvg)": {"use_ckks": False},
    "C. Without Multi-Agent Defense": {"use_multi_agent": False},
    "D. Without L2 Clipping": {"use_clipping": False},
    "E. Without Gaussian Sanitization": {"use_noise": False},
    "F. Without FPE": {"use_fpe": False},
}


def baseline_config() -> RunConfig:
    """The project's original configuration (config.py defaults)."""
    return original_config()


def selected_config() -> RunConfig:
    sel = rio.load_selection()
    if sel is None:
        raise SystemExit("No selected_config.json -- run `python -m experiments.sweep` first.")
    return RunConfig(**sel["selected_config"])


def attack_variants(base: RunConfig):
    """{(attack, 'on'|'off'): RunConfig}.  'none' = no malicious clients."""
    out = {}
    for atk in ["none"] + ALL_ATTACKS:
        for d in ("on", "off"):
            kw = {"use_multi_agent": d == "on"}
            if atk == "none":
                kw["malicious_ids"] = []
            else:
                kw["attack_type"] = atk
            out[(atk, d)] = replace(base, **kw)
    return out


def all_final_configs(seeds_n: int = len(FINAL_SEEDS)):
    base = selected_config()
    cfgs = {"baseline_ckks": baseline_config(), "baseline_plain": replace(baseline_config(), use_ckks=False)}
    for name, kw in ABLATIONS.items():
        cfgs["ablation::" + name] = replace(base, **kw)
    for (atk, d), c in attack_variants(base).items():
        cfgs[f"attack::{atk}::{d}"] = c
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(FINAL_SEEDS))
    a = ap.parse_args()
    seeds = FINAL_SEEDS[: a.seeds]
    cfgs = all_final_configs()
    for i, (name, cfg) in enumerate(cfgs.items(), 1):
        rs = [rio.get_or_run(cfg, s, "test", run_config, "final") for s in seeds]
        ms = rio.mean_std(rs)
        print(f"[{i}/{len(cfgs)}] {name}: acc={ms['accuracy_mean']:.4f}±{ms['accuracy_std']:.4f} "
              f"f1={ms['f1_mean']:.4f} auc={ms['roc_auc_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
