"""
ONE source of truth for experimental results.

Every federated run executed by the experiment scripts is appended, as a
single JSON line, to ``results/raw/runs.jsonl``.  A run is identified by a
hash of (config, seed, holdout); re-requesting the same run re-uses the
stored record instead of re-running or copying numbers around.

Everything else -- summary CSVs (results/summary/), figures
(results/figures/) and the Streamlit dashboard -- is DERIVED from this file
through the functions below.  No number is typed in by hand anywhere.
"""
import hashlib
import json
import os
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

import config

RUNS_PATH = os.path.join(config.RAW_RESULTS_DIR, "runs.jsonl")
SELECTION_PATH = os.path.join(config.SUMMARY_DIR, "selected_config.json")
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                 "f1": "F1", "roc_auc": "ROC-AUC"}

# Fields of RunConfig that describe the model/training (not the component
# switches / threat model) -- used to recognise "the same configuration".
HP_FIELDS = ["rounds", "local_epochs", "lr", "hidden", "dropout", "knn_k",
             "anomaly_threshold", "clip_threshold", "noise_scale"]


def run_key(cfg: Dict, seed: int, holdout: str) -> str:
    blob = json.dumps({"cfg": cfg, "seed": seed, "holdout": holdout}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# raw store
# --------------------------------------------------------------------------
_CACHE: Dict[str, Dict] = {}
_CACHE_MTIME: Optional[float] = None


def load_runs(path: Optional[str] = None) -> List[Dict]:
    global _CACHE, _CACHE_MTIME
    path = path or RUNS_PATH
    if not os.path.exists(path):
        return []
    mtime = (path, os.path.getmtime(path), os.path.getsize(path))
    if _CACHE_MTIME == mtime:
        return list(_CACHE.values())
    runs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                runs[r["key"]] = r  # last write wins (identical by construction)
    _CACHE, _CACHE_MTIME = runs, mtime
    return list(runs.values())


def append_run(record: Dict, path: Optional[str] = None) -> None:
    path = path or RUNS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def get_or_run(cfg, seed: int, holdout: str, runner: Callable, stage: str) -> Dict:
    """Return the stored run for (cfg, seed, holdout) or execute it."""
    cfgd = cfg.as_dict()
    key = run_key(cfgd, seed, holdout)
    existing = {r["key"]: r for r in load_runs()}
    if key in existing:
        return existing[key]
    rec = runner(cfg, seed, holdout=holdout)
    rec["key"] = key
    rec["stage"] = stage
    append_run(rec)
    return rec


# --------------------------------------------------------------------------
# selectors / summaries
# --------------------------------------------------------------------------
def select(runs: Iterable[Dict], holdout: Optional[str] = None, seeds: Optional[Iterable[int]] = None,
           **cfg_equals) -> List[Dict]:
    seeds = set(seeds) if seeds is not None else None
    out = []
    for r in runs:
        if holdout is not None and r["holdout"] != holdout:
            continue
        if seeds is not None and r["seed"] not in seeds:
            continue
        c = r["config"]
        if all(c.get(k) == v for k, v in cfg_equals.items()):
            out.append(r)
    return sorted(out, key=lambda r: r["seed"])


def metrics_at_round(r: Dict, rnd: Optional[int] = None) -> Dict:
    """Metrics of a run after `rnd` rounds (None = final)."""
    if rnd is None or rnd == len(r["history"]):
        return {m: r["final"][m] for m in METRICS}
    h = r["history"][rnd - 1]
    return {m: h[m] for m in METRICS}


def mean_std(runs: List[Dict], rnd: Optional[int] = None) -> Dict:
    out = {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs]}
    for m in METRICS:
        v = np.array([metrics_at_round(r, rnd)[m] for r in runs], dtype=float)
        out[m + "_mean"] = float(np.nanmean(v)) if len(v) else float("nan")
        out[m + "_std"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
    return out


def curve(runs: List[Dict], key: str):
    """Per-round mean and std of `key` across runs."""
    a = np.array([[h.get(key, np.nan) for h in r["history"]] for r in runs], dtype=float)
    return np.nanmean(a, 0), (np.nanstd(a, 0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1]))


def load_selection() -> Optional[Dict]:
    path = SELECTION_PATH
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def runs_dataframe(runs: Iterable[Dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        row = {"key": r["key"], "stage": r.get("stage"), "holdout": r["holdout"], "seed": r["seed"],
               "timestamp": r.get("timestamp")}
        row.update({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r["config"].items()})
        row.update({m: r["final"][m] for m in METRICS})
        row.update({"client_macro_" + m: r["final"].get("client_macro_" + m) for m in METRICS})
        row.update({"loss": r["final"].get("loss"), "full_defense_rounds": r["full_defense_rounds"],
                    "ckks_max_abs_err": r.get("ckks_max_abs_err_overall"),
                    "total_s": r["timing"]["total_s"],
                    "aggregate_s": r["timing"]["aggregate_s"], "encrypt_s": r["timing"]["encrypt_s"],
                    "decrypt_s": r["timing"]["decrypt_s"], "local_train_s": r["timing"]["local_train_s"]})
        rows.append(row)
    return pd.DataFrame(rows)
