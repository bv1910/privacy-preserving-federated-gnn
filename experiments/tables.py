"""
Derived tables.  Every function here reads results/raw/runs.jsonl (via
results_io) and computes its numbers on the fly; the summary CSVs, the
figures and the dashboard all call these same functions.

    python -m experiments.tables     -> (re)writes results/summary/*.csv
"""
import json
import os
from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

import config
from experiments import results_io as rio
from experiments.pipeline_runner import RunConfig, original_config

M = rio.METRICS


def _runs_for(cfg: RunConfig, holdout: str, seeds) -> List[Dict]:
    want = {rio.run_key(cfg.as_dict(), s, holdout) for s in seeds}
    return sorted([r for r in rio.load_runs() if r["key"] in want], key=lambda r: r["seed"])


def _final_seeds():
    from experiments.final_experiments import FINAL_SEEDS
    present = {r["seed"] for r in rio.load_runs() if r["holdout"] == "test"}
    return [s for s in FINAL_SEEDS if s in present]


def _row(name, runs, extra=None):
    ms = rio.mean_std(runs)
    row = {"configuration": name, "n_seeds": ms["n_seeds"]}
    for m in M:
        row[m + "_mean"] = ms[m + "_mean"]
        row[m + "_std"] = ms[m + "_std"]
    row["full_defense_rounds_mean"] = float(np.mean([r["full_defense_rounds"] for r in runs])) if runs else np.nan
    if extra:
        row.update(extra)
    return row


# --------------------------------------------------------------------------
def selected_cfg() -> Optional[RunConfig]:
    sel = rio.load_selection()
    return RunConfig(**sel["selected_config"]) if sel else None


def ckks_vs_plaintext(which: str = "selected") -> pd.DataFrame:
    """Per-metric CKKS vs plaintext on the same test seeds, with paired diff."""
    seeds = _final_seeds()
    cfg = selected_cfg() if which == "selected" else original_config()
    if cfg is None or not seeds:
        return pd.DataFrame()
    ck = _runs_for(cfg, "test", seeds)
    pl = _runs_for(replace(cfg, use_ckks=False), "test", seeds)
    common = sorted({r["seed"] for r in ck} & {r["seed"] for r in pl})
    ck = [r for r in ck if r["seed"] in common]
    pl = [r for r in pl if r["seed"] in common]
    rows = []
    for m in M:
        a = np.array([r["final"][m] for r in ck]); b = np.array([r["final"][m] for r in pl])
        d = a - b
        if len(d) > 1 and np.any(d != 0):
            p = float(stats.ttest_rel(a, b).pvalue)
        else:
            p = float("nan")
        rows.append({"metric": m, "ckks_mean": a.mean(), "ckks_std": a.std(ddof=1) if len(a) > 1 else 0,
                     "plaintext_mean": b.mean(), "plaintext_std": b.std(ddof=1) if len(b) > 1 else 0,
                     "mean_paired_diff": d.mean(), "max_abs_paired_diff": np.abs(d).max(),
                     "seeds_with_any_difference": int(np.sum(d != 0)), "paired_t_p_value": p,
                     "n_seeds": len(common)})
    return pd.DataFrame(rows)


def per_seed_final() -> pd.DataFrame:
    seeds = _final_seeds()
    out = []
    for label, cfg in (("Selected", selected_cfg()), ("Original baseline", original_config())):
        if cfg is None:
            continue
        for arm, c in (("CKKS", cfg), ("Plaintext", replace(cfg, use_ckks=False))):
            for r in _runs_for(c, "test", seeds):
                row = {"model": label, "aggregation": arm, "seed": r["seed"]}
                row.update({m: r["final"][m] for m in M})
                row.update({"client_macro_" + m: r["final"]["client_macro_" + m] for m in M})
                row.update({k: r["final"][k] for k in ("tn", "fp", "fn", "tp", "n")})
                row["full_defense_rounds"] = r["full_defense_rounds"]
                row["ckks_max_abs_err"] = r.get("ckks_max_abs_err_overall")
                row["total_s"] = r["timing"]["total_s"]
                out.append(row)
    return pd.DataFrame(out)


def before_after() -> pd.DataFrame:
    seeds = _final_seeds()
    rows = []
    for label, cfg in (("Original config", original_config()), ("Selected config", selected_cfg())):
        if cfg is None:
            continue
        for arm, c in (("CKKS", cfg), ("Plaintext", replace(cfg, use_ckks=False))):
            rs = _runs_for(c, "test", seeds)
            row = _row(f"{label} - {arm}", rs)
            for m in M:
                v = [r["final"]["client_macro_" + m] for r in rs]
                row["client_macro_" + m + "_mean"] = float(np.nanmean(v)) if v else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def ablation() -> pd.DataFrame:
    from experiments.final_experiments import ABLATIONS
    cfg, seeds = selected_cfg(), _final_seeds()
    if cfg is None:
        return pd.DataFrame()
    return pd.DataFrame([_row(n, _runs_for(replace(cfg, **kw), "test", seeds)) for n, kw in ABLATIONS.items()])


def attacks() -> pd.DataFrame:
    from experiments.final_experiments import attack_variants
    cfg, seeds = selected_cfg(), _final_seeds()
    if cfg is None:
        return pd.DataFrame()
    v = attack_variants(cfg)
    clean_on = rio.mean_std(_runs_for(v[("none", "on")], "test", seeds))
    rows = []
    for (atk, d), c in v.items():
        rs = _runs_for(c, "test", seeds)
        if not rs:
            continue
        row = _row(atk, rs, {"attack": atk, "defense": "ON" if d == "on" else "OFF"})
        for m in M:
            row[m + "_drop_vs_clean"] = clean_on[m + "_mean"] - row[m + "_mean"]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.drop(columns=["configuration"]) if not df.empty else df


def attack_defense_gain() -> pd.DataFrame:
    df = attacks()
    if df.empty:
        return df
    rows = []
    for atk, g in df.groupby("attack", sort=False):
        on = g[g.defense == "ON"].iloc[0]; off = g[g.defense == "OFF"].iloc[0]
        rows.append({"attack": atk, **{m + "_gain_defense_on_minus_off": on[m + "_mean"] - off[m + "_mean"] for m in M},
                     "full_defense_rounds_mean_on": on["full_defense_rounds_mean"]})
    return pd.DataFrame(rows)


def sweep(stage_seeds=None, holdout="val") -> pd.DataFrame:
    """Every validation config x round choice, CKKS and plaintext side by side."""
    from experiments.sweep import ROUND_CHOICES, S1_SEEDS
    seeds = stage_seeds or S1_SEEDS
    runs = [r for r in rio.load_runs() if r["holdout"] == holdout and r["seed"] in seeds
            and r["config"]["use_multi_agent"] and r["config"]["attack_type"] == "scale_gaussian"]
    groups = {}
    for r in runs:
        hp = {k: r["config"][k] for k in rio.HP_FIELDS if k != "rounds"}
        groups.setdefault((json.dumps(hp, sort_keys=True), r["config"]["use_ckks"]), []).append(r)
    rows = []
    for (hpk, ck), rs in groups.items():
        if ck is False or len({r["seed"] for r in rs}) < len(seeds):
            continue
        pl = groups.get((hpk, False), [])
        hp = json.loads(hpk)
        for R in ROUND_CHOICES:
            if R > rs[0]["config"]["rounds"]:
                continue
            a = rio.mean_std(rs, R)
            row = {**hp, "rounds": R, "n_seeds": a["n_seeds"],
                   "stage": rs[0].get("stage")}
            for m in M:
                row["ckks_" + m + "_mean"] = a[m + "_mean"]
                row["ckks_" + m + "_std"] = a[m + "_std"]
            if len({r["seed"] for r in pl}) >= len(seeds):
                b = rio.mean_std(pl, R)
                for m in M:
                    row["plain_" + m + "_mean"] = b[m + "_mean"]
            row["ckks_mean_total_s"] = float(np.mean([r["timing"]["total_s"] for r in rs]))
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["ckks_accuracy_mean", "ckks_f1_mean"], ascending=False).reset_index(drop=True)
    return df


def round_curves(cfg: RunConfig, holdout="test", seeds=None) -> pd.DataFrame:
    seeds = seeds or _final_seeds()
    rows = []
    for arm, c in (("CKKS", cfg), ("Plaintext", replace(cfg, use_ckks=False))):
        rs = _runs_for(c, holdout, seeds)
        if not rs:
            continue
        for key in M + ["loss", "train_loss", "ckks_max_abs_err", "ckks_rmse", "ckks_rel_l2_err",
                        "local_train_s", "encrypt_s", "aggregate_s", "decrypt_s", "update_norm", "anomaly_score"]:
            if key not in rs[0]["history"][0]:
                continue
            mu, sd = rio.curve(rs, key)
            for i, (a, b) in enumerate(zip(mu, sd), 1):
                rows.append({"aggregation": arm, "quantity": key, "round": i, "mean": a, "std": b, "n_seeds": len(rs)})
    return pd.DataFrame(rows)


def timing() -> pd.DataFrame:
    seeds = _final_seeds()
    rows = []
    for label, cfg in (("Selected", selected_cfg()), ("Original baseline", original_config())):
        if cfg is None:
            continue
        for arm, c in (("CKKS", cfg), ("Plaintext", replace(cfg, use_ckks=False))):
            rs = _runs_for(c, "test", seeds)
            if not rs:
                continue
            R = len(rs[0]["history"])
            row = {"model": label, "aggregation": arm, "rounds": R, "n_seeds": len(rs)}
            for k in ("total_s", "local_train_s", "encrypt_s", "aggregate_s", "decrypt_s", "eval_s"):
                v = [r["timing"][k] for r in rs]
                row[k + "_mean"] = float(np.mean(v)); row[k + "_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0
                row[k.replace("_s", "") + "_per_round_ms"] = 1000 * row[k + "_mean"] / R
            rows.append(row)
    return pd.DataFrame(rows)


def ckks_error_summary() -> pd.DataFrame:
    seeds = _final_seeds()
    cfg = selected_cfg()
    if cfg is None:
        return pd.DataFrame()
    rs = _runs_for(cfg, "test", seeds)
    allh = [h for r in rs for h in r["history"]]
    if not allh:
        return pd.DataFrame()
    out = {}
    for k in ("ckks_max_abs_err", "ckks_mean_abs_err", "ckks_rmse", "ckks_rel_l2_err"):
        v = np.array([h[k] for h in allh])
        out[k + "_mean"] = v.mean(); out[k + "_max"] = v.max()
    out["n_round_measurements"] = len(allh)
    out["ciphertexts_per_client"] = allh[0]["n_ciphertexts_per_client"]
    out["poly_modulus_degree"] = config.POLY_MODULUS_DEGREE
    out["scale_bits"] = int(np.log2(config.CKKS_SCALE))
    out["qi_sizes"] = str(config.QI_SIZES)
    out["backend"] = config.CKKS_BACKEND
    return pd.DataFrame([out])


def defense_sensitivity() -> pd.DataFrame:
    from experiments.sweep import S1_SEEDS
    sel = rio.load_selection()
    if not sel:
        return pd.DataFrame()
    base = sel["selected_config"]
    rows = []
    for param, vals in (("anomaly_threshold", (0.3, 0.5, config.ANOMALY_THRESHOLD, 0.9)),
                        ("clip_threshold", (0.5, config.CLIP_THRESHOLD, 2.0, 5.0)),
                        ("noise_scale", (0.001, config.NOISE_SCALE, 0.05, 0.1))):
        for v in vals:
            c = RunConfig(**{**base, param: v})
            ck = _runs_for(c, "val", S1_SEEDS)
            pl = _runs_for(replace(c, use_ckks=False), "val", S1_SEEDS)
            if not ck:
                continue
            a = rio.mean_std(ck)
            row = {"parameter": param, "value": v, "is_default": v == getattr(original_config(), param), "n_seeds": len(ck)}
            for m in M:
                row["ckks_" + m + "_mean"] = a[m + "_mean"]; row["ckks_" + m + "_std"] = a[m + "_std"]
            if pl:
                b = rio.mean_std(pl)
                row["plain_accuracy_mean"] = b["accuracy_mean"]
            row["full_defense_rounds_mean"] = float(np.mean([r["full_defense_rounds"] for r in ck]))
            rows.append(row)
    return pd.DataFrame(rows)


def final_pooled_predictions(cfg: Optional[RunConfig] = None):
    """Concatenated test labels / malignant probabilities over all final seeds."""
    cfg = cfg or selected_cfg()
    rs = _runs_for(cfg, "test", _final_seeds())
    y = np.concatenate([r["predictions"]["y_true"] for r in rs]) if rs else np.array([])
    p = np.concatenate([r["predictions"]["p_malignant"] for r in rs]) if rs else np.array([])
    return y, p, rs


def write_all():
    os.makedirs(config.SUMMARY_DIR, exist_ok=True)
    outs = {
        "final_ckks_vs_plaintext.csv": ckks_vs_plaintext("selected"),
        "baseline_ckks_vs_plaintext.csv": ckks_vs_plaintext("baseline"),
        "final_per_seed.csv": per_seed_final(),
        "before_after.csv": before_after(),
        "ablation.csv": ablation(),
        "attack_defense.csv": attacks(),
        "attack_defense_gain.csv": attack_defense_gain(),
        "sweep_stage1_2.csv": sweep(),
        "defense_sensitivity.csv": defense_sensitivity(),
        "timing.csv": timing(),
        "ckks_error.csv": ckks_error_summary(),
        "all_runs.csv": rio.runs_dataframe(rio.load_runs()),
        "graph_homophily.csv": graph_homophily(),
    }
    from experiments.sweep import S3_SEEDS
    outs["sweep_stage3.csv"] = sweep(S3_SEEDS)
    if selected_cfg() is not None:
        outs["final_round_curves.csv"] = round_curves(selected_cfg())
    for fn, df in outs.items():
        df.to_csv(os.path.join(config.SUMMARY_DIR, fn), index=False)
        print(f"wrote results/summary/{fn} ({len(df)} rows)")


if __name__ == "__main__":
    write_all()


def graph_homophily(ks=(2, 3, 4, 6, 8), seeds=None) -> pd.DataFrame:
    """Measured data property (no model involved): fraction of k-NN edges
    joining two patients with the same diagnosis, in train vs held-out
    graphs, on the final test seeds."""
    from client.data_loader import load_and_partition
    seeds = seeds or list(range(100, 110))
    rows = []
    for k in ks:
        for split in ("train", "test"):
            vals = []
            for s in seeds:
                for tr, te in load_and_partition(seed=s, k=k, holdout="test").values():
                    g = tr if split == "train" else te
                    a, b = g.edge_index
                    vals.append(float((g.y[a] == g.y[b]).float().mean()))
            rows.append({"knn_k": k, "graph": split, "edge_homophily_mean": float(np.mean(vals)),
                         "edge_homophily_std": float(np.std(vals, ddof=1)), "n_graphs": len(vals)})
    return pd.DataFrame(rows)
