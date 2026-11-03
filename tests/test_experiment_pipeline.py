"""Tests for the experiment/results pipeline added for the tuning study."""
import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

import config
from client.data_loader import load_and_partition, split_indices
from crypto.ckks_engine import CKKSEngine
from evaluation.metrics import binary_metrics, evaluate_federation
from experiments import results_io as rio
from experiments.pipeline_runner import RunConfig, run_config, run_pipeline
from models.graphsage_model import GraphSAGENet

FAST = RunConfig(rounds=2, local_epochs=1)


# ----------------------------------------------------------------- metrics
def test_binary_metrics_hand_computed():
    # 1 = malignant (positive).  TP=2, FN=1, FP=1, TN=2
    y = [1, 1, 1, 0, 0, 0]
    pred = [1, 1, 0, 1, 0, 0]
    prob = [.9, .8, .4, .6, .2, .1]
    m = binary_metrics(y, prob, pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 2)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 * (2 / 3) * (2 / 3) / (4 / 3))
    # AUC: positives {.9,.8,.4} vs negatives {.6,.2,.1}: 8 of 9 pairs ranked right
    assert m["roc_auc"] == pytest.approx(8 / 9)


def test_pooled_metrics_use_malignant_as_positive():
    torch.manual_seed(0)
    g = load_and_partition(seed=42)
    model = GraphSAGENet()
    out = evaluate_federation(model, [te for _, te in g.values()])
    y = np.array(out["_y_true"])
    n_malignant = sum(int((te.y == 0).sum()) for _, te in g.values())
    assert y.sum() == n_malignant  # label 0 (M) mapped to positive 1
    assert out["tp"] + out["fn"] == n_malignant
    assert out["n"] == sum(te.num_nodes for _, te in g.values())
    assert 0 <= out["accuracy"] <= 1 and "client_macro_accuracy" in out


# ----------------------------------------------------------------- splits
def test_train_val_test_disjoint_and_test_unchanged():
    idx = split_indices(seed=123)
    all_test = []
    for cid, d in idx.items():
        tr, va, te = set(d["train"]), set(d["val"]), set(d["test"])
        assert not (tr & va) and not (tr & te) and not (va & te)
        all_test += list(te)
    assert len(all_test) == len(set(all_test))
    # holdout="test" graphs use train+val for training and the same test nodes
    g_test = load_and_partition(seed=123, holdout="test")
    g_val = load_and_partition(seed=123, holdout="val")
    for cid, d in idx.items():
        assert g_test[cid][1].num_nodes == len(d["test"])
        assert g_test[cid][0].num_nodes == len(d["train"]) + len(d["val"])
        assert g_val[cid][0].num_nodes == len(d["train"])
        assert g_val[cid][1].num_nodes == len(d["val"])


# ----------------------------------------------------------------- aggregation
def test_ckks_aggregation_matches_plaintext_mean_multi_chunk():
    eng = CKKSEngine()
    rng = np.random.default_rng(0)
    vecs = [rng.normal(0, 0.1, eng.slots + 123) for _ in range(5)]  # forces 2 ciphertexts
    agg = eng.decrypt_vector(eng.aggregate_encrypted([eng.encrypt_vector(v) for v in vecs]), len(vecs[0]))
    assert np.max(np.abs(agg - np.mean(vecs, axis=0))) < 1e-4


def test_ckks_path_really_used_and_error_recorded():
    r = run_config(FAST, seed=7, holdout="val")
    assert all("ckks_max_abs_err" in h for h in r["history"])
    assert all(h["encrypt_s"] > 0 for h in r["history"])
    assert r["ckks_max_abs_err_overall"] < 1e-3
    p = run_config(replace(FAST, use_ckks=False), seed=7, holdout="val")
    assert all("ckks_max_abs_err" not in h and h["encrypt_s"] == 0 for h in p["history"])


def test_ckks_and_plaintext_share_everything_but_aggregation():
    a = run_config(FAST, seed=11, holdout="val")
    b = run_config(replace(FAST, use_ckks=False), seed=11, holdout="val")
    assert a["predictions"]["y_true"] == b["predictions"]["y_true"]  # same split
    # CKKS error ~1e-6 -> probabilities agree to well within 1e-3
    assert np.max(np.abs(np.array(a["predictions"]["p_malignant"]) - np.array(b["predictions"]["p_malignant"]))) < 1e-3


# ----------------------------------------------------------------- ablation flags
def test_no_multi_agent_bypasses_defense():
    r = run_config(replace(FAST, use_multi_agent=False), seed=3, holdout="val")
    assert {h["path"] for h in r["history"]} == {"no_defense"} and r["full_defense_rounds"] == 0


def test_clipping_flag_disables_only_clipping():
    always = replace(FAST, anomaly_threshold=0.0, clip_threshold=0.01, noise_scale=0.0, use_ckks=False)
    clipped = run_config(always, seed=5, holdout="val")
    unclipped = run_config(replace(always, use_clipping=False), seed=5, holdout="val")
    assert all(h["path"] == "full_defense" for h in clipped["history"])
    assert all(h["update_norm"] == pytest.approx(0.01, rel=1e-4) for h in clipped["history"])
    assert all(h["update_norm"] > 0.05 for h in unclipped["history"])


def test_noise_flag_disables_only_noise():
    always = replace(FAST, anomaly_threshold=0.0, clip_threshold=1e9, noise_scale=0.5, use_ckks=False)
    noisy = run_config(always, seed=5, holdout="val")
    clean = run_config(replace(always, use_noise=False), seed=5, holdout="val")
    plain = run_config(replace(always, use_multi_agent=False), seed=5, holdout="val")
    assert noisy["history"][0]["update_norm"] > 10 * clean["history"][0]["update_norm"]
    # no clip (huge threshold) + no noise == passing the update through unchanged
    assert clean["history"][0]["update_norm"] == pytest.approx(plain["history"][0]["update_norm"])


def test_fpe_is_lossless_so_ablating_it_changes_nothing():
    a = run_config(replace(FAST, use_ckks=False), seed=9, holdout="val")
    b = run_config(replace(FAST, use_ckks=False, use_fpe=False), seed=9, holdout="val")
    assert a["final"]["accuracy"] == b["final"]["accuracy"]
    assert a["predictions"]["p_malignant"] == b["predictions"]["p_malignant"]


def test_hyperparameters_reach_clients_via_fpe():
    from crypto.fpe_engine import FPEEngine
    enc = FPEEngine().encrypt_hyperparams(learning_rate=0.003, local_epochs=10)
    assert FPEEngine().decrypt_hyperparams(enc) == {"learning_rate": 0.003, "local_epochs": 10}


# ----------------------------------------------------------------- serialization / store
def test_run_records_serialize_without_circular_reference():
    r = run_config(FAST, seed=1, holdout="val")
    s = json.dumps(r)
    assert json.loads(s)["history"][-1]["round"] == FAST.rounds
    legacy = run_pipeline(seed=1, num_rounds=2, holdout="val", local_epochs=1)
    json.dumps(legacy)  # previously raised "Circular reference detected"


def test_results_store_get_or_run_reuses(tmp_path, monkeypatch):
    monkeypatch.setattr(rio, "RUNS_PATH", str(tmp_path / "runs.jsonl"))
    calls = []

    def runner(cfg, seed, holdout):
        calls.append(seed)
        return run_config(cfg, seed, holdout=holdout)

    a = rio.get_or_run(FAST, 21, "val", runner, "unit")
    b = rio.get_or_run(FAST, 21, "val", runner, "unit")
    assert calls == [21] and a["key"] == b["key"]
    runs = rio.load_runs()
    assert len(runs) == 1
    ms = rio.mean_std(runs)
    assert ms["accuracy_mean"] == pytest.approx(a["final"]["accuracy"])
    df = rio.runs_dataframe(runs)
    assert set(rio.METRICS) <= set(df.columns)


def test_tables_and_dashboard_loader_on_store(tmp_path, monkeypatch):
    """Dashboard data come from experiments.tables; check they read the store."""
    from experiments import tables as T
    from experiments import final_experiments as F
    monkeypatch.setattr(rio, "RUNS_PATH", str(tmp_path / "runs.jsonl"))
    monkeypatch.setattr(rio, "SELECTION_PATH", str(tmp_path / "sel.json"))
    monkeypatch.setattr(F, "FINAL_SEEDS", [100, 101])
    sel = FAST
    with open(rio.SELECTION_PATH, "w") as f:
        json.dump({"selected_config": {k: getattr(sel, k) for k in rio.HP_FIELDS}}, f)
    for s in (100, 101):
        for c in (sel, replace(sel, use_ckks=False)):
            rio.get_or_run(c, s, "test", run_config, "unit")
    cv = T.ckks_vs_plaintext("selected")
    assert list(cv.metric) == rio.METRICS and (cv.n_seeds == 2).all()
    stored = [r for r in rio.load_runs() if r["config"]["use_ckks"]]
    assert cv.set_index("metric").loc["accuracy", "ckks_mean"] == pytest.approx(
        np.mean([r["final"]["accuracy"] for r in stored]))
