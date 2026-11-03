"""
Parameterised version of main.py's federated loop, used by every
experiment script (sweep, final validation, ablation, attacks).

It orchestrates the EXISTING components exactly as main.py does --
GraphSAGE clients, CKKS encrypt -> homomorphic aggregate -> decrypt,
4-agent defense (Classifier -> Threshold -> Coordinator(clip + Gaussian
noise) -> Logger), FPE hyperparameter broadcast -- and only exposes knobs:

  * use_ckks          True: CKKS aggregation;  False: plaintext FedAvg
  * use_multi_agent   False: bypass the 4-agent screening + sanitisation
  * use_clipping      False: Full Defense path does not clip (threshold=inf)
  * use_noise         False: Full Defense path adds no Gaussian noise
  * use_fpe           False: hyperparameters broadcast in plaintext
  * training / model / graph / defense hyperparameters (see RunConfig)
  * holdout           "val" for tuning, "test" for final reporting

Every returned value is a fresh, JSON-serialisable object (the previous
version stored the final-round dict inside its own history list, which
created a circular reference that json.dump could not serialise).

CPU only: config.py hides all CUDA devices before torch initialises CUDA.
"""
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

import config
from agents.classifier_agent import ClassifierAgent
from agents.coordinator_agent import CoordinatorAgent
from agents.logger_agent import LoggerAgent
from agents.threshold_agent import ThresholdAgent
from client.data_loader import load_and_partition
from client.federated_client import FederatedClient
from crypto.ckks_engine import CKKSEngine
from crypto.fpe_engine import FPEEngine
from evaluation.metrics import evaluate_federation
from models.graphsage_model import GraphSAGENet

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


@dataclass
class RunConfig:
    # federated training
    rounds: int = config.NUM_ROUNDS
    local_epochs: int = config.LOCAL_EPOCHS
    lr: float = config.LEARNING_RATE
    # model
    hidden: int = config.HIDDEN_CHANNELS
    dropout: float = config.DROPOUT
    # graph construction
    knn_k: int = config.KNN_NEIGHBORS
    # defense
    anomaly_threshold: float = config.ANOMALY_THRESHOLD
    clip_threshold: float = config.CLIP_THRESHOLD
    noise_scale: float = config.NOISE_SCALE
    # threat model (default = the project's default: clients 3 & 7, scale_gaussian)
    attack_type: str = "scale_gaussian"
    malicious_ids: List[int] = field(default_factory=lambda: list(config.MALICIOUS_CLIENT_IDS))
    # component switches
    use_ckks: bool = True
    use_multi_agent: bool = True
    use_clipping: bool = True
    use_noise: bool = True
    use_fpe: bool = True

    def as_dict(self) -> Dict:
        return asdict(self)


# The project's configuration BEFORE this study (config.py defaults as
# shipped).  Frozen explicitly so the "original" baseline cannot drift.
ORIGINAL_CONFIG = dict(rounds=8, local_epochs=3, lr=0.05, hidden=32, dropout=0.3, knn_k=6,
                       anomaly_threshold=0.7, clip_threshold=1.0, noise_scale=0.01)


def original_config(**kw) -> "RunConfig":
    return RunConfig(**{**ORIGINAL_CONFIG, **kw})


def _strip_private(m: Dict) -> Dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


def run_config(cfg: RunConfig, seed: int, holdout: str = "test",
               quiet: bool = True, log_path: Optional[str] = None) -> Dict:
    """Runs the federated pipeline once and returns a JSON-serialisable record."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    t_run = time.perf_counter()

    client_graphs = load_and_partition(seed=seed, k=cfg.knn_k, holdout=holdout)
    model_kw = dict(hidden_channels=cfg.hidden, dropout=cfg.dropout)
    make_model = lambda: GraphSAGENet(**model_kw)  # noqa: E731
    global_model = make_model()

    attack_kwargs = {"clip_threshold": cfg.clip_threshold} if cfg.attack_type == "adaptive" else {}
    clients = [
        FederatedClient(
            client_id=cid, train_graph=tr, test_graph=ev, model_class=make_model,
            is_malicious=(cid in cfg.malicious_ids), attack_type=cfg.attack_type,
            attack_kwargs=attack_kwargs,
        )
        for cid, (tr, ev) in client_graphs.items()
    ]
    # Round-1 hyperparameters (later rounds: FPE-decrypted broadcast, or
    # the same values in plaintext when FPE is ablated).
    for c in clients:
        c.lr, c.epochs = cfg.lr, cfg.local_epochs
    eval_graphs = [c.test_graph for c in clients]

    ckks = CKKSEngine() if cfg.use_ckks else None
    fpe = FPEEngine()
    classifier_agent = ClassifierAgent()
    threshold_agent = ThresholdAgent(anomaly_threshold=cfg.anomaly_threshold)
    coordinator_agent = CoordinatorAgent()
    coordinator_agent.defense.clip_threshold = cfg.clip_threshold if cfg.use_clipping else float("inf")
    coordinator_agent.defense.noise_scale = cfg.noise_scale if cfg.use_noise else 0.0
    logger_agent = LoggerAgent(log_path) if log_path else None

    history: List[Dict] = []
    full_defense_rounds = 0
    totals = {"local_train_s": 0.0, "encrypt_s": 0.0, "aggregate_s": 0.0,
              "decrypt_s": 0.0, "defense_s": 0.0, "eval_s": 0.0}
    final_eval = None

    for rnd in range(1, cfg.rounds + 1):
        rec = {"round": rnd}
        global_state = global_model.state_dict()

        t = time.perf_counter()
        deltas = [c.local_train(global_state) for c in clients]
        rec["local_train_s"] = time.perf_counter() - t
        honest = [c.last_train_loss for c in clients if c.client_id not in cfg.malicious_ids]
        rec["train_loss"] = float(np.mean(honest)) if honest else float("nan")

        flats, flat_shapes = [], None
        for d in deltas:
            f, flat_shapes = CKKSEngine.flatten_state_dict(d)
            flats.append(f)
        # reference plaintext mean (in the CKKS arm used ONLY to measure CKKS error)
        t = time.perf_counter()
        plain_mean = np.mean(flats, axis=0)
        plain_agg_s = time.perf_counter() - t

        if cfg.use_ckks:
            t = time.perf_counter()
            enc = [ckks.encrypt_vector(f) for f in flats]
            rec["encrypt_s"] = time.perf_counter() - t
            t = time.perf_counter()
            agg_ct = ckks.aggregate_encrypted(enc)
            rec["aggregate_s"] = time.perf_counter() - t
            t = time.perf_counter()
            agg_flat = ckks.decrypt_vector(agg_ct, len(flats[0]))
            rec["decrypt_s"] = time.perf_counter() - t
            err = agg_flat - plain_mean
            rec["ckks_max_abs_err"] = float(np.max(np.abs(err)))
            rec["ckks_mean_abs_err"] = float(np.mean(np.abs(err)))
            rec["ckks_rmse"] = float(np.sqrt(np.mean(err ** 2)))
            rec["ckks_rel_l2_err"] = float(np.linalg.norm(err) / (np.linalg.norm(plain_mean) or 1e-12))
            rec["n_ciphertexts_per_client"] = len(enc[0])
        else:
            agg_flat = plain_mean
            rec["encrypt_s"] = 0.0
            rec["aggregate_s"] = plain_agg_s
            rec["decrypt_s"] = 0.0

        t = time.perf_counter()
        if cfg.use_multi_agent:
            c_dec = classifier_agent.process(agg_flat, round_number=rnd)
            t_dec = threshold_agent.process(c_dec)
            co_dec = coordinator_agent.process(t_dec, agg_flat)
            if logger_agent:
                for d in (c_dec, t_dec, co_dec):
                    logger_agent.log_decision(d, rnd)
            rec["anomaly_score"] = float(c_dec.metadata["anomaly_score"])
            rec["path"] = co_dec.action
            if co_dec.action != "fast_path":
                full_defense_rounds += 1
            final_update = co_dec.metadata["update"]
        else:
            rec["anomaly_score"] = float("nan")
            rec["path"] = "no_defense"
            final_update = agg_flat
        rec["defense_s"] = time.perf_counter() - t
        rec["update_norm"] = float(np.linalg.norm(final_update))

        new_state = CKKSEngine.unflatten_to_state_dict(final_update, flat_shapes)
        updated = {k: global_state[k] + new_state[k] for k in global_state}
        global_model.load_state_dict(updated)

        enc_hp = fpe.encrypt_hyperparams(learning_rate=cfg.lr, local_epochs=cfg.local_epochs) if cfg.use_fpe else None
        for c in clients:
            c.apply_global_update(updated, enc_hp)
            if not cfg.use_fpe:
                c.lr, c.epochs = cfg.lr, cfg.local_epochs

        t = time.perf_counter()
        ev = evaluate_federation(global_model, eval_graphs)
        rec["eval_s"] = time.perf_counter() - t
        for k in METRIC_KEYS + ["loss"]:
            rec[k] = ev[k]
        for k in METRIC_KEYS:
            rec["client_macro_" + k] = ev["client_macro_" + k]
        for k in totals:
            totals[k] += rec.get(k, 0.0)
        history.append(rec)
        final_eval = ev
        if not quiet:
            print(f"  round {rnd}: acc={ev['accuracy']:.4f} f1={ev['f1']:.4f} path={rec['path']}")

    final = _strip_private(final_eval)
    ck = [h["ckks_max_abs_err"] for h in history if "ckks_max_abs_err" in h]
    return {
        "seed": seed,
        "holdout": holdout,
        "config": cfg.as_dict(),
        "final": final,
        "predictions": {"y_true": final_eval["_y_true"], "p_malignant": final_eval["_p_malignant"]},
        "history": history,
        "full_defense_rounds": full_defense_rounds,
        "timing": {**totals, "total_s": time.perf_counter() - t_run},
        "ckks_max_abs_err_overall": max(ck) if ck else None,
        "ckks_backend": config.CKKS_BACKEND if cfg.use_ckks else None,
        "device": "cpu",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_pipeline(seed: int, use_ckks: bool = True, attack_type: str = "scale_gaussian",
                 malicious_ids: Optional[List[int]] = None, num_rounds: Optional[int] = None,
                 quiet: bool = True, use_multi_agent: bool = True, use_clipping: bool = True,
                 use_noise: bool = True, use_fpe: bool = True, holdout: str = "test",
                 **overrides) -> Dict:
    """Backward-compatible wrapper returning flat final metrics plus a
    (non-circular) per-round history."""
    cfg = RunConfig(
        use_ckks=use_ckks, attack_type=attack_type,
        malicious_ids=list(config.MALICIOUS_CLIENT_IDS if malicious_ids is None else malicious_ids),
        use_multi_agent=use_multi_agent, use_clipping=use_clipping, use_noise=use_noise,
        use_fpe=use_fpe, **overrides,
    )
    if num_rounds is not None:
        cfg.rounds = num_rounds
    r = run_config(cfg, seed, holdout=holdout, quiet=quiet)
    out = {k: r["final"][k] for k in METRIC_KEYS}
    out["full_defense_rounds"] = r["full_defense_rounds"]
    out["num_rounds"] = cfg.rounds
    out["round_history"] = [{k: h[k] for k in METRIC_KEYS + ["loss", "round"]} for h in r["history"]]
    return out
