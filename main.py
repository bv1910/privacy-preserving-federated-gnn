"""
Main orchestrator for the Privacy-Preserving Federated GraphSAGE system.

Pipeline per round:
  1. Local training on each client's k-NN patient graph
  2. CKKS encryption of each client's weight delta (Pyfhel)
  3. Homomorphic aggregation (encrypted addition + 1/N scaling)
  4. Decryption of the aggregated update
  5. 4-agent defense pipeline (Classifier -> Threshold -> Coordinator -> Logger)
  6. Global model update
  7. Testing loop (evaluate + FPE-encrypt metrics + explain + log)
  8. Round summary
"""
import json
import os
import sys
import time

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
from explainability.explainer import Explainer
from models.graphsage_model import GraphSAGENet
from testing.testing_loop import TestingLoop


def _apply_selected_config():
    """--selected: use the configuration chosen on validation by
    experiments/sweep.py (results/summary/selected_config.json)."""
    path = os.path.join(config.SUMMARY_DIR, "selected_config.json")
    with open(path) as f:
        sel = json.load(f)["selected_config"]
    config.NUM_ROUNDS, config.LOCAL_EPOCHS, config.LEARNING_RATE = sel["rounds"], sel["local_epochs"], sel["lr"]
    config.HIDDEN_CHANNELS, config.DROPOUT, config.KNN_NEIGHBORS = sel["hidden"], sel["dropout"], sel["knn_k"]
    print(f"[Setup] Using selected configuration: {sel}")


def run_federated_training():
    print("=" * 70)
    print("Privacy-Preserving Federated GraphSAGE — Breast Cancer Diagnosis")
    print("=" * 70)

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print(f"\n[Setup] Loading & partitioning breast cancer dataset across {config.NUM_CLIENTS} clients...")
    client_graphs = load_and_partition(k=config.KNN_NEIGHBORS)
    make_model = lambda: GraphSAGENet(hidden_channels=config.HIDDEN_CHANNELS, dropout=config.DROPOUT)  # noqa: E731
    global_model = make_model()

    clients = [
        FederatedClient(
            client_id=cid,
            train_graph=train_g,
            test_graph=test_g,
            model_class=make_model,
            is_malicious=(cid in config.MALICIOUS_CLIENT_IDS),
        )
        for cid, (train_g, test_g) in client_graphs.items()
    ]
    print(f"[Setup] {len(clients)} clients ready "
          f"(malicious: {config.MALICIOUS_CLIENT_IDS or 'none'})")

    print("[Setup] Initializing CKKS engine (Pyfhel)... this may take a few seconds.")
    ckks = CKKSEngine()
    fpe = FPEEngine()
    explainer = Explainer()

    classifier_agent = ClassifierAgent()
    threshold_agent = ThresholdAgent()
    coordinator_agent = CoordinatorAgent()
    logger_agent = LoggerAgent()
    testing_loop = TestingLoop(logger_agent)

    prior_metrics = {}
    round_summaries = []

    for round_num in range(1, config.NUM_ROUNDS + 1):
        t0 = time.time()
        print(f"\nRound {round_num}/{config.NUM_ROUNDS}:")

        # 1. Local training
        global_state = global_model.state_dict()
        client_deltas = [c.local_train(global_state) for c in clients]
        print(f"  [1] Local training: {len(clients)} clients trained ({config.LOCAL_EPOCHS} epochs each)")

        # 2. CKKS encryption
        flat_shapes = None
        encrypted_deltas = []
        for delta in client_deltas:
            flat, shapes = ckks.flatten_state_dict(delta)
            flat_shapes = shapes
            encrypted_deltas.append(ckks.encrypt_vector(flat))
        n_ciphertexts = len(encrypted_deltas[0])
        print(f"  [2] CKKS encryption: {len(clients)} client updates encrypted ({n_ciphertexts} ciphertexts each)")

        # 3. Homomorphic aggregation
        aggregated_ct = ckks.aggregate_encrypted(encrypted_deltas)
        print("  [3] Homomorphic aggregation: encrypted addition + scaling")

        # 4. Decryption
        flat_len, _ = ckks.flatten_state_dict(client_deltas[0])
        agg_flat = ckks.decrypt_vector(aggregated_ct, len(flat_len))
        print("  [4] Decryption: aggregated update decrypted")

        # 5. Agent pipeline
        print("  [5] Agent Pipeline:")
        c_decision = classifier_agent.process(agg_flat, round_number=round_num)
        logger_agent.log_decision(c_decision, round_num)
        print(f"      Classifier: label={c_decision.action}, "
              f"anomaly_score={c_decision.metadata['anomaly_score']:.3f}")

        t_decision = threshold_agent.process(c_decision)
        logger_agent.log_decision(t_decision, round_num)
        print(f"      Threshold: risk_level={t_decision.action} "
              f"({c_decision.metadata['anomaly_score']:.2f} vs {config.ANOMALY_THRESHOLD})")

        co_decision = coordinator_agent.process(t_decision, agg_flat)
        logger_agent.log_decision(co_decision, round_num)
        path_label = "FAST PATH" if co_decision.action == "fast_path" else "FULL DEFENSE"
        print(f"      Coordinator: {path_label} -> update {'passed through' if co_decision.action == 'fast_path' else 'sanitized'}")

        final_update = co_decision.metadata["update"]

        # 6. Global model update
        new_state = ckks.unflatten_to_state_dict(final_update, flat_shapes)
        # deltas are averaged; apply on top of previous global weights
        updated_state = {
            k: global_state[k] + new_state[k] for k in global_state
        }
        global_model.load_state_dict(updated_state)
        print("  [6] Global model updated")

        # Broadcast: send new global weights to every client, with the
        # next round's training hyperparameters FPE-encrypted in transit.
        encrypted_hp = fpe.encrypt_hyperparams(
            learning_rate=config.LEARNING_RATE, local_epochs=config.LOCAL_EPOCHS
        )
        for c in clients:
            c.apply_global_update(updated_state, encrypted_hp)

        broadcast_explanation = explainer.explain_hyperparam_broadcast(
            round_number=round_num,
            learning_rate=config.LEARNING_RATE,
            local_epochs=config.LOCAL_EPOCHS,
            num_clients=len(clients),
        )
        from agents.base_agent import Decision as _Decision

        logger_agent.log_decision(
            _Decision(
                agent_name="hyperparam_broadcast",
                action="broadcast",
                reasoning=broadcast_explanation,
                confidence=1.0,
                metadata={
                    "encrypted_learning_rate": encrypted_hp["learning_rate"],
                    "encrypted_local_epochs": encrypted_hp["local_epochs"],
                    "num_clients": len(clients),
                },
            ),
            round_number=round_num,
        )
        print(f"  [6b] Hyperparameter broadcast: FPE-encrypted "
              f"(lr='{encrypted_hp['learning_rate']}', epochs='{encrypted_hp['local_epochs']}')")

        # 7. Testing loop
        print("  [7] Testing Loop:")
        update_stats = {
            "norm": float(np.linalg.norm(final_update)),
            "path_taken": co_decision.action,
        }
        results = testing_loop.run(clients, global_model, round_num, prior_metrics, update_stats)
        for r in results:
            print(f"      Client {r['client_id']}: acc={r['accuracy']:.4f}, loss={r['loss']:.4f} "
                  f"[FPE: {r['fpe_encrypted_accuracy']}] [{r['pass_fail']}]")

        # 8. Round summary
        avg_acc = float(np.mean([r["accuracy"] for r in results]))
        avg_loss = float(np.mean([r["loss"] for r in results]))
        elapsed = time.time() - t0
        print(f"  [8] Round summary: avg_accuracy={avg_acc:.4f}, avg_loss={avg_loss:.4f} ({elapsed:.1f}s)")

        round_summaries.append(
            {
                "round": round_num,
                "avg_accuracy": avg_acc,
                "avg_loss": avg_loss,
                "anomaly_score": c_decision.metadata["anomaly_score"],
                "risk_level": t_decision.action,
                "path_taken": co_decision.action,
                "per_client": results,
                "elapsed_sec": elapsed,
            }
        )

    with open(config.ROUND_RESULTS_PATH, "w") as f:
        json.dump(round_summaries, f, indent=2, default=str)

    torch.save(global_model.state_dict(), f"{config.CHECKPOINT_DIR}/global_model_final.pt")

    print("\n" + "=" * 70)
    print(f"Training complete. Final avg_accuracy={round_summaries[-1]['avg_accuracy']:.4f}")
    print(f"Round results saved to {config.ROUND_RESULTS_PATH}")
    print(f"Audit log saved to {config.AUDIT_LOG_PATH}")
    print("=" * 70)

    return round_summaries


if __name__ == "__main__":
    if "--selected" in sys.argv:
        _apply_selected_config()
    run_federated_training()
