"""Post-distribution testing loop: evaluates the new global model on every
client's held-out test graph, FPE-encrypts the metrics, generates an
explanation, and logs everything via Agent 4."""
from typing import Dict, List

import numpy as np
import torch

from agents.logger_agent import LoggerAgent
from client.federated_client import FederatedClient
from crypto.fpe_engine import FPEEngine
from explainability.explainer import Explainer


class TestingLoop:
    def __init__(self, logger: LoggerAgent):
        self.fpe = FPEEngine()
        self.explainer = Explainer()
        self.logger = logger

    def run(
        self,
        clients: List[FederatedClient],
        global_model: torch.nn.Module,
        round_number: int,
        prior_metrics: Dict[int, Dict[str, float]],
        update_stats: Dict[str, float],
    ) -> List[Dict]:
        results = []
        for client in clients:
            acc, loss = client.evaluate(global_model)
            new_metrics = {"accuracy": acc, "loss": loss}
            old_metrics = prior_metrics.get(client.client_id, new_metrics)

            fpe_acc = self.fpe.encrypt_metric(acc)
            fpe_loss = self.fpe.encrypt_metric(loss)

            explanation = self.explainer.explain_test_result(
                client.client_id, old_metrics, new_metrics, update_stats
            )

            result = {
                "client_id": client.client_id,
                "accuracy": acc,
                "loss": loss,
                "fpe_encrypted_accuracy": fpe_acc,
                "fpe_encrypted_loss": fpe_loss,
                "explanation": explanation,
                "pass_fail": "PASS" if acc >= old_metrics["accuracy"] - 1e-6 else "FAIL",
            }
            results.append(result)

            self.logger.log_decision(
                decision=_as_decision(result),
                round_number=round_number,
                client_id=client.client_id,
            )
            prior_metrics[client.client_id] = new_metrics

        return results


def _as_decision(result: Dict):
    from agents.base_agent import Decision

    return Decision(
        agent_name="testing_loop",
        action=result["pass_fail"],
        reasoning=result["explanation"],
        confidence=result["accuracy"],
        metadata={
            "accuracy": result["accuracy"],
            "loss": result["loss"],
            "fpe_encrypted_accuracy": result["fpe_encrypted_accuracy"],
            "fpe_encrypted_loss": result["fpe_encrypted_loss"],
        },
    )
