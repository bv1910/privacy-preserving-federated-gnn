"""Agent 1 — Classifier: statistically scores the decrypted aggregated
weight update for anomalous / adversarial behaviour."""
from typing import Dict, List

import numpy as np

from agents.base_agent import BaseAgent, Decision


class ClassifierAgent(BaseAgent):
    name = "classifier_agent"

    def __init__(self):
        self.history: List[np.ndarray] = []  # past aggregated-update vectors

    def _features(self, update: np.ndarray) -> Dict[str, float]:
        norm = float(np.linalg.norm(update))
        mean = float(np.mean(update))
        std = float(np.std(update))
        max_abs = float(np.max(np.abs(update)))

        if self.history:
            ref = self.history[-1]
            denom = (np.linalg.norm(update) * np.linalg.norm(ref)) or 1e-8
            cosine_sim = float(np.dot(update, ref) / denom)
        else:
            cosine_sim = 1.0  # no history yet -> assume aligned

        return {"norm": norm, "mean": mean, "std": std, "max_abs": max_abs, "cosine_sim": cosine_sim}

    def _anomaly_score(self, features: Dict[str, float]) -> float:
        """Rule-based anomaly score in [0, 1], combining:
        - deviation of the update norm from the historical mean norm (z-score)
        - low cosine similarity with the previous round's update direction
        """
        if self.history:
            hist_norms = [float(np.linalg.norm(h)) for h in self.history]
            hist_mean, hist_std = np.mean(hist_norms), (np.std(hist_norms) or 1e-6)
            z = abs(features["norm"] - hist_mean) / hist_std
            norm_score = min(1.0, z / 3.0)  # 3-sigma -> score 1.0
        else:
            norm_score = 0.0

        direction_score = max(0.0, 1.0 - features["cosine_sim"]) if self.history else 0.0
        score = 0.6 * norm_score + 0.4 * direction_score
        return float(np.clip(score, 0.0, 1.0))

    def process(self, update: np.ndarray, round_number: int = 0, **kwargs) -> Decision:
        features = self._features(update)
        score = self._anomaly_score(features)
        label = "suspicious" if score >= 0.5 else "benign"

        self.history.append(update)

        return Decision(
            agent_name=self.name,
            action=label,
            reasoning=(
                f"Update L2 norm={features['norm']:.3f}, cosine_sim to prior "
                f"round={features['cosine_sim']:.3f} -> anomaly_score={score:.3f}"
            ),
            confidence=score,
            metadata={"round": round_number, "features": features, "anomaly_score": score},
        )
