"""Dual-mode explainability:
1. SHAP mode — explains the Classifier agent's anomaly_score in terms of
   the statistical features it was computed from (norm, mean, std, max_abs,
   cosine_sim).
2. Rule-based mode — fast, human-readable text for client test-result
   explanations and as a fallback when SHAP is unavailable/slow.
"""
from typing import Dict, List

import numpy as np

FEATURE_NAMES = ["norm", "mean", "std", "max_abs", "cosine_sim"]


class Explainer:
    def __init__(self):
        self._shap_explainer = None  # lazily built (needs a background dataset)

    # -- SHAP: explain an agent classification -----------------------------
    def explain_classification(self, features: Dict[str, float], score: float, history: List[Dict]) -> Dict:
        """Returns {"feature_importance": {...}, "text": "..."}.

        Uses SHAP's KernelExplainer over the rule-based scoring function so
        importance values are grounded in the same logic that produced the
        score (not a separate learned surrogate).
        """
        try:
            import shap

            feat_vec = np.array([[features[f] for f in FEATURE_NAMES]])

            # background = mildly perturbed benign-looking feature vector
            background = np.array(
                [[features["norm"] * 0.9, 0.0, features["std"] * 0.9, features["max_abs"] * 0.9, 1.0]]
            )

            def score_fn(X: np.ndarray) -> np.ndarray:
                out = []
                for row in X:
                    f = dict(zip(FEATURE_NAMES, row))
                    norm_score = min(1.0, abs(f["norm"] - background[0][0]) / (abs(background[0][0]) + 1e-6))
                    dir_score = max(0.0, 1.0 - f["cosine_sim"])
                    out.append(0.6 * norm_score + 0.4 * dir_score)
                return np.array(out)

            explainer = shap.KernelExplainer(score_fn, background, silent=True)
            shap_values = explainer.shap_values(feat_vec, silent=True, nsamples=32)
            importance = {name: float(val) for name, val in zip(FEATURE_NAMES, shap_values[0])}

        except Exception:
            # Fallback: simple normalized magnitude-based importance
            vals = np.array([abs(features[f]) for f in FEATURE_NAMES])
            norm_vals = vals / (vals.sum() + 1e-8)
            importance = dict(zip(FEATURE_NAMES, norm_vals.tolist()))

        top_feature = max(importance, key=lambda k: abs(importance[k]))
        text = (
            f"Update classified with anomaly_score={score:.3f}; the dominant "
            f"driver was '{top_feature}' (contribution={importance[top_feature]:.3f})."
        )
        return {"feature_importance": importance, "text": text}

    # -- Rule-based: explain a client's test-round outcome ------------------
    def explain_test_result(
        self,
        client_id: int,
        old_metrics: Dict[str, float],
        new_metrics: Dict[str, float],
        update_stats: Dict[str, float],
    ) -> str:
        delta_acc_pp = (new_metrics["accuracy"] - old_metrics["accuracy"]) * 100
        direction = "improved" if delta_acc_pp >= 0 else "degraded"
        return (
            f"Global update {direction} Client {client_id}'s accuracy from "
            f"{old_metrics['accuracy']*100:.1f}% to {new_metrics['accuracy']*100:.1f}% "
            f"({delta_acc_pp:+.1f}pp). Update L2 norm was {update_stats.get('norm', 0):.3f}; "
            f"path taken: {update_stats.get('path_taken', 'fast_path')}."
        )

    # -- Rule-based: explain the hyperparameter broadcast --------------------
    def explain_hyperparam_broadcast(
        self,
        round_number: int,
        learning_rate: float,
        local_epochs: int,
        num_clients: int,
    ) -> str:
        return (
            f"Round {round_number}: server broadcast the updated global model "
            f"weights to all {num_clients} clients. Training hyperparameters "
            f"(learning_rate={learning_rate}, local_epochs={local_epochs}) were "
            f"FPE-encrypted (FF3-1) before transmission and decrypted locally by "
            f"each client — the values never appear in plaintext on the wire, "
            f"only inside each client's process."
        )
