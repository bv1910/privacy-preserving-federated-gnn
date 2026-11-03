"""Agent 3 — Coordinator: routes the update to the Fast Path (low risk) or
Full Defense path (high risk, applies clipping + noise)."""
import numpy as np

from agents.base_agent import BaseAgent, Decision
from defense.defense_module import FullDefense


class CoordinatorAgent(BaseAgent):
    name = "coordinator_agent"

    def __init__(self):
        self.defense = FullDefense()

    def process(self, threshold_decision: Decision, update: np.ndarray, **kwargs) -> Decision:
        risk_level = threshold_decision.metadata["risk_level"]

        if risk_level == "low":
            return Decision(
                agent_name=self.name,
                action="fast_path",
                reasoning="Risk level=low -> update passed through unchanged.",
                confidence=threshold_decision.confidence,
                metadata={
                    **threshold_decision.metadata,
                    "path_taken": "fast_path",
                    "update": update,
                    "defense_applied": None,
                },
            )

        sanitized, defense_meta = self.defense.apply(update)
        return Decision(
            agent_name=self.name,
            action="full_defense",
            reasoning=(
                f"Risk level=high -> routed to Full Defense "
                f"(clipped={defense_meta['clipped']}, "
                f"norm {defense_meta['original_norm']:.3f} -> {defense_meta['clipped_norm']:.3f})."
            ),
            confidence=threshold_decision.confidence,
            metadata={
                **threshold_decision.metadata,
                "path_taken": "full_defense",
                "update": sanitized,
                "defense_applied": defense_meta,
            },
        )
