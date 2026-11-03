"""Agent 2 — Threshold: converts the classifier's anomaly score into a
discrete risk level using config.ANOMALY_THRESHOLD."""
import config
from agents.base_agent import BaseAgent, Decision


class ThresholdAgent(BaseAgent):
    name = "threshold_agent"

    def __init__(self, anomaly_threshold: float = None):
        # None -> config.ANOMALY_THRESHOLD (original behaviour)
        self.anomaly_threshold = anomaly_threshold

    def process(self, classifier_decision: Decision, **kwargs) -> Decision:
        score = classifier_decision.metadata["anomaly_score"]
        thr = config.ANOMALY_THRESHOLD if self.anomaly_threshold is None else self.anomaly_threshold
        risk_level = "high" if score >= thr else "low"

        return Decision(
            agent_name=self.name,
            action=risk_level,
            reasoning=(
                f"anomaly_score={score:.3f} "
                f"{'>=' if risk_level == 'high' else '<'} "
                f"threshold={thr} -> risk_level={risk_level}"
            ),
            confidence=score,
            metadata={**classifier_decision.metadata, "risk_level": risk_level},
        )
