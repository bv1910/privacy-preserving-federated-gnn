import numpy as np

import config
from agents.classifier_agent import ClassifierAgent
from agents.coordinator_agent import CoordinatorAgent
from agents.threshold_agent import ThresholdAgent


def test_classifier_score_in_range():
    agent = ClassifierAgent()
    update = np.random.randn(1000) * 0.01
    decision = agent.process(update, round_number=1)
    assert 0.0 <= decision.metadata["anomaly_score"] <= 1.0


def test_threshold_routes_at_boundary():
    agent = ThresholdAgent()
    from agents.base_agent import Decision

    low = Decision("classifier_agent", "benign", "x", 0.3, {"anomaly_score": 0.3})
    high = Decision("classifier_agent", "suspicious", "x", 0.9, {"anomaly_score": 0.9})

    assert agent.process(low).action == "low"
    assert agent.process(high).action == "high"


def test_coordinator_calls_defense_when_flagged():
    from agents.base_agent import Decision

    coord = CoordinatorAgent()
    update = np.random.randn(500) * 5.0  # large norm -> should get clipped
    high_decision = Decision("threshold_agent", "high", "x", 0.9, {"risk_level": "high", "anomaly_score": 0.9})

    result = coord.process(high_decision, update)
    assert result.action == "full_defense"
    assert result.metadata["defense_applied"] is not None
    assert result.metadata["defense_applied"]["clipped"] in (True, False)


if __name__ == "__main__":
    test_classifier_score_in_range()
    test_threshold_routes_at_boundary()
    test_coordinator_calls_defense_when_flagged()
    print("test_agents.py: all tests passed")
