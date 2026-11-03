import json
import os

import config


def test_end_to_end_small_run():
    """Runs a reduced pipeline (2 rounds) and verifies audit log + round
    results are populated and well-formed."""
    orig_rounds, orig_clients = config.NUM_ROUNDS, config.NUM_CLIENTS
    config.NUM_ROUNDS = 2
    try:
        # fresh log file for this test run
        if os.path.exists(config.AUDIT_LOG_PATH):
            os.remove(config.AUDIT_LOG_PATH)

        from main import run_federated_training

        summaries = run_federated_training()

        assert len(summaries) == 2
        assert os.path.exists(config.ROUND_RESULTS_PATH)
        with open(config.ROUND_RESULTS_PATH) as f:
            data = json.load(f)
        assert isinstance(data, list) and len(data) == 2

        assert os.path.exists(config.AUDIT_LOG_PATH)
        with open(config.AUDIT_LOG_PATH) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        agent_names = {e["agent_name"] for e in lines}
        assert {"classifier_agent", "threshold_agent", "coordinator_agent", "testing_loop"} <= agent_names
    finally:
        config.NUM_ROUNDS, config.NUM_CLIENTS = orig_rounds, orig_clients


if __name__ == "__main__":
    test_end_to_end_small_run()
    print("test_pipeline.py: all tests passed")
