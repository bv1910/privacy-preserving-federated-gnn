from experiments.pipeline_runner import run_pipeline


def test_plaintext_pipeline_runs_and_returns_full_metrics():
    """Uses use_ckks=False so this test doesn't require Pyfhel/TenSEAL to
    be installed -- it exercises the shared training/agent/metrics path."""
    result = run_pipeline(seed=123, use_ckks=False, num_rounds=2)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in result
    assert result["num_rounds"] == 2
    assert len(result["round_history"]) == 2
    assert result["full_defense_rounds"] >= 0


def test_plaintext_pipeline_with_each_attack_type_runs():
    for attack in ["sign_flip", "scale", "gaussian", "adaptive", "label_flip"]:
        result = run_pipeline(seed=7, use_ckks=False, attack_type=attack, num_rounds=1)
        assert "accuracy" in result
