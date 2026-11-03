import torch
from torch_geometric.data import Data

from attacks.attack_types import (
    sign_flip, scale_attack, gaussian_attack, scale_gaussian_attack,
    adaptive_attack, label_flip_graph, apply_attack, ATTACK_REGISTRY,
)


def _toy_delta():
    return {"w1": torch.ones(4, 4), "b1": torch.ones(4)}


def test_sign_flip_negates():
    delta = _toy_delta()
    out = sign_flip(delta)
    assert torch.allclose(out["w1"], -delta["w1"])


def test_scale_attack_multiplies():
    delta = _toy_delta()
    out = scale_attack(delta, factor=3.0)
    assert torch.allclose(out["w1"], delta["w1"] * 3.0)


def test_gaussian_attack_changes_values_but_same_shape():
    delta = _toy_delta()
    out = gaussian_attack(delta, scale=0.5)
    assert out["w1"].shape == delta["w1"].shape
    assert not torch.allclose(out["w1"], delta["w1"])


def test_scale_gaussian_matches_original_default_behavior():
    torch.manual_seed(0)
    delta = _toy_delta()
    out = scale_gaussian_attack(delta, factor=6.0, scale=0.3)
    # should be centered around delta * 6 (noise has mean 0)
    assert out["w1"].mean().item() > delta["w1"].mean().item() * 5.0


def test_adaptive_attack_respects_clip_threshold():
    delta = {"w1": torch.ones(100) * 10.0}  # large norm
    out = adaptive_attack(delta, clip_threshold=1.0)
    flat_norm = torch.norm(out["w1"]).item()
    assert flat_norm <= 1.0 + 1e-4


def test_label_flip_graph_flips_all_labels_when_fraction_1():
    y = torch.tensor([0, 1, 0, 1])
    g = Data(x=torch.randn(4, 3), edge_index=torch.zeros(2, 0, dtype=torch.long), y=y.clone())
    flipped = label_flip_graph(g, flip_fraction=1.0)
    assert torch.equal(flipped.y, 1 - y)
    assert torch.equal(g.y, y)  # original untouched (clone)


def test_apply_attack_dispatches_all_registry_entries():
    delta = _toy_delta()
    for name in ATTACK_REGISTRY:
        out = apply_attack(delta, name)
        assert set(out.keys()) == set(delta.keys())


def test_apply_attack_rejects_unknown_type():
    delta = _toy_delta()
    try:
        apply_attack(delta, "not_a_real_attack")
        assert False, "expected ValueError"
    except ValueError:
        pass
