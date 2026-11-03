"""
Malicious-client attack strategies used to evaluate the 4-agent defense
pipeline. The original pipeline only exercised one fixed corruption
(scale x6 + Gaussian jitter on the weight delta). For a publication-grade
robustness claim, the defense needs to be shown working across several
qualitatively different attack families, not just the one it happened to be
tuned against.

All functions operate on a client's weight-delta state_dict (post local
training, pre-encryption) except `label_flip_graph`, which operates on the
training graph itself before local training runs.
"""
from typing import Dict

import torch
from torch_geometric.data import Data

import config


def label_flip_graph(train_graph: Data, flip_fraction: float = 1.0) -> Data:
    """Data-poisoning attack: flips the diagnosis label on a fraction of
    this client's training nodes before local training. flip_fraction=1.0
    flips every label (worst case)."""
    g = train_graph.clone()
    n = g.y.shape[0]
    n_flip = max(1, int(n * flip_fraction))
    idx = torch.randperm(n)[:n_flip]
    g.y[idx] = 1 - g.y[idx]
    return g


def sign_flip(delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Model-poisoning attack: negates the honestly-trained delta so the
    aggregated update moves the global model in the wrong direction."""
    return {name: -d for name, d in delta.items()}


def scale_attack(delta: Dict[str, torch.Tensor], factor: float = 6.0) -> Dict[str, torch.Tensor]:
    """Amplifies the delta magnitude to try to dominate the federated
    average (the original hardcoded attack in this pipeline, factor=6)."""
    return {name: d * factor for name, d in delta.items()}


def gaussian_attack(delta: Dict[str, torch.Tensor], scale: float = 0.3) -> Dict[str, torch.Tensor]:
    """Pure noise injection: replaces the honest signal with additive
    Gaussian noise, without deliberately scaling it up."""
    return {name: d + torch.randn_like(d) * scale for name, d in delta.items()}


def scale_gaussian_attack(
    delta: Dict[str, torch.Tensor], factor: float = 6.0, scale: float = 0.3
) -> Dict[str, torch.Tensor]:
    """Original attack used by this pipeline: scale + Gaussian jitter
    combined. Kept as the default so existing behavior (main.py, tests) is
    unchanged."""
    return {name: d * factor + torch.randn_like(d) * scale for name, d in delta.items()}


def adaptive_attack(
    delta: Dict[str, torch.Tensor], clip_threshold: float = None
) -> Dict[str, torch.Tensor]:
    """Evasive attack: sign-flips the delta, then rescales its L2 norm to
    sit just under the defense module's clip threshold, attempting to slip
    through as a 'fast path' update that looks normal in magnitude while
    still being adversarial in direction."""
    clip_threshold = clip_threshold if clip_threshold is not None else config.CLIP_THRESHOLD
    flipped = {name: -d for name, d in delta.items()}
    flat = torch.cat([d.flatten() for d in flipped.values()])
    norm = torch.norm(flat).item()
    if norm == 0:
        return flipped
    target_norm = clip_threshold * 0.95
    scale_factor = target_norm / norm
    return {name: d * scale_factor for name, d in flipped.items()}


ATTACK_REGISTRY = {
    "scale_gaussian": scale_gaussian_attack,  # original default behavior
    "sign_flip": sign_flip,
    "scale": scale_attack,
    "gaussian": gaussian_attack,
    "adaptive": adaptive_attack,
}


def apply_attack(delta: Dict[str, torch.Tensor], attack_type: str, **kwargs) -> Dict[str, torch.Tensor]:
    """kwargs are forwarded to the attack (e.g. clip_threshold for the
    adaptive attacker, so it targets the clip threshold actually in use)."""
    if attack_type not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack_type: {attack_type}. Options: {list(ATTACK_REGISTRY)}")
    return ATTACK_REGISTRY[attack_type](delta, **kwargs)


ALL_ATTACKS = list(ATTACK_REGISTRY) + ["label_flip"]
