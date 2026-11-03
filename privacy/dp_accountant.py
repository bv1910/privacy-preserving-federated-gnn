"""
Formal (epsilon, delta)-differential-privacy accounting for the Gaussian
noise added by defense.defense_module.FullDefense.

The original README calls the clip+noise step "differential-privacy-style"
without attaching a number. That's not sufficient for a publication claim.
This module computes an actual (epsilon, delta) bound using the classical
Gaussian mechanism guarantee (Dwork & Roth, 2014, Theorem A.1):

    sigma >= (Delta * sqrt(2 * ln(1.25 / delta))) / epsilon

Rearranged for a given sigma (=config.NOISE_SCALE) and L2 sensitivity
Delta (=config.CLIP_THRESHOLD, since the update is clipped to this norm
*before* noise is added):

    epsilon = (Delta * sqrt(2 * ln(1.25 / delta))) / sigma

This is a single-mechanism (single "Full Defense" round) guarantee. Across
multiple rounds where Full Defense fires, privacy composes -- this module
reports both the per-application epsilon and the basic-composition bound
across the rounds actually sanitized (a simple, conservative, defensible
number for a paper; note in the limitations section that an RDP/moments
accountant would give a tighter bound but is out of scope for this build).
"""
import math
from dataclasses import dataclass

import config


@dataclass
class DPBudget:
    sigma: float
    sensitivity: float
    delta: float
    epsilon_per_round: float
    rounds_composed: int
    epsilon_total_basic_composition: float


def gaussian_mechanism_epsilon(sigma: float, sensitivity: float, delta: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    return (sensitivity * math.sqrt(2 * math.log(1.25 / delta))) / sigma


def compute_dp_budget(
    rounds_with_full_defense: int,
    sigma: float = config.NOISE_SCALE,
    sensitivity: float = config.CLIP_THRESHOLD,
    delta: float = 1e-5,
) -> DPBudget:
    """rounds_with_full_defense: number of training rounds in which the
    Coordinator agent actually routed through Full Defense (i.e. noise was
    actually added) -- pass this in from the observed round_summaries, not
    NUM_ROUNDS, since Fast Path rounds add no noise and spend no budget."""
    eps_per_round = gaussian_mechanism_epsilon(sigma, sensitivity, delta)
    eps_total = eps_per_round * max(rounds_with_full_defense, 0)  # basic composition
    return DPBudget(
        sigma=sigma,
        sensitivity=sensitivity,
        delta=delta,
        epsilon_per_round=eps_per_round,
        rounds_composed=rounds_with_full_defense,
        epsilon_total_basic_composition=eps_total,
    )


if __name__ == "__main__":
    # Example: if Full Defense fired on 2 of 8 rounds (as in the original
    # logs/round_results.json run).
    budget = compute_dp_budget(rounds_with_full_defense=2)
    print(budget)
