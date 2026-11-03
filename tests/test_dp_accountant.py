import math

import pytest

from privacy.dp_accountant import gaussian_mechanism_epsilon, compute_dp_budget


def test_gaussian_mechanism_epsilon_matches_closed_form():
    sigma, sensitivity, delta = 0.01, 1.0, 1e-5
    eps = gaussian_mechanism_epsilon(sigma, sensitivity, delta)
    expected = (sensitivity * math.sqrt(2 * math.log(1.25 / delta))) / sigma
    assert abs(eps - expected) < 1e-9


def test_gaussian_mechanism_epsilon_rejects_bad_delta():
    with pytest.raises(ValueError):
        gaussian_mechanism_epsilon(0.01, 1.0, 1.5)


def test_gaussian_mechanism_epsilon_rejects_bad_sigma():
    with pytest.raises(ValueError):
        gaussian_mechanism_epsilon(0.0, 1.0, 1e-5)


def test_compute_dp_budget_composes_linearly():
    budget_1 = compute_dp_budget(rounds_with_full_defense=1)
    budget_3 = compute_dp_budget(rounds_with_full_defense=3)
    assert abs(budget_3.epsilon_total_basic_composition - 3 * budget_1.epsilon_per_round) < 1e-9


def test_compute_dp_budget_zero_rounds_is_zero_epsilon():
    budget = compute_dp_budget(rounds_with_full_defense=0)
    assert budget.epsilon_total_basic_composition == 0.0
