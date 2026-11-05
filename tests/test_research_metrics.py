import numpy as np
import pytest

from causal_ai.modeling import (
    compare_policy_ipw,
    evaluate_policy_ipw,
    qini_curve,
)


def test_ipw_value_for_always_treat_policy_matches_known_formula():
    outcome = np.array([2.0, 4.0, 1.0, 3.0])
    treatment = np.array([1, 1, 0, 0])
    policy = np.ones(4, dtype=int)

    result = evaluate_policy_ipw(outcome, treatment, policy, propensity=0.5)

    # Only treated observations contribute; divide by assignment probability.
    assert result["value"] == pytest.approx((2.0 + 4.0 + 0.0 + 0.0) / 4 / 0.5)
    assert result["n"] == 4
    assert result["ci_lower"] < result["value"] < result["ci_upper"]


def test_paired_policy_contrast_is_zero_for_identical_policies():
    outcome = np.array([1.0, 2.0, 3.0, 4.0])
    treatment = np.array([1, 0, 1, 0])
    policy = np.array([1, 1, 0, 0])

    result = compare_policy_ipw(outcome, treatment, policy, policy, propensity=2 / 3)

    assert result["difference"] == pytest.approx(0.0)
    assert result["ci_lower"] == pytest.approx(0.0)
    assert result["ci_upper"] == pytest.approx(0.0)


def test_paired_policy_contrast_reports_direction_and_uncertainty():
    outcome = np.array([2.0, 1.0, 4.0, 1.0, 3.0, 2.0])
    treatment = np.array([1, 0, 1, 0, 1, 0])
    policy_a = np.ones(6, dtype=int)
    policy_b = np.zeros(6, dtype=int)

    result = compare_policy_ipw(outcome, treatment, policy_a, policy_b, propensity=0.5)

    assert result["difference"] > 0
    assert result["n"] == 6
    assert result["ci_lower"] <= result["difference"] <= result["ci_upper"]


def test_qini_curve_returns_ipw_gain_and_random_reference():
    outcome = np.array([3.0, 0.0, 2.0, 0.0, 4.0, 0.0])
    treatment = np.array([1, 0, 1, 0, 1, 0])
    uplift = np.array([0.9, 0.8, 0.7, 0.2, 0.1, -0.1])

    result = qini_curve(outcome, treatment, uplift, propensity=0.5)

    assert len(result["fraction"]) == len(result["gain"])
    assert len(result["fraction"]) == len(result["random_gain"])
    assert result["fraction"][0] == 0.0
    assert result["fraction"][-1] == 1.0
    assert np.isfinite(result["qini"])
    assert np.isfinite(result["auuc"])


def test_qini_rejects_invalid_propensity():
    with pytest.raises(ValueError, match="propensity"):
        qini_curve([1, 0], [1, 0], [0.2, -0.1], propensity=1.0)
