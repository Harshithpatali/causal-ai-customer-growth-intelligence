import numpy as np
from causal_ai.statistics import two_proportion_test, mean_difference

def test_two_proportion_test_direction():
    y = np.array([1]*40+[0]*60+[1]*20+[0]*80)
    t = np.array([1]*100+[0]*100)
    result = two_proportion_test(y,t)
    assert result["ate"] > 0
    assert 0 <= result["p_value"] <= 1

def test_mean_difference():
    y = np.array([5,6,7,1,2,3], dtype=float)
    t = np.array([1,1,1,0,0,0])
    result = mean_difference(y,t)
    assert result["ate"] == 4.0


def test_randomized_assignment_diagnostics_detects_perfect_numeric_balance():
    import pandas as pd
    from causal_ai.statistics import randomized_assignment_diagnostics
    frame = pd.DataFrame({"treatment": [1, 1, 0, 0], "x": [1.0, 3.0, 1.0, 3.0],
                          "segment": ["A", "B", "A", "B"]})
    result = randomized_assignment_diagnostics(frame, features=["x", "segment"])
    assert result["max_abs_smd"] == 0.0
    assert result["share_abs_smd_below_0_1"] == 1.0


def test_randomized_assignment_diagnostics_rejects_missing_values():
    import pandas as pd
    import pytest
    from causal_ai.statistics import randomized_assignment_diagnostics
    frame = pd.DataFrame({"treatment": [1, 0], "x": [1.0, float("nan")]})
    with pytest.raises(ValueError, match="missing values"):
        randomized_assignment_diagnostics(frame, features=["x"])


def test_placebo_permutation_test_is_reproducible_and_bounded():
    from causal_ai.statistics import placebo_permutation_test
    result = placebo_permutation_test([1, 2, 1, 0, 0, 1], [1, 1, 1, 0, 0, 0],
                                     n_permutations=500, seed=17)
    repeat = placebo_permutation_test([1, 2, 1, 0, 0, 1], [1, 1, 1, 0, 0, 0],
                                      n_permutations=500, seed=17)
    assert result == repeat
    assert 0 < result["p_value_two_sided"] <= 1


def test_aipw_matches_randomized_difference_when_nuisance_models_are_constant():
    from causal_ai.statistics import aipw_average_treatment_effect
    y = np.array([3.0, 5.0, 1.0, 2.0, 4.0, 0.0])
    t = np.array([1, 1, 1, 0, 0, 0])
    result = aipw_average_treatment_effect(y, t, 0.5, np.repeat(2.0, 6), np.repeat(2.0, 6))
    assert np.isfinite(result["ate"])
    assert result["ci_lower"] <= result["ate"] <= result["ci_upper"]


def test_cuped_uses_pre_treatment_covariate_and_reports_variance_reduction():
    from causal_ai.statistics import cuped_adjusted_ate
    x = np.arange(20, dtype=float)
    t = np.tile([0, 1], 10)
    y = 2 * x + 3 * t + np.random.default_rng(3).normal(0, 0.2, 20)
    result = cuped_adjusted_ate(y, t, x)
    assert abs(result["ate"] - 3.0) < 0.2
    assert result["variance_reduction_fraction"] > 0.9


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    from causal_ai.statistics import holm_adjust
    p = [0.01, 0.04, 0.03]
    adjusted = holm_adjust(p)
    assert adjusted[0] <= adjusted[2] <= adjusted[1]
    assert all(0 <= value <= 1 for value in adjusted)


def test_two_proportion_sample_size_returns_valid_design():
    from causal_ai.statistics import two_proportion_sample_size
    result = two_proportion_sample_size(0.05, 0.06, power=0.8)
    assert result["n_control"] > 0
    assert result["n_treatment"] > 0
    assert result["total_n"] == result["n_control"] + result["n_treatment"]
