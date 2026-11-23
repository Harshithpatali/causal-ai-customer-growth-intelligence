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
