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
