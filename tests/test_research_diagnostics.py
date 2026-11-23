import numpy as np
import pandas as pd
import pytest

from causal_ai.research_diagnostics import (
    assignment_probability_stress_test,
    randomized_design_diagnostics,
    randomized_pseudo_outcome,
    ranking_metrics,
    subgroup_effects,
)


def synthetic_randomized_data(n=30000, seed=2026, zero_effect=False):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n)
    t = rng.binomial(1, 2 / 3, n)
    tau = np.zeros(n) if zero_effect else 0.02 + 0.12 * x
    p0 = np.full(n, 0.12)
    y = rng.binomial(1, p0 + t * tau)
    return pd.DataFrame({"x": x, "treatment": t, "outcome": y}), tau


def test_known_heterogeneous_effect_is_recovered_in_large_randomized_sample():
    df, true_tau = synthetic_randomized_data()
    pseudo = randomized_pseudo_outcome(df.outcome, df.treatment, 2 / 3)
    assert abs(float(pseudo.mean()) - float(true_tau.mean())) < 0.015
    metrics = ranking_metrics(df.outcome, df.treatment, df.x, 2 / 3, fractions=(0.10, 0.50, 1.0))
    assert metrics["uplift_at"]["10%"]["estimate"] > metrics["uplift_at"]["100%"]["estimate"]


def test_true_zero_effect_does_not_create_material_average_effect():
    df, _ = synthetic_randomized_data(zero_effect=True)
    pseudo = randomized_pseudo_outcome(df.outcome, df.treatment, 2 / 3)
    assert abs(float(pseudo.mean())) < 0.012
    metrics = ranking_metrics(df.outcome, df.treatment, df.x, 2 / 3)
    assert abs(metrics["overall_ate_ipw"]) < 0.012


def test_randomized_design_diagnostics_use_known_probability_not_fitted_propensity():
    df, _ = synthetic_randomized_data(n=1000)
    result = randomized_design_diagnostics(df, propensity=2 / 3)
    assert result["propensity_source"] == "known randomized assignment design"
    assert result["design_overlap"]["individual_propensity_model_fitted"] is False
    assert result["design_overlap"]["positivity_satisfied_by_design"] is True


def test_subgroup_and_probability_stress_diagnostics_are_explicit():
    df, _ = synthetic_randomized_data(n=2000)
    subgroup = subgroup_effects(df, "outcome", subgroup_cols=("x",))
    assert set(subgroup) == {"x"}
    stress = assignment_probability_stress_test(df, "outcome")
    assert "Perturbation stress test only" in stress["interpretation"]
    assert len(stress["results"]) == 3


def test_invalid_propensity_and_misaligned_scores_fail_clearly():
    df, _ = synthetic_randomized_data(n=100)
    with pytest.raises(ValueError):
        randomized_pseudo_outcome(df.outcome, df.treatment, 1.0)
    with pytest.raises(ValueError):
        ranking_metrics(df.outcome, df.treatment, np.zeros(2), 2 / 3)
