import numpy as np
import pandas as pd
import pytest

from causal_ai.benchmarking import run_uplift_benchmark


def _hillstrom_fixture(n=900, seed=91):
    rng = np.random.default_rng(seed)
    segments = np.tile(["No E-Mail", "Mens E-Mail", "Womens E-Mail"], n // 3)
    rng.shuffle(segments)
    history = rng.gamma(2.5, 90, n)
    recency = rng.integers(1, 13, n)
    base = 0.7 + history * 0.003
    effect = np.select(
        [segments == "Mens E-Mail", segments == "Womens E-Mail"],
        [0.35 + (history > 250) * 0.3, 0.15 + (recency < 5) * 0.2],
        default=0.0,
    )
    spend = np.maximum(0, rng.normal(base + effect, 2.5, n))
    spend[rng.random(n) < 0.68] = 0.0
    conversion = (spend > 0).astype(int)
    return pd.DataFrame({
        "recency": recency,
        "history_segment": rng.choice(["low", "medium", "high"], n),
        "history": history,
        "mens": rng.integers(0, 2, n),
        "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Urban", "Rural", "Suburban"], n),
        "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(["Web", "Phone", "Multichannel"], n),
        "segment": segments,
        "spend": spend,
        "conversion": conversion,
        "visit": conversion,
    })


def test_uplift_benchmark_includes_meta_learners_baselines_and_creatives(tmp_path):
    frame = _hillstrom_fixture()
    output = tmp_path / "benchmark.json"
    result = run_uplift_benchmark(
        frame, seed=17, n_bootstrap=100, artifact_path=output
    )

    ranking = result["binary_uplift_benchmark"]
    assert {"T-learner", "S-learner", "X-learner"}.issubset(ranking)
    assert {"Random (seeded)", "Recency (more recent first)",
            "Historical spend (highest first)"}.issubset(ranking)
    for values in ranking.values():
        assert 0 <= values["contact_share"] <= 1
        assert np.isfinite(values["auuc_ht"])
        assert np.isfinite(values["qini_ht"])
        assert "policy_value_ipw" in values

    assert "T-learner vs Email everyone" in result["bootstrap_policy_contrasts_vs_email_everyone"]
    interval = result["bootstrap_policy_contrasts_vs_email_everyone"]["T-learner vs Email everyone"]
    assert interval["bootstrap_ci_lower"] <= interval["bootstrap_ci_upper"]
    assert interval["bootstrap_replicates"] >= 100
    assert "T-learner vs S-learner" in result["pairwise_ranking_policy_bootstrap_contrasts"]
    assert "T-learner vs Random (seeded)" in result["pairwise_ranking_policy_bootstrap_contrasts"]

    creative = result["multi_treatment_average_effects"]
    assert set(creative) == {
        "Mens E-Mail vs No E-Mail",
        "Womens E-Mail vs No E-Mail",
        "Mens E-Mail vs Womens E-Mail",
    }
    assert set(result["multi_treatment_targeting"]) == {
        "Mens E-Mail vs No E-Mail", "Womens E-Mail vs No E-Mail"
    }
    assert output.exists()
    assert set(result["model_feature_importance"]) == {"T-learner", "S-learner", "X-learner"}
    for learner, importance in result["model_feature_importance"].items():
        assert all(value >= 0 for value in importance.values())
        assert np.isclose(sum(importance.values()), 1.0)
    assert set(result["uplift_score_diagnostics"]) == {"T-learner", "S-learner", "X-learner"}
    for diagnostics in result["uplift_score_diagnostics"].values():
        assert diagnostics["histogram"]["counts"]
        assert sum(diagnostics["histogram"]["counts"]) == 225
        assert set(diagnostics["quantiles"]) == {"min", "p10", "p25", "p50", "p75", "p90", "max"}
    assert len(result["uplift_score_heterogeneity_deciles"]["S-learner"]) == 10
    assert "cumulative_curve" in ranking["S-learner"]


def test_uplift_benchmark_rejects_invalid_segments():
    frame = _hillstrom_fixture()
    frame.loc[0, "segment"] = "Unknown"
    with pytest.raises(ValueError, match="segment"):
        run_uplift_benchmark(frame, n_bootstrap=100)


def test_uplift_benchmark_requires_bootstrap_replicates():
    frame = _hillstrom_fixture()
    with pytest.raises(ValueError, match="100"):
        run_uplift_benchmark(frame, n_bootstrap=50)
