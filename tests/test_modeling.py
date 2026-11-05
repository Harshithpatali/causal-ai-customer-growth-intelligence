
import numpy as np
import pandas as pd
from causal_ai.modeling import train_uplift, load_models, score_customers, evaluate_policy_ipw

def test_t_learner_train_save_load_and_score(tmp_path):
    rng = np.random.default_rng(7)
    n = 500
    treatment = np.tile([0, 1], n // 2)
    history = rng.uniform(20, 500, n)
    df = pd.DataFrame({
        "recency": rng.integers(1, 13, n),
        "history_segment": rng.choice(["1) $0 - $100", "2) $100 - $200", "3) $200 - $350"], n),
        "history": history,
        "mens": rng.integers(0, 2, n),
        "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Urban", "Rural", "Surburban"], n),
        "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(["Web", "Phone", "Multichannel"], n),
        "treatment": treatment,
    })
    base = 0.02 * history + rng.normal(0, 2, n)
    df["spend"] = np.maximum(0, base + treatment * (3 + 0.01 * history))
    metrics = train_uplift(df, artifact_dir=tmp_path, seed=11)
    assert (tmp_path/"uplift_model.joblib").exists()
    assert (tmp_path/"metrics.json").exists()
    model = load_models(tmp_path)
    scored = score_customers(df.head(10), model)
    assert len(scored) == 10
    assert "estimated_incremental_spend" in scored
    assert metrics["n_test"] > 0


def test_ipw_policy_value_is_finite():
    y = np.array([1.0, 2.0, 0.0, 4.0])
    treatment = np.array([1, 0, 1, 0])
    policy = np.array([1, 0, 0, 1])
    result = evaluate_policy_ipw(y, treatment, policy, propensity=0.5)
    assert np.isfinite(result["value"])
    assert result["ci_lower"] <= result["value"] <= result["ci_upper"]
