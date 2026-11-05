import pandas as pd
import pytest
from causal_ai.optimization import optimize_budget

def test_budget_respected_and_positive_only():
    df = pd.DataFrame({"estimated_incremental_spend":[10,3,-2],
                       "history":[100,50,25]})
    out = optimize_budget(df, budget=0.10, contact_cost=0.05)
    assert out["customers_targeted"] == 2
    assert out["marketing_cost"] <= 0.10
    assert (out["selected"]["expected_incremental_profit"] > 0).all()

def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        optimize_budget(pd.DataFrame({"estimated_incremental_spend":[1]}),-1)
