import numpy as np
import pandas as pd

def optimize_budget(scored, budget, contact_cost=0.05, contribution_margin=1.0):
    """Greedy optimal policy for equal per-customer contact costs."""
    if budget < 0 or contact_cost < 0:
        raise ValueError("Budget and contact cost must be non-negative")
    frame = scored.copy()
    frame["expected_incremental_profit"] = (
        frame["estimated_incremental_spend"] * contribution_margin - contact_cost
    )
    eligible = frame[frame["expected_incremental_profit"] > 0].sort_values(
        "expected_incremental_profit", ascending=False
    ).copy()
    capacity = len(eligible) if contact_cost == 0 else int(budget // contact_cost)
    selected = eligible.head(capacity).copy()
    selected["recommended_action"] = "Send email"
    frame["recommended_action"] = "Do not target"
    frame.loc[selected.index, "recommended_action"] = "Send email"
    spend = float(len(selected)*contact_cost)
    return {
        "recommendations": frame.sort_values("expected_incremental_profit", ascending=False),
        "selected": selected,
        "customers_targeted": int(len(selected)),
        "marketing_cost": spend,
        "expected_incremental_profit": float(selected["expected_incremental_profit"].sum()),
        "budget": float(budget),
        "unused_budget": float(max(0, budget-spend))
    }
