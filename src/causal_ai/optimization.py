import numpy as np


def optimize_budget(scored, budget, contact_cost=0.05, contribution_margin=0.4):
    """Build an illustrative equal-cost contact allocation scenario.

    estimated_incremental_spend is treated as incremental revenue per customer.
    contribution_margin is a fraction in [0, 1]. Net contribution is estimated
    incremental revenue * margin - contact cost. These are assumptions, not
    observed costs or validated customer-level profit predictions.
    """
    if not np.isfinite(budget) or not np.isfinite(contact_cost) or not np.isfinite(contribution_margin):
        raise ValueError("Budget, contact cost and contribution margin must be finite")
    if budget < 0 or contact_cost < 0:
        raise ValueError("Budget and contact cost must be non-negative")
    if not 0 <= contribution_margin <= 1:
        raise ValueError("Contribution margin must be a fraction between 0 and 1")
    if "estimated_incremental_spend" not in scored.columns:
        raise ValueError("scored must contain estimated_incremental_spend")

    frame = scored.copy()
    uplift = frame["estimated_incremental_spend"].astype(float)
    frame["modelled_incremental_revenue"] = uplift
    frame["expected_incremental_contribution"] = uplift * contribution_margin
    frame["contact_cost"] = float(contact_cost)
    frame["expected_net_contribution"] = frame["expected_incremental_contribution"] - contact_cost
    frame["expected_incremental_profit"] = frame["expected_net_contribution"]

    eligible = frame[frame["expected_net_contribution"] > 0].sort_values(
        "expected_net_contribution", ascending=False
    ).copy()
    capacity = len(eligible) if contact_cost == 0 else int(np.floor((budget + 1e-12) / contact_cost))
    selected = eligible.head(capacity).copy()
    selected["recommended_action"] = "Send email (scenario)"
    frame["recommended_action"] = "Do not target (scenario)"
    frame.loc[selected.index, "recommended_action"] = "Send email (scenario)"

    contact_spend = float(len(selected) * contact_cost)
    gross_incremental_revenue = float(selected["modelled_incremental_revenue"].sum())
    expected_contribution = float(selected["expected_incremental_contribution"].sum())
    net_contribution = float(selected["expected_net_contribution"].sum())
    return {
        "recommendations": frame.sort_values("expected_net_contribution", ascending=False),
        "selected": selected, "customers_targeted": int(len(selected)),
        "marketing_cost": contact_spend,
        "gross_incremental_revenue": gross_incremental_revenue,
        "expected_incremental_contribution": expected_contribution,
        "expected_net_contribution": net_contribution,
        "expected_incremental_profit": net_contribution,
        "budget": float(budget),
        "unused_budget": float(max(0, budget-contact_spend)),
        "contribution_margin": float(contribution_margin),
        "contact_cost": float(contact_cost),
    }
