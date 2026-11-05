import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from causal_ai.modeling import load_models, score_customers
from causal_ai.optimization import optimize_budget

app = FastAPI(title="Causal AI Customer Growth API", version="1.0.0")

class Customer(BaseModel):
    recency: int = Field(ge=1, le=12)
    history_segment: str
    history: float = Field(ge=0)
    mens: int = Field(ge=0, le=1)
    womens: int = Field(ge=0, le=1)
    zip_code: str
    newbie: int = Field(ge=0, le=1)
    channel: str

class BudgetRequest(BaseModel):
    budget: float = Field(ge=0)
    contact_cost: float = Field(default=0.05, ge=0)
    contribution_margin: float = Field(default=1.0, gt=0)

@app.get("/health")
def health():
    return {"status":"ok", "service":"causal-ai-api"}

@app.get("/metrics")
def metrics():
    import json
    p = Path("artifacts/summary.json")
    if not p.exists():
        raise HTTPException(503, "Run training pipeline first")
    return json.loads(p.read_text())

@app.post("/predict")
def predict(customer: Customer):
    try:
        model = load_models()
        df = pd.DataFrame([customer.model_dump()])
        result = score_customers(df, model).iloc[0]
        return {
            "predicted_spend_without_email": float(result.predicted_spend_control),
            "predicted_spend_with_email": float(result.predicted_spend_treatment),
            "estimated_incremental_spend": float(result.estimated_incremental_spend),
            "recommendation": "Consider targeting" if result.estimated_incremental_spend > 0.05 else "Do not target"
        }
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))

@app.post("/optimize")
def optimize(request: BudgetRequest):
    try:
        model = load_models()
        # Use the real customer pre-treatment population for ranking.
        from causal_ai.data import load_data
        df = load_data()
        scored = score_customers(df, model)
        result = optimize_budget(scored, request.budget, request.contact_cost,
                                 request.contribution_margin)
        return {"customers_targeted": result["customers_targeted"],
                "marketing_cost": result["marketing_cost"],
                "expected_incremental_profit": result["expected_incremental_profit"],
                "unused_budget": result["unused_budget"]}
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
