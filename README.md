# Causal AI — Customer Growth & Revenue Optimization

An end-to-end causal machine learning and decision-science application built around Kevin Hillstrom's randomized email marketing experiment (64,000 customers; men's email, women's email, and no-email control).

## What it does

- Analyzes a real randomized marketing experiment.
- Estimates average treatment effects for conversion, visits, and spend.
- Trains a two-part T-learner: conversion probability × conditional positive spend, with separate models for treatment and control.
- Estimates customer-level incremental spend from pre-treatment attributes.
- Ranks customers for a budget-constrained contact policy.
- Serves scoring and optimization through FastAPI.
- Provides a five-page Streamlit decision intelligence dashboard.

## Important interpretation

The randomized experiment supports population-level causal estimates. A customer's two potential outcomes cannot both be observed, so customer-level uplift is a model estimate, not a directly observed individual causal effect. The combined treatment pools two different email creatives. Profit outputs use editable contact-cost and contribution-margin assumptions; they are planning estimates, not realized profit.

## Dataset

The supplied Hillstrom CSV is included at `data/raw/hillstrom.csv`, so the project can train offline after dependencies are installed. If you replace it, preserve the required schema. Dataset description and original challenge: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

The file contains 64,000 rows, three randomized arms, and no missing cells. There are 6,562 exact duplicate rows; these are retained because the source has no unique customer ID and row equality alone does not establish duplicate customer identities.

## Local setup

Python 3.10+ (recommended 3.12).

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux/WSL:
source .venv/bin/activate

pip install -r requirements.txt
python scripts/run_pipeline.py
```

Start Streamlit:

```bash
streamlit run app.py
```

Start FastAPI in another terminal:

```bash
uvicorn backend.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

## API endpoints

- `GET /health`
- `GET /metrics`
- `POST /predict`
- `POST /optimize`

## Tests

```bash
pytest -q
```

The automated unit tests use small deterministic fixtures and do not substitute for the full Hillstrom data run. Run `python scripts/run_pipeline.py` to train and evaluate on the actual dataset.

## Deployment

### FastAPI backend on Render
1. Push this repository to GitHub.
2. In Render, create a new Blueprint from the repository (the included `render.yaml` defines the API service).
3. Render installs dependencies, downloads Hillstrom and trains the models during the build.
4. After deployment, verify `https://YOUR-API.onrender.com/health` and `/docs`.

### Streamlit frontend on Streamlit Community Cloud
1. Create a new app at https://share.streamlit.io/ and connect this GitHub repository.
2. Set the main file path to `app.py`.
3. Streamlit installs `requirements.txt` and starts the app.
4. On first launch, use **Train causal models now**. The app downloads the dataset and trains the local model artifacts.
5. For a connected frontend/backend deployment, add the Render API URL as a Streamlit secret in a later integration step. The current dashboard performs its analysis locally in the Streamlit process, so it works independently.

For a persistent production deployment, store model artifacts in durable object storage or a persistent disk; ephemeral service files can be lost on redeploy. The included Render blueprint trains during deployment, which increases build time.

## GitHub

```bash
git init
git add .
git commit -m "Build causal AI customer growth platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/causal-ai-customer-growth.git
git push -u origin main
```

## Model and evaluation

The current implementation uses a two-part T-learner: logistic regression estimates conversion probability and regularized HistGradientBoostingRegressor models estimate positive spend. Holdout diagnostics include conversion ROC-AUC/average precision, spend RMSE/R², experiment ATE and a Qini-style ranking diagnostic. Spend R² is close to zero on this dataset, so customer-level predictions and ranking are exploratory; benchmark against random targeting and simple business rules, then validate policy value on a fresh randomized holdout before operational use. See `reports/full_evaluation.md` for actual full-dataset results.

## Project layout

```text
app.py                  Streamlit frontend
backend/main.py         FastAPI service
src/causal_ai/data.py   Dataset download, load and validation
src/causal_ai/statistics.py
src/causal_ai/modeling.py
src/causal_ai/optimization.py
src/causal_ai/pipeline.py
scripts/run_pipeline.py
tests/
configs/
```
