# Causal AI — Customer Growth & Revenue Optimization

An end-to-end causal machine learning and decision-science application built around the Kevin Hillstrom randomized email marketing experiment (64,000 customers; men's email, women's email, and no-email control).

## Capabilities

- Randomized experiment analysis and average treatment effect estimation.
- Two-part T-learner for conversion probability and conditional positive spend.
- Customer-level incremental-spend estimates using pre-treatment features.
- Budget-constrained customer ranking and contact allocation.
- Five-page Streamlit decision-intelligence dashboard.
- FastAPI scoring and optimization endpoints.
- Automated tests and model-evaluation report.

## Statistical interpretation

The randomized experiment supports population-level causal estimates. Individual customer uplift is model-estimated because both potential outcomes cannot be observed for the same customer. The model-policy inverse-propensity-weighted holdout value is currently below the email-everyone benchmark ($1.14 vs. $1.31 per customer in the evaluated holdout). Therefore, targeting performance is exploratory and is not claimed to outperform a simple baseline. Profit outputs also depend on editable contact-cost and contribution-margin assumptions and are planning estimates, not realized profit.

## Dataset

The project uses the Kevin Hillstrom / MineThatData email marketing experiment. The CSV is included in `data/raw/hillstrom.csv`; no customer identifiers or private credentials are included. The source data has 64,000 rows across three randomized groups. Exact duplicate rows are retained because the source has no unique customer ID and row equality alone does not prove duplicate customer identity.

Original dataset description: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

## Local setup

Python 3.10+ (Python 3.12 recommended).

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux/WSL:
source .venv/bin/activate

pip install -r requirements.txt
python scripts/run_pipeline.py
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

Run the FastAPI backend in another terminal:

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

The automated tests use deterministic fixtures. Run `python scripts/run_pipeline.py` to regenerate the full-data model and evaluation artifacts.

## Deployment

### FastAPI on Render

The repository includes `render.yaml`. Create a Render Blueprint from this repository and review the plan before applying it. The current blueprint installs dependencies and trains the model during deployment, so the build may take longer than a simple API deploy.

### Streamlit Community Cloud

Create an app at https://share.streamlit.io/, connect this repository, and set the main file to `app.py`. Install dependencies from `requirements.txt`. The current dashboard runs its analysis in the Streamlit process; a later integration step can connect it to the deployed FastAPI service.

For persistent production use, store model artifacts in durable object storage or persistent storage. Do not rely on ephemeral service files surviving redeployments.

## Evaluation

See [`reports/full_evaluation.md`](reports/full_evaluation.md) for experiment estimates, holdout diagnostics, and policy benchmarks. Spend prediction R² is close to zero on this dataset, so individual spend predictions and uplift rankings should be treated as exploratory. A policy should be validated on a fresh randomized holdout and compared with random targeting and simple business-rule baselines before operational use.

## Project structure

```text
app.py                  Streamlit frontend
backend/main.py         FastAPI service
src/causal_ai/data.py   Dataset loading and validation
src/causal_ai/statistics.py
src/causal_ai/modeling.py
src/causal_ai/optimization.py
src/causal_ai/pipeline.py
scripts/run_pipeline.py
artifacts/               Fitted model and compact metrics
reports/                 Dataset profile and evaluation report
tests/
configs/
```

## License

Dataset rights and terms remain with the original dataset provider. Add a project license only after choosing terms for your own code and verifying compatibility with the dataset terms.
