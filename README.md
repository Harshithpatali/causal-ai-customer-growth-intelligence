# Causal AI — Customer Growth & Revenue Optimization

An end-to-end causal machine-learning and decision-science application built around the Kevin Hillstrom / MineThatData randomized email marketing experiment. It combines experiment analysis, customer-level uplift modelling, holdout policy evaluation, and budget-constrained planning in a five-page Streamlit dashboard with a FastAPI service.

> **Research status:** the randomized experiment shows a positive average effect for the pooled email treatment, but the current uplift targeting policy has not demonstrated that it outperforms emailing everyone. Spend prediction R² is close to zero on the holdout. Customer-level rankings and contribution scenarios are exploratory—not validated production recommendations.

## Product modules

1. **Executive Overview** — experiment-arm outcomes and evidence hierarchy.
2. **Customer Intelligence** — pre-treatment customer attributes and model-implied outcomes.
3. **Experimentation Lab** — average treatment effects, confidence intervals, and statistical tests.
4. **Uplift Intelligence** — two-part T-learner scores, inverse-propensity-weighted (IPW) holdout policy values, and cumulative-gain diagnostic.
5. **Budget Scenario Optimizer** — equal-cost contact allocation under editable cost and contribution-margin assumptions.

## Dataset

The project uses the 64,000-row Kevin Hillstrom email marketing experiment, with three randomized arms: Mens E-Mail, Womens E-Mail, and No E-Mail. The raw CSV is **not committed** to this repository. On first run, the application downloads it to `data/raw/hillstrom.csv`; see [`data/README.md`](data/README.md) for details. Confirm upstream dataset redistribution terms before mirroring it elsewhere.

The dataset has no unique customer identifier. It contains 6,562 exact duplicate rows, which are retained: identical row values do not prove that two records belong to the same customer.

## Methodology

### Randomized experiment

The primary estimand pools both email creatives into one treatment group and compares that treatment mixture with no email. Binary outcomes use a two-proportion z test with a pooled null standard error and an unpooled confidence interval. Spend uses Welch's two-sample t test with a Satterthwaite confidence interval.

### Uplift model

A two-part T-learner is trained separately in the email and control groups:
- Logistic regression estimates conversion probability under each action.
- Regularized gradient-boosting regressors estimate positive spend conditional on conversion.
- Expected spend is estimated as `P(conversion) × E(spend | conversion)`.
- Estimated incremental spend is expected spend with email minus expected spend without email.

Only pre-treatment features are used: `recency`, `history_segment`, `history`, `mens`, `womens`, `zip_code`, `newbie`, and `channel`.

### Policy evaluation

A 75/25 stratified train/holdout split is used. The model policy is selected using training data and evaluated on the randomized holdout with the known 2/3 email assignment propensity. The report includes IPW values for the model policy, emailing everyone, and emailing nobody, plus a paired IPW comparison of model targeting against emailing everyone.

### Budget scenario

```text
modelled net contribution per contacted customer
  = estimated incremental revenue × contribution margin fraction
    − email contact cost
```

With equal contact costs, selecting the highest positive net-contribution scores up to the budget is optimal for the stated point estimates. This is **scenario arithmetic**, not realized profit. The dataset does not contain email delivery costs or product contribution margins, and model scores are not sufficiently validated for a live policy.

## Current evaluation snapshot

See [`reports/full_evaluation.md`](reports/full_evaluation.md) for the reproducible metrics and limitations. In the current seeded holdout run:

| Holdout policy | IPW spend estimate per customer |
|---|---:|
| Model policy | $1.127 |
| Email everyone | $1.310 |
| Email nobody | $0.671 |

The paired model-policy minus email-everyone estimate is **−$0.183 per customer** (95% CI: −$0.444 to +$0.077). The interval includes zero; this does not establish that the model policy is better or worse than emailing everyone.

Spend prediction R² is approximately 0.002 in control and 0.0002 in treatment. Treat customer-level uplift as exploratory. A new randomized policy experiment is needed before claiming targeting performance.

## Local setup

Python 3.10+ (Python 3.12 recommended).

```bash
python -m venv .venv

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS / Linux / WSL
source .venv/bin/activate

pip install -r requirements.txt
python scripts/run_pipeline.py
pytest -q
streamlit run app.py
```

Start the FastAPI service in another terminal:

```bash
uvicorn backend.main:app --reload
```

API documentation: http://127.0.0.1:8000/docs

## API endpoints

- `GET /health`
- `GET /metrics`
- `POST /predict`
- `POST /optimize`

For `/optimize`, `contribution_margin` is a fraction from `0` to `1` (for example, `0.40` means 40%).

## Tests

```bash
pytest -q
```

GitHub Actions runs compilation and the automated test suite on pushes and pull requests. The full dataset training run is separate from CI.

## Deployment

### FastAPI on Render

The repository contains `render.yaml`. Create a Render Blueprint and review planned resources before applying it. The build downloads the public dataset if needed and trains the models; it may take longer than a simple API build.

### Streamlit Community Cloud

Create an app at https://share.streamlit.io/, connect this repository, and set the main file to `app.py`. Streamlit installs `requirements.txt`. On a fresh deployment, use the dashboard's **Train causal models now** action to download data and create model artifacts.

For persistent production use, store model artifacts in durable object storage or a persistent disk. The dashboard currently scores in its own Streamlit process; connecting the frontend to the deployed FastAPI service can be a subsequent integration step.

## Project structure

```text
app.py                  Streamlit frontend
backend/main.py         FastAPI service
src/causal_ai/data.py   Dataset download, load and validation
src/causal_ai/statistics.py
src/causal_ai/modeling.py
src/causal_ai/optimization.py
src/causal_ai/pipeline.py
scripts/run_pipeline.py
reports/                 Evaluation report and data profile
tests/                   Automated tests
configs/                 Project configuration
```

## Data and licensing note

The raw Hillstrom data is downloaded from an upstream source and is not committed here. Dataset terms remain with the original provider. This repository does not add a license for the dataset or project code; choose a code license separately after reviewing the source-data terms.
