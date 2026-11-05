# Full-dataset training and evaluation report

## Dataset verification

- File: `hillstrom.csv`
- Shape: 64,000 rows × 14 original columns
- Missing cells: 0
- Exact duplicate rows: 6,562
- Experiment arms: 3
- Treatment definition used in the pooled analysis: either email creative vs no email.
- Pre-treatment model features: recency, history_segment, history, mens, womens, zip_code, newbie, channel.
- Outcomes: visit, conversion and spend.

Exact duplicate rows were retained. The file has no unique customer identifier, so row-level equality is not sufficient evidence that records are duplicate customer identities. Removing them could alter the randomized sample and treatment proportions.

## Randomized experiment descriptive results

| segment       |   customers |   visit_rate |   conversion_rate |   mean_spend |   total_spend |
|:--------------|------------:|-------------:|------------------:|-------------:|--------------:|
| Mens E-Mail   |       21307 |      0.18276 |           0.01253 |      1.42262 |   30311.69000 |
| No E-Mail     |       21306 |      0.10617 |           0.00573 |      0.65279 |   13908.33000 |
| Womens E-Mail |       21387 |      0.15140 |           0.00884 |      1.07720 |   23038.11000 |

## Pooled email vs no-email experiment estimate

| Outcome | Email mean | Control mean | Difference (ATE) | 95% CI | p-value |
|---|---:|---:|---:|---:|---:|
| Conversion | 0.01068 | 0.00573 | 0.00495 | [0.00355, 0.00636] | 4.98645e-12 |
| Visit | 0.16705 | 0.10617 | 0.06088 | [0.05544, 0.06633] | 1.61043e-106 |
| Spend | $1.24959 | $0.65279 | $0.59680 | [$0.37619, $0.81741] | 1.14837e-07 |

Binary outcomes use a two-proportion z-test with an unpooled standard error. Spend uses Welch's two-sample t-test and Welch-Satterthwaite confidence interval. Email pools men's and women's email creatives; interpret the pooled estimate as the effect of this treatment mixture.

## Model design

Two-part T-learner trained using pre-treatment features only:

1. One logistic-regression conversion model for each treatment arm estimates P(conversion | X, T=t).
2. One regularized conditional spend model per arm estimates E(spend | conversion, X, T=t) using positive-spend training observations.
3. Expected spend under each action is the product of those terms.
4. Estimated incremental spend is mu1(X) - mu0(X).

A 75/25 stratified train/test split was used (random_state=42). Test predictions are saved in `artifacts/test_predictions.csv`. The fitted models are in `artifacts/uplift_model.joblib`.

## Holdout results

- Train rows: 48,000
- Test rows: 16,000
- Holdout treatment spend mean: $1.30915
- Holdout control spend mean: $0.67204
- Holdout spend ATE: $0.63712
- Holdout 95% CI: [$0.20622, $1.06802]
- Holdout ATE p-value: 0.00375861
- Conversion ROC-AUC, control model: 0.65062
- Conversion ROC-AUC, treated model: 0.60998
- Conversion average precision, control model: 0.01245
- Conversion average precision, treated model: 0.01925
- Spend RMSE, control model: $10.78966
- Spend RMSE, treated model: $16.79138
- Spend R², control model: 0.00196
- Spend R², treated model: 0.00022
- Qini-style ranking diagnostic: 1045.60161

## Interpretation and limitations

The experiment-level randomized estimate is the strongest causal result in this project. The model's customer-level treatment effects are estimates, not observable individual causal truths. Spend is sparse and highly zero-inflated. The spend R² values show that the current models explain little of the individual variation in spend; therefore, customer-level rankings and optimizer outputs must be treated as exploratory decision support, not deployment-ready profit guarantees.

The Qini-style value is a ranking diagnostic implemented for this project, not an externally benchmarked or independently validated uplift score. Before a live campaign, evaluate policy value on a fresh randomized holdout, compare against random and business-rule baselines, and run sensitivity analysis on contact cost and contribution margin.

Profit optimization assumes equal contact cost and a user-entered contribution margin multiplier. The dataset does not contain email delivery costs, product-level margins, or realized post-campaign costs.


## Randomized holdout policy-value check (inverse propensity weighting)

The test-set policy sends email when predicted incremental spend exceeds $0.05. The pooled email propensity is set to 2/3 based on the original 1:1:1 allocation across two email arms and one no-email arm.

| Holdout policy | IPW estimated spend per customer | 95% CI |
|---|---:|---:|
| Model policy | $1.12671 | [$0.82008, $1.43334] |
| Email everyone | $1.30993 | [$0.99081, $1.62905] |
| Email nobody | $0.67124 | [$0.38117, $0.96131] |

The current model policy's point estimate is $0.45548 per customer above the all-control policy, but below the email-everyone point estimate. This is not evidence that the uplift policy is better than blanket emailing. Its confidence interval is wide and overlaps the baseline policy intervals. Do not claim validated policy improvement. This benchmark is a critical model-selection result: improve and revalidate the uplift ranking before presenting the optimizer as production-ready.

IPW evaluation is unbiased under the known randomized assignment propensity when the treatment assignment mechanism is correctly specified. Its variance can be high, especially with sparse outcomes. The test set was used for this diagnostic; further model selection requires nested validation or a new untouched randomized experiment.
