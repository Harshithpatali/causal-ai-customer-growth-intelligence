# Model card: two-part T-learner for email spend uplift

## Intended purpose

This model is an educational research baseline for estimating how expected customer spend differs between the pooled email-treatment condition and the no-email condition in the Hillstrom randomized experiment.

It is not approved for live campaign targeting, customer-level causal claims, or profit forecasting.

## Data

- Source: Kevin Hillstrom / MineThatData Email Analytics Challenge (2008).
- Observations: 64,000.
- Treatment: Mens E-Mail and Womens E-Mail pooled as email; No E-Mail as control.
- Pre-treatment predictors: recency, history segment, historical spend, prior mens/womens purchases, ZIP category, new-customer flag, and channel.
- Outcomes: conversion and spend.

## Model

A separate model is trained for each treatment condition.

1. Logistic regression estimates conversion probability.
2. A regularized histogram gradient-boosting regressor estimates positive spend conditional on conversion.
3. Expected spend is the product of those predictions.
4. Estimated uplift is predicted expected spend under email minus predicted expected spend under no email.

The train/holdout split is stratified by treatment, with a fixed seed of 42 and a 75/25 split. Only pre-treatment variables enter the model.

## Evaluation

The independent holdout is used for spend RMSE and R², conversion discrimination metrics, an inverse-propensity-weighted (IPW) Qini-style ranking diagnostic, and IPW policy-value estimates.

The candidate policy is defined using training-model uplift and evaluated on the randomized holdout. It is compared with emailing everyone and emailing nobody. The paired IPW contrast against emailing everyone is reported with a normal-approximation confidence interval.

## Current evidence and limitations

The current evaluation report records:

- Holdout spend R² close to zero in both treatment conditions.
- Model-policy IPW spend estimate: approximately $1.127 per customer.
- Email-everyone IPW spend estimate: approximately $1.310 per customer.
- Email-nobody IPW spend estimate: approximately $0.671 per customer.
- Model policy minus email-everyone: approximately −$0.183 per customer (95% CI −$0.444 to +$0.077).

The interval includes zero. The model has not demonstrated that it improves on emailing everyone. Spend is sparse and zero-inflated, individual predictions are weak, and the Qini-style diagnostic is exploratory. One split is not sufficient evidence of robust policy value.

## Known risks

- Pooling two creatives can hide creative-specific treatment heterogeneity.
- Individual treatment effects are not directly observable.
- IPW estimates can be noisy, especially with sparse outcomes.
- Near-zero spend R² limits the usefulness of customer-level predictions.
- Contact cost and contribution margin are assumptions absent from the source data.
- The budget optimizer performs deterministic scenario arithmetic; it does not turn weak uplift estimates into validated profit.

## Appropriate interpretation

Use the dashboard to learn and communicate the workflow from randomized experiments to causal ML and decision analysis. Do not use the model's individual scores as factual customer effects or execute its allocation in a real campaign without a prospective randomized policy experiment.

## Meta-learner and creative benchmark extension

The optional benchmark runner `python scripts/run_uplift_benchmark.py` compares T-, S-, and X-learners against seeded random, recency-first, and historical-spend-first ranking strategies on one stratified holdout. The binary meta-learners estimate pooled email versus no-email expected-spend differences. Each ranking policy is evaluated at an approximately 2/3 contact share using the randomized assignment probability.

The report also includes:
- Horvitz-Thompson cumulative-gain AUUC and Qini diagnostics with an explicitly stated random-ranking reference.
- Paired holdout-row bootstrap percentile intervals for policy-value differences.
- Three-arm randomized average spend contrasts for Mens E-Mail, Womens E-Mail, and No E-Mail.
- Model-implied creative-specific effects and pairwise creative-versus-control targeting policy values.

Use restrictions:
- The three binary meta-learners are model comparisons, not proof that one learner identifies true individual effects.
- AUUC/Qini are exploratory ranking diagnostics; the supplied random baseline is a single seeded ranking.
- Bootstrap intervals condition on fitted models and do not capture model-training or model-selection uncertainty.
- Creative-level observed contrasts are average randomized effects. Individual creative effects are model-implied and unobserved.
- Policy value is expected spend per eligible customer, not profit. Contact costs and contribution margins are unavailable in the source data.
- This benchmark uses one held-out partition from the historical dataset; it is not external validation.

## Repeat-split robustness extension

An optional diagnostic in `src/causal_ai/robustness.py` refits the two-part T-learner across several seeded stratified train/test partitions and summarizes variation in targeting share, model-policy value, benchmark policy values, paired model-minus-email-everyone contrasts, and exploratory ranking diagnostics. It is invoked with `python scripts/run_robustness.py` and writes `reports/repeated_split_robustness.json`.

Interpretation safeguards:
- Repeated splits reuse the same historical observations and are not independent experiments.
- Their dispersion describes sensitivity to partition choice; it is not a confidence interval for deployment performance.
- The policy threshold and modelling procedure are fixed across splits; do not select a model from these diagnostics and then report the same splits as confirmatory evidence.
- A prospective randomized policy experiment on a new eligible population remains necessary before operational targeting.

## Recommended next validation

1. Repeat evaluation over multiple stratified random seeds.
2. Compare against random, recency, and historical-spend ranking baselines.
3. Estimate creative-specific or multi-treatment effects.
4. Examine calibration and subgroup stability.
5. Pre-register and run a prospective randomized comparison of the candidate policy against email-everyone.


## Repeated-seed meta-learner diagnostics

The expanded benchmark runner `scripts/run_repeated_seed_diagnostics.py` repeats the ExtraTrees T-, S-, and X-learner comparison over the default seeds 21, 42, 63, 84, and 105. Each split uses the same source experiment with a stratified 75/25 train/holdout partition. It summarizes policy value, AUUC, Qini, rank winners, and split-to-split dispersion.

The diagnostics also publish:

- Aggregated ExtraTrees impurity importance for each learner. For T- and X-learners, the two arm-specific model importances are averaged; S-learner importance includes the treatment indicator. These importances describe predictive use of features and must not be interpreted as causal explanations.
- Holdout uplift-score quantiles, histograms, positive-score share, and Spearman association with pre-treatment historical spend.
- Predicted-uplift decile summaries of mean predicted uplift, mean historical spend, and mean Horvitz-Thompson pseudo-outcome.
- Interactive cumulative-gain, policy-value, AUUC, Qini, bootstrap-contrast, feature-importance, score-distribution, and decile diagnostic charts.

The default repeated-split run uses 100 paired bootstrap replicates per split to keep the sensitivity analysis tractable; the separate primary seed-42 benchmark retains 1,000 replicates by default. The bootstrap intervals condition on fitted models. Repeated splits and their dispersion are not independent replications, external validation, or a deployment confidence interval.

The decile-level HT pseudo-outcome is noisy. It is a descriptive check for whether the predicted ranking aligns with observed randomized holdout outcomes, not a calibrated estimate of conditional average treatment effect. Feature importance cannot establish why historical-spend ranking performs well. A plausible explanation is that historical spend predicts outcome level; demonstrating incremental treatment-benefit prediction requires stronger validation.

## Identification assumptions

The treatment is assigned pooled email versus no-email (intention-to-treat). Identification of the average effect rests on the upstream randomized assignment, consistency of campaign versions, no interference, accurate outcome measurement, and non-zero assignment probabilities. Pooling creatives defines a mixture treatment. Covariates are pre-treatment only; post-assignment opens, clicks, visits and conversion are excluded from the treatment-effect feature set. Baseline balance is descriptive and does not prove randomization. An observational application would require a separately justified DAG, pre-treatment adjustment set, overlap checks, and sensitivity analysis for unmeasured confounding; this model card makes no such observational claim.