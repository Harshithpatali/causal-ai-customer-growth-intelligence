# Decision Brief — Email Experiment and Customer Growth

## Executive decision

**The email campaign increased average customer spend in the randomized experiment. The current uplift-targeting model has not demonstrated that selecting customers is better than emailing everyone.**

**Decision:** Treat the positive average email effect as evidence about the tested campaign mixture in this historical experiment. Do not deploy the current customer-level ranking as a superior targeting policy. A fresh randomized policy test is needed before changing contact strategy.

## The experiment

The public 2008 Hillstrom / MineThatData Email Analytics Challenge contains 64,000 randomized customer records assigned to three approximately equal groups:

- **No E-Mail:** control group
- **Mens E-Mail:** one email creative
- **Womens E-Mail:** another email creative

The primary analysis pools the two email arms and compares assignment to either email with assignment to no email. Under this pooled design, the email assignment propensity is 2/3 and the control propensity is 1/3. The estimand is intention-to-treat: the effect of assignment, whether or not a customer opened or engaged with the email.

## Average treatment effect: what happened on average?

The latest reproducible advanced-statistics report gives the following pooled email-versus-control estimates:

| Outcome | Email group | No-email group | Average effect | 95% confidence interval |
|---|---:|---:|---:|---:|
| Visit | 16.705% | 10.617% | +6.088 percentage points | [+5.544, +6.633] pp |
| Conversion | 1.068% | 0.573% | +0.495 percentage points | [+0.355, +0.636] pp |
| Spend per customer | $1.2496 | $0.6528 | +$0.5968 | [+$0.3762, +$0.8174] |

The spend p-value is approximately 1.15e-7. The AIPW spend estimate is +$0.5943 (95% CI +$0.3739 to +$0.8148), closely agreeing with the unadjusted randomized estimate. Baseline balance is strong: the largest absolute standardized mean difference is about 0.015.

**Interpretation:** assignment to the pooled email treatment increased average spend in this sample. This is spend, not profit. The dataset does not record email delivery cost or contribution margin, and the experiment is historical rather than a forecast of current campaign performance.

## Targeting: did the model improve on emailing everyone?

Average treatment effect and targeting performance answer different questions. A positive average effect says that email helped on average. It does not show that the model can correctly rank which customers benefit most.

On the seed-42 randomized holdout, the S-learner's estimated policy value was $1.1108 per customer. Its paired difference versus emailing everyone was **-$0.0775 per customer**, with a 95% bootstrap interval of **[-$0.4115, +$0.2531]**. The interval crosses zero, so the evaluation does not establish that the S-learner is better or worse than emailing everyone. Other learned policies did not show a reliable advantage either; T-learner and recency-only targeting were significantly worse in this paired bootstrap.

Across five stratified splits of the same 64,000-row dataset, the S-learner had the highest mean policy value and won three of five splits. These are repeated partitions of one experiment, not five independent replications. The result is evidence of split sensitivity, not independent confirmation.

**Decision:** the targeting model is exploratory. Do not claim personalization has won. A new randomized policy experiment should compare the pre-specified targeting rule directly against email-everyone and no-email policies.

## Prediction diagnostics are not targeting proof

The separate one-period spend prediction model achieved MAE 2.081 versus 2.122 for a training-mean baseline, but its RMSE (15.492) was slightly worse than the baseline (15.476) and its R² was -0.0021. This is essentially no useful predictive lift over the baseline. Split-conformal intervals achieved 90.1% empirical marginal coverage against a 90% nominal target, but that is not a subgroup-specific guarantee.

These prediction metrics do not validate CATE estimates or uplift ranking. Predicting who spends is not the same as predicting whose spending changes because of treatment.

## Recommended next action

1. Keep the average treatment effect and the targeting-policy result separate in all stakeholder communication.
2. Obtain actual contact costs and contribution margins before making profitability claims.
3. Pre-register a new randomized policy test: targeted policy versus email-everyone versus no-email.
4. Define the primary outcome, guardrails, sample size, eligibility rules and stopping rule before launch.
5. Evaluate policy value on a randomized holdout and report uncertainty; do not select the policy on the same data used to train it.

## Stakeholder handoff

The intended decision owner is a CRM / lifecycle marketing lead. Before an operational decision, confirm campaign cost, contribution margin, eligible audience, contact capacity and customer-contact frequency limits. This portfolio project does not claim client delivery, employment on a marketing team or external stakeholder sign-off.

## Source and limitations

The source is the public historical Hillstrom randomized email experiment. Individual customer treatment effects are not directly observed. The analysis relies on correct assignment and outcome records, consistency, positivity and no interference. Balance diagnostics are reassuring but cannot prove every randomization assumption. Opens, clicks and visits are post-treatment variables and are not adjustment covariates for the total assignment effect.

**Source of truth:** [Full evaluation report](full_evaluation.md) and the generated JSON snapshots in `reports/`.
