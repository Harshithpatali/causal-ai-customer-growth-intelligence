# Decision Brief — Email Experiment and Customer Growth

## Executive decision

**Decision owner:** CRM / lifecycle marketing lead  
**Decision:** Treat the randomized email experiment as evidence that the tested email treatment mixture increased average customer response and spend in this historical sample. Do not deploy the current customer-level uplift ranking as a superior targeting policy.

## Evidence from the randomized experiment

The analysis covers 64,000 randomized records across two email arms and a no-email control. Pooling the two email creatives:

| Outcome | Email | No email | Absolute effect | 95% CI | p-value |
|---|---:|---:|---:|---:|---:|
| Conversion | 1.068% | 0.573% | +0.495 percentage points | [+0.355, +0.636] pp | 4.99e-12 |
| Visit | 16.705% | 10.617% | +6.088 percentage points | [+5.544, +6.633] pp | 1.61e-106 |
| Spend per customer | $1.2496 | $0.6528 | +$0.5968 | [+$0.3762, +$0.8174] | 1.15e-7 |

The spend effect is an experimental average effect for the pooled email mixture, not profit. The dataset does not include campaign delivery costs or contribution margins.

## What the evidence supports

1. The tested email treatment mixture increased average visit, conversion and spend relative to no email in this experiment.
2. The uplift model's customer-level policy did not outperform emailing everyone on the randomized holdout: estimated spend was $1.1267 versus $1.3099 per customer.
3. The model's spend R² is close to zero, so customer-level rankings should remain exploratory.

## Recommended next action

- Use the experiment-level effect to inform a controlled campaign discussion, not a blanket profitability claim.
- Obtain actual contact cost and contribution margin before calculating incremental contribution.
- Run a fresh randomized policy test comparing the proposed targeted policy with email-everyone and a no-email control.
- Predefine the primary outcome, guardrails, sample size and stopping rule with the campaign owner before launch.

## Stakeholder handoff

For a CRM/product review, present the randomized effect table first, then the holdout policy comparison and assumptions. Ask the decision owner to confirm campaign cost, margin, eligible audience, capacity and acceptable customer-contact frequency. These are the inputs required to turn experimental lift into an operational decision.

## Collaboration and provenance

This is an independently developed portfolio research project using a public historical experiment. The stakeholder role above describes the intended decision workflow; it is **not a claim of client delivery, employment on a marketing team, or external stakeholder sign-off**.

**Source of truth:** [Full evaluation report](full_evaluation.md).