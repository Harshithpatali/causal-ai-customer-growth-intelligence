# Interview Demonstration — Causal AI Customer Growth Intelligence

## The 5-minute walkthrough

### 1. Start with the business question (30 seconds)

“I wanted to answer two questions separately: does an email campaign increase customer outcomes on average, and can we improve on emailing everyone by targeting customers based on their likely incremental response? I used the public 64,000-row Hillstrom randomized email experiment.”

### 2. Explain the experiment (45 seconds)

“Customers were randomly assigned to No E-Mail, Mens E-Mail or Womens E-Mail. For the primary analysis I pooled the two email creatives into one treatment group and compared that group with no email. The pooled treatment assignment probability is 2/3 and the control probability is 1/3. Because assignment was randomized, the difference in group means estimates the intention-to-treat average effect, subject to the usual consistency, positivity and no-interference assumptions.”

Be precise:
- The estimand is the effect of assignment, not the effect of opening an email.
- Do not adjust for post-treatment opens, clicks or visits when estimating the total assignment effect.
- Baseline balance is a diagnostic, not proof that all assumptions hold.

### 3. Explain ATE versus CATE / uplift (60 seconds)

**ATE** is the average treatment effect across the study population:

`ATE = E[Y(1) - Y(0)]`

**CATE** is the conditional average treatment effect for customers with characteristics `X=x`:

`CATE(x) = E[Y(1) - Y(0) | X=x]`

An uplift model tries to estimate CATE for different customer profiles and rank customers by expected incremental response. It does not observe both potential outcomes for one person. Each customer receives only one randomized assignment, so individual treatment effects are counterfactual and must be estimated from patterns across comparable customers.

A standard outcome model predicts `Y`. An uplift model targets the difference between the expected outcome under treatment and the expected outcome under control. A customer who is likely to spend a lot is not necessarily a customer whose spend is changed by an email.

### 4. Explain IPW policy evaluation (60 seconds)

The model turns its estimated uplift scores into a policy: email this customer or do not email them. To evaluate that rule, I used the randomized holdout and inverse-propensity weighting (IPW).

For a policy `π(X)`, the estimator weights an observation when its randomized assignment matches the policy action:

`V_hat(π) = (1/n) Σ [ I(A_i = π(X_i)) Y_i / P(A_i = π(X_i)) ]`

In this experiment, the matching probability is 2/3 when the policy says email and 1/3 when it says no email. The weights correct for the known randomized assignment probabilities. This lets us estimate how the policy would perform without pretending we observed both outcomes for each customer.

IPW depends on correct assignment probabilities, positivity and a policy specified without using the evaluation outcomes. Large or unstable weights can increase variance; here the assignment probabilities are known from the randomized design.

### 5. Explain the key result (60 seconds)

“The pooled email treatment increased average spend by about $0.597 per customer, with a 95% confidence interval of about $0.376 to $0.817. Conversion increased by about 0.495 percentage points. So the average email effect is positive in this historical experiment.

“But that does not mean the targeting model works. The S-learner’s estimated policy value on the seed-42 holdout was about $1.111 per customer. Its paired difference against emailing everyone was -$0.077 per customer, with a 95% bootstrap interval from -$0.412 to +$0.253. Since that interval crosses zero, I cannot claim that the targeting policy is better than emailing everyone.”

### 6. Explain why positive ATE does not guarantee successful targeting (45 seconds)

A positive ATE is an average. It can coexist with:
- little meaningful treatment-effect variation;
- real variation that the available features do not explain;
- noisy CATE estimates that rank customers poorly;
- a threshold or budget rule that selects the wrong customers;
- a targeting policy whose costs and operational constraints erase any gross-spend gain.

Even when CATEs vary, a model must rank them accurately enough for the chosen policy to beat a simple baseline on untouched randomized data. This project has not shown that yet.

## Questions an interviewer may ask

**Q: If email has a positive ATE, why not email everyone?**  
A: That is a reasonable baseline for this dataset. The experiment supports a positive average spend effect, but it does not measure contact costs or margin, and it is historical. I would validate the economics and run a current randomized campaign before making a live rollout decision.

**Q: What does IPW do intuitively?**  
A: It gives more weight to observed customers whose randomized action was less likely under the design, so the weighted sample represents the actions a specified policy would take. Since the assignment probabilities are known, we can evaluate the policy on randomized holdout data.

**Q: Is a CATE prediction the true individual treatment effect?**  
A: No. It is a conditional average for customers with similar observed covariates. We never observe both potential outcomes for the same customer.

**Q: Why use T-, S- and X-learners?**  
A: They provide different ways to estimate the two conditional outcome functions or treatment-effect function. Comparing them against simple baselines helps show whether the modelling choice improves ranking and policy value rather than merely fitting outcomes.

**Q: Did the spend prediction model perform well?**  
A: No. MAE improved slightly over the mean baseline, but RMSE was worse and R² was slightly negative. I report that honestly. Predictive spend accuracy is also not the same thing as uplift accuracy.

**Q: What would you do next?**  
A: Diagnose heterogeneity and ranking stability, keep the targeting model exploratory, obtain real cost and margin inputs, and run a pre-registered randomized test comparing the targeted policy with email-everyone and no-email.

## Demo checklist

- [ ] Open Executive summary and state the two-part conclusion before showing charts.
- [ ] Show the three randomized arms and explain why the primary analysis pools the two email creatives.
- [ ] Open Experiment evidence and explain the spend ATE and confidence interval.
- [ ] Open Uplift model diagnostics and distinguish estimated CATE scores from observed individual effects.
- [ ] Open Research Analytics and show the paired IPW policy contrast against emailing everyone.
- [ ] State the limitations: historical data, no delivery cost or margin, no individual counterfactuals, weak spend prediction and no demonstrated targeting superiority.
