from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from causal_ai.research_reports import (
    load_json_report,
    validate_advanced_statistics,
    validate_research_summary,
    validate_spend_report,
)

st.set_page_config(page_title="Research Analytics | Causal AI", page_icon="📊", layout="wide")
st.title("Research Analytics & Model Validation")
st.caption("Reproducible evidence from randomized holdout policy evaluation, repeated splits and spend-prediction diagnostics.")

REPORTS_DIR = Path("reports")
SUMMARY_REQUIRED = (
    "dataset", "repeated_seed_summary", "winner_counts",
    "primary_seed_policy_values", "bootstrap_vs_email_everyone",
    "creative_average_effects", "methodology",
)


def show_report_problem(label: str, message: str, required: bool = False) -> None:
    if required:
        st.error(f"{label} cannot be displayed: {message}")
        st.info("Run the **Uplift Meta-Learner Benchmark** workflow to regenerate the dashboard snapshots. The workflow builds the JSON reports, validates them in CI, commits the compact snapshots, and uploads the full run artifacts.")
    else:
        st.warning(f"{label} is temporarily unavailable: {message}")


summary, error = load_json_report(REPORTS_DIR / "research_summary.json", SUMMARY_REQUIRED)
if error:
    show_report_problem("Research summary", error, required=True)
    st.stop()

schema_issues = validate_research_summary(summary)
if schema_issues:
    show_report_problem("Research summary", "; ".join(schema_issues), required=True)
    st.stop()

stats_report, stats_error = load_json_report(REPORTS_DIR / "advanced_statistics.json")
if stats_report is not None:
    stats_issues = validate_advanced_statistics(stats_report)
    if stats_issues:
        stats_error = "; ".join(stats_issues)
        stats_report = None
if stats_error:
    show_report_problem("Advanced statistics report", stats_error)

forecast, forecast_error = load_json_report(REPORTS_DIR / "spend_prediction.json")
if forecast is not None:
    forecast_issues = validate_spend_report(forecast)
    if forecast_issues:
        forecast_error = "; ".join(forecast_issues)
        forecast = None
if forecast_error:
    show_report_problem("Spend prediction report", forecast_error)

# Put the decision in front of the technical diagnostics.
st.subheader("Executive conclusion")
if stats_report:
    spend = stats_report["spend"]
    st.markdown(
        f"**Email increased average spend by ${spend['ate']:.3f} per customer** "
        f"(95% CI: ${spend['ci_lower']:.3f} to ${spend['ci_upper']:.3f}) in this randomized historical experiment."
    )
else:
    st.markdown("**The experiment-level email effect is positive in the latest completed research run.**")
st.warning("**The targeting model has not proved better than emailing everyone.** A positive average treatment effect does not validate customer-level uplift rankings.")
s_learner = summary["bootstrap_vs_email_everyone"].get("S-learner vs Email everyone")
if s_learner:
    st.caption(
        "Seed-42 S-learner minus email-everyone policy value: "
        f"${s_learner['estimate']:.3f} per customer "
        f"(paired 95% bootstrap CI: ${s_learner['bootstrap_ci_lower']:.3f} to "
        f"${s_learner['bootstrap_ci_upper']:.3f}). The interval crosses zero."
    )

dataset = summary["dataset"]
st.caption(
    f"Evidence base: {dataset.get('n_total', '—'):,} randomized records · "
    f"{dataset.get('dataset', 'Hillstrom experiment')} · "
    f"primary split seed {dataset.get('seed', '—')}"
)

seed_summary = summary["repeated_seed_summary"]
metric_labels = {
    "policy_value_ipw": "Policy value",
    "auuc_ht": "AUUC",
    "qini_ht": "Qini",
}
metric = st.selectbox(
    "Evaluation metric",
    list(metric_labels),
    format_func=lambda key: metric_labels[key],
)
rows = []
for strategy, values in seed_summary.items():
    item = values[metric]
    rows.append({
        "Strategy": strategy,
        "Mean": item["mean"],
        "SD across splits": item["standard_deviation"],
        "Minimum": item["minimum"],
        "Maximum": item["maximum"],
    })
frame = pd.DataFrame(rows).sort_values("Mean", ascending=False)
st.subheader("Repeated-split strategy comparison")
st.caption("Five stratified partitions of the same 64,000-row experiment; SD is split sensitivity, not an independent-experiment confidence interval.")
st.dataframe(frame, use_container_width=True, hide_index=True)
st.plotly_chart(
    px.bar(frame, x="Strategy", y="Mean", error_y="SD across splits",
           title=f"Mean {metric_labels[metric]} across repeated splits"),
    use_container_width=True,
)

st.subheader("Winner counts")
counts = summary["winner_counts"]
winner_rows = [
    {"Metric": metric_name, "Strategy": strategy, "Wins": wins}
    for metric_name, strategies in counts.items()
    if isinstance(strategies, dict)
    for strategy, wins in strategies.items()
]
if winner_rows:
    winner_frame = pd.DataFrame(winner_rows)
    st.dataframe(
        winner_frame.pivot(index="Strategy", columns="Metric", values="Wins").fillna(0),
        use_container_width=True,
    )
else:
    st.info("Winner-count diagnostics are not included in this report snapshot.")

st.subheader("Primary randomized-holdout policy values")
primary = summary["primary_seed_policy_values"]
primary_rows = [
    {"Strategy": name, "IPW value": item["value"], "95% CI lower": item["ci_lower"],
     "95% CI upper": item["ci_upper"], "AUUC": item["auuc"], "Qini": item["qini"]}
    for name, item in primary.items()
]
st.dataframe(
    pd.DataFrame(primary_rows).sort_values("IPW value", ascending=False),
    use_container_width=True, hide_index=True,
)

st.subheader("Paired policy contrasts against emailing everyone")
contrasts = summary["bootstrap_vs_email_everyone"]
contrast_rows = [
    {"Strategy": name.replace(" vs Email everyone", ""),
     "Difference": value["estimate"],
     "95% CI lower": value["bootstrap_ci_lower"],
     "95% CI upper": value["bootstrap_ci_upper"]}
    for name, value in contrasts.items()
]
st.dataframe(pd.DataFrame(contrast_rows), use_container_width=True, hide_index=True)
st.info("A confidence interval crossing zero does not establish a difference. The paired bootstrap conditions on fitted policies and does not include model-training or model-selection uncertainty.")

if stats_report:
    st.divider()
    st.subheader("Advanced statistical analysis")
    cuped = stats_report["spend_cuped"]
    aipw = stats_report["spend_aipw"]
    s1, s2, s3 = st.columns(3)
    s1.metric("Unadjusted spend ATE", f"${stats_report['spend']['ate']:.3f}")
    s2.metric("CUPED-adjusted ATE", f"${cuped['ate']:.3f}",
              delta=f"{cuped.get('variance_reduction_fraction', 0):.2%} variance reduction")
    s3.metric("Cross-fitted AIPW ATE", f"${aipw['ate']:.3f}")
    st.caption(
        "CUPED 95% CI: [${:.3f}, ${:.3f}] | AIPW 95% CI: [${:.3f}, ${:.3f}]".format(
            cuped["ci_lower"], cuped["ci_upper"], aipw["ci_lower"], aipw["ci_upper"]
        )
    )
    power = stats_report["conversion_power_plan"]
    st.write("**Prospective conversion-test planning**")
    st.write(
        "For a {:.3%} absolute lift from a {:.3%} baseline, the 80% power / 5% alpha "
        "normal approximation requires {:,} control and {:,} treatment observations.".format(
            power["absolute_effect"], power["control_rate"], power["n_control"], power["n_treatment"]
        )
    )
    balance = stats_report["randomization_balance"]
    st.metric("Maximum absolute baseline SMD", f"{balance['max_abs_smd']:.3f}")
    test_rows = [
        {"Outcome": name.title(), "Raw p-value": stats_report[name]["p_value"],
         "Holm-adjusted p-value": stats_report[name]["holm_adjusted_p_value"]}
        for name in ("visit", "conversion", "spend")
        if "holm_adjusted_p_value" in stats_report.get(name, {})
    ]
    if test_rows:
        st.write("Holm correction across the three pooled outcomes")
        st.dataframe(pd.DataFrame(test_rows), use_container_width=True, hide_index=True)
    st.caption("AIPW uses five-fold cross-fitted outcome regressions and the known 2/3 assignment propensity. CUPED uses a pre-treatment covariate. Power is a planning approximation.")

st.subheader("Creative-level randomized effects")
creative_rows = []
for comparison, values in summary["creative_average_effects"].items():
    effect = values["randomized_arm_spend_effect"]
    creative_rows.append({
        "Contrast": comparison, "Spend effect": effect["difference"],
        "95% CI lower": effect["ci_lower"], "95% CI upper": effect["ci_upper"],
    })
if creative_rows:
    st.dataframe(pd.DataFrame(creative_rows), use_container_width=True, hide_index=True)
else:
    st.info("Creative-level contrasts are not present in this snapshot.")

st.divider()
st.subheader("Spend prediction and uncertainty")
if forecast:
    baseline, model = forecast["baseline"], forecast["model_metrics"]
    a, b, c, d = st.columns(4)
    a.metric("Model MAE", f"{model['mae']:.3f}",
             delta=f"{model['mae'] - baseline['mae']:.3f} vs mean baseline (lower is better)")
    b.metric("Model RMSE", f"{model['rmse']:.3f}")
    c.metric("Holdout R²", f"{model['r2']:.3f}")
    interval = model["conformal_interval"]
    d.metric("Conformal coverage", f"{interval['empirical_coverage']:.1%}",
             help=f"Nominal coverage: {interval['nominal_coverage']:.0%}")
    split = forecast["split"]
    st.caption(
        f"Train / calibration / test rows: {split.get('train', '—'):,} / "
        f"{split.get('calibration', '—'):,} / {split.get('test', '—'):,}."
    )
    importance = forecast["feature_permutation_importance"]
    if isinstance(importance, list) and importance:
        imp = pd.DataFrame(importance)
        if {"mae_increase", "feature"}.issubset(imp.columns):
            st.plotly_chart(
                px.bar(imp.sort_values("mae_increase"), x="mae_increase", y="feature",
                       orientation="h", title="Permutation importance (increase in holdout MAE)"),
                use_container_width=True,
            )
    arm_results = forecast["by_randomized_arm"]
    if isinstance(arm_results, dict) and arm_results:
        st.dataframe(
            pd.DataFrame([{"Arm": arm, **values} for arm, values in arm_results.items()]),
            use_container_width=True, hide_index=True,
        )
    st.caption("This is one-period spend prediction, not time-series forecasting. Conformal coverage is marginal under exchangeability, not a guarantee for each subgroup.")
else:
    st.info("Spend prediction diagnostics are omitted until a complete spend_prediction.json snapshot is published.")

with st.expander("Research interpretation and limitations"):
    st.markdown(
        "- The S-learner leads mean policy value across the five supplied splits, but no learned strategy has demonstrated a statistically significant advantage over emailing everyone.
"
        "- Repeated seeds are partitions of the same experiment, not independent replications.
"
        "- Paired bootstrap intervals are conditional on fitted policies and do not include model-training or model-selection uncertainty.
"
        "- Feature importance is predictive, not causal.
"
        "- Hillstrom has no time index, campaign calendar or repeated customer observations. The spend task is cross-sectional, not temporal forecasting.
"
        "- Contact cost and contribution margin are not observed; scenario outputs are not realized profit.
"
        "- Marginal conformal coverage does not guarantee coverage within every subgroup."
    )
