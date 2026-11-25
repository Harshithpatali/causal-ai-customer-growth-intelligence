from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Research Analytics | Causal AI", page_icon="📊", layout="wide")
st.title("Research Analytics & Model Validation")
st.caption("Reproducible evidence from the five-split uplift benchmark and the independent spend-prediction evaluation.")

summary_path = Path("reports/research_summary.json")
forecast_path = Path("reports/spend_prediction.json")

if not summary_path.exists():
    st.warning("Research summary is not present yet. Run the Uplift Meta-Learner Benchmark GitHub Actions workflow.")
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
seed_summary = summary["repeated_seed_summary"]
metric = st.selectbox("Evaluation metric", ["policy_value_ipw", "auuc_ht", "qini_ht"],
                      format_func=lambda x: {"policy_value_ipw": "Policy value", "auuc_ht": "AUUC", "qini_ht": "Qini"}[x])
rows = []
for strategy, values in seed_summary.items():
    item = values[metric]
    rows.append({"Strategy": strategy, "Mean": item["mean"], "SD across splits": item["standard_deviation"],
                 "Minimum": item["minimum"], "Maximum": item["maximum"]})
frame = pd.DataFrame(rows).sort_values("Mean", ascending=False)
st.subheader("Repeated-split strategy comparison")
st.caption("Five stratified partitions of the same 64,000-row experiment; SD is split sensitivity, not an independent-experiment confidence interval.")
st.dataframe(frame, use_container_width=True, hide_index=True)
fig = px.bar(frame, x="Strategy", y="Mean", error_y="SD across splits",
             title=f"Mean {metric.replace('_', ' ').upper()} across five splits")
fig.update_layout(xaxis_tickangle=-25)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Winner counts")
counts = summary["winner_counts"]
winner_frame = pd.DataFrame([
    {"Metric": metric_name, "Strategy": strategy, "Wins": wins}
    for metric_name, strategies in counts.items()
    for strategy, wins in strategies.items()
])
st.dataframe(winner_frame.pivot(index="Strategy", columns="Metric", values="Wins").fillna(0),
             use_container_width=True)

st.subheader("Primary seed-42 holdout policy values")
primary = summary["primary_seed_policy_values"]
primary_rows = [{"Strategy": name, "IPW value": item["value"],
                 "95% CI lower": item["ci_lower"], "95% CI upper": item["ci_upper"],
                 "AUUC": item["auuc"], "Qini": item["qini"]}
                for name, item in primary.items()]
st.dataframe(pd.DataFrame(primary_rows).sort_values("IPW value", ascending=False),
             use_container_width=True, hide_index=True)

st.subheader("Paired policy contrasts against emailing everyone")
contrasts = summary["bootstrap_vs_email_everyone"]
contrast_rows = [{"Strategy": name.replace(" vs Email everyone", ""),
                  "Difference": value["estimate"],
                  "95% CI lower": value["bootstrap_ci_lower"],
                  "95% CI upper": value["bootstrap_ci_upper"]}
                 for name, value in contrasts.items()]
st.dataframe(pd.DataFrame(contrast_rows), use_container_width=True, hide_index=True)
st.info("A confidence interval crossing zero does not establish a difference. The bootstrap conditions on fitted policies and does not include model-training uncertainty.")

st.subheader("Creative-level randomized effects")
creative = summary["creative_average_effects"]
creative_rows = []
for comparison, values in creative.items():
    effect = values["randomized_arm_spend_effect"]
    creative_rows.append({"Contrast": comparison, "Spend effect": effect["difference"],
                          "95% CI lower": effect["ci_lower"], "95% CI upper": effect["ci_upper"]})
st.dataframe(pd.DataFrame(creative_rows), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Spend prediction and uncertainty")
if not forecast_path.exists():
    st.warning("Spend prediction report has not been generated. Run the benchmark workflow after this upgrade is merged.")
else:
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    baseline, model = forecast["baseline"], forecast["model_metrics"]
    a, b, c, d = st.columns(4)
    a.metric("Model MAE", f"{model['mae']:.3f}", delta=f"{model['mae']-baseline['mae']:.3f} vs mean baseline (lower is better)")
    b.metric("Model RMSE", f"{model['rmse']:.3f}")
    c.metric("Holdout R²", f"{model['r2']:.3f}")
    d.metric("Conformal coverage", f"{model['conformal_interval']['empirical_coverage']:.1%}",
             help=f"Nominal coverage: {model['conformal_interval']['nominal_coverage']:.0%}")
    st.caption(f"Train / calibration / test rows: {forecast['split']['train']:,} / {forecast['split']['calibration']:,} / {forecast['split']['test']:,}.")
    imp = pd.DataFrame(forecast["feature_permutation_importance"])
    st.plotly_chart(px.bar(imp.sort_values("mae_increase"), x="mae_increase", y="feature", orientation="h",
                           title="Permutation importance (increase in holdout MAE)"), use_container_width=True)
    st.dataframe(pd.DataFrame([
        {"Arm": arm, **values} for arm, values in forecast["by_randomized_arm"].items()
    ]), use_container_width=True, hide_index=True)
    st.caption("This is one-period spend prediction, not time-series forecasting. Conformal coverage is marginal under exchangeability, not a guarantee for each subgroup.")

with st.expander("Research interpretation and limitations"):
    st.markdown(
        "- The S-learner currently leads mean policy value, AUUC and Qini across five splits, but split sensitivity remains and no strategy has demonstrated a statistically significant advantage over emailing everyone.\n"
        "- The primary-seed bootstrap intervals are conditional on fitted policies. They do not include model-training or model-selection uncertainty.\n"
        "- Feature importance is predictive, not causal.\n"
        "- Hillstrom has no time index, campaign calendar or repeated customer observations. Do not label this dataset's spend task as temporal forecasting.\n"
        "- Contact cost and contribution margin are not observed; do not call scenario outputs realized profit."
    )
