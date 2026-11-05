from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from causal_ai.data import load_data, validate_data
from causal_ai.modeling import load_models, score_customers
from causal_ai.statistics import two_proportion_test, mean_difference
from causal_ai.optimization import optimize_budget
from causal_ai.pipeline import run_pipeline


st.set_page_config(
    page_title="Causal AI | Customer Growth",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(circle at 15% 0%,#13243b 0,#0a1220 36%,#080e18 100%);color:#e8eef8;}
section[data-testid="stSidebar"] {background:#0c1727;border-right:1px solid #20324a;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;color:#f3f7ff;}
div[data-testid="stMetric"] {background:linear-gradient(145deg,#14243a,#101b2d);border:1px solid #263b56;padding:16px;border-radius:14px;}
div[data-testid="stMetricLabel"] {color:#9eb2ca;}
div[data-testid="stMetricValue"] {color:#f3f7ff;}
.block-container {padding-top:2rem;max-width:1500px;}
div.stButton>button {background:#39d6b4;color:#07131d;border:0;border-radius:9px;font-weight:700;}
.hero {padding:26px 30px;border-radius:18px;background:linear-gradient(115deg,#142b43,#102033 65%,#153a43);border:1px solid #294b63;margin-bottom:22px;}
.hero-kicker {color:#51dfc0;letter-spacing:2px;font-size:11px;font-weight:700;}
.hero-title {font-family:'Space Grotesk';font-size:34px;font-weight:700;color:#f5f8ff;margin:6px 0;}
.hero-sub {color:#b4c5d8;font-size:14px;}
.business-callout {padding:18px 20px;border:1px solid #31536a;border-radius:14px;background:#10263a;margin:10px 0 18px 0;}
.muted {color:#a9bbce;font-size:0.9rem;}
div[data-testid="stDataFrame"] {border:1px solid #263b56;border-radius:12px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


@st.cache_resource(show_spinner=False)
def get_model():
    return load_models()


def header(kicker, title, subtitle):
    st.markdown(
        f'<div class="hero"><div class="hero-kicker">{kicker}</div>'
        f'<div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def money(value):
    return f"${value:,.2f}"


def explain(label, detail):
    st.markdown(f"**{label}**")
    st.caption(detail)


st.sidebar.markdown("## ◈ CAUSAL AI")
st.sidebar.caption("CUSTOMER GROWTH INTELLIGENCE")
mode = st.sidebar.radio(
    "WHO IS THIS VIEW FOR?",
    ["Executive view", "Data science view"],
    help="Executive view translates the findings into business decisions. Data science view exposes the statistical analysis and model diagnostics.",
)
st.sidebar.divider()

try:
    df = get_data()
except Exception as exc:
    st.error(f"Dataset unavailable: {exc}")
    st.info("The app downloads the Hillstrom dataset on first launch. Check the dataset source or try again.")
    st.stop()

validation = validate_data(df)
if not validation["valid"]:
    st.error(validation["issues"])
    st.stop()

try:
    models = get_model()
except Exception:
    st.warning("The trained model artifacts are not available in this app session.")
    st.write("The experiment data is ready. Train the models to enable customer scoring and campaign scenarios.")
    if st.button("Train models", type="primary"):
        with st.spinner("Training the causal models. This can take a few minutes..."):
            run_pipeline()
            st.cache_resource.clear()
        st.rerun()
    st.stop()

scored = score_customers(df, models)
metrics_path = Path("artifacts/metrics.json")
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

# These are the same experiment estimators and trained-model outputs used by the research view.
ate_conv = two_proportion_test(df.conversion, df.treatment)
ate_spend = mean_difference(df.spend, df.treatment)
comparison = metrics.get("policy_vs_all_email_ipw", {})
policy_values = metrics.get("policy_value_ipw", {})
all_email = metrics.get("all_email_value_ipw", {})
no_email = metrics.get("all_control_value_ipw", {})

if mode == "Executive view":
    page = st.sidebar.radio(
        "BUSINESS WORKSPACE",
        ["Executive summary", "Customer insights", "Campaign planner", "AI Business Reports"],
    )

    if page == "Executive summary":
        header(
            "BUSINESS DECISION BRIEF",
            "Did the email campaign work?",
            "A plain-English summary of a historical randomized marketing experiment.",
        )
        st.caption(f"Evidence base: {len(df):,} randomized customer records · Hillstrom email experiment")

        conversion_lift_pp = ate_conv["ate"] * 100
        spend_lift = ate_spend["ate"]
        significant = ate_conv["p_value"] < 0.05 and ate_spend["p_value"] < 0.05

        a, b, c = st.columns(3)
        a.metric("Conversion improvement", f"{conversion_lift_pp:+.2f} percentage points")
        b.metric("Additional spend per customer", money(spend_lift))
        c.metric("Customers in experiment", f"{len(df):,}")

        st.markdown("### The business takeaway")
        st.markdown(
            '<div class="business-callout"><h3>Emails increased average customer response and spend.</h3>'
            '<p class="muted">The experiment randomly assigned customers to email or no-email groups. '
            'That makes the overall campaign comparison stronger evidence than a simple before-and-after comparison.</p></div>',
            unsafe_allow_html=True,
        )
        if significant:
            st.success("Both the conversion and spend comparisons are statistically significant at the 5% level.")
        else:
            st.info("At least one of the primary outcome comparisons is not statistically significant at the 5% level.")

        left, right = st.columns(2)
        with left:
            st.subheader("What happened to conversion?")
            conv = pd.DataFrame({
                "Group": ["No email", "Email campaign"],
                "Conversion rate": [
                    df.loc[df.treatment == 0, "conversion"].mean(),
                    df.loc[df.treatment == 1, "conversion"].mean(),
                ],
            })
            fig = px.bar(
                conv, x="Group", y="Conversion rate", text=conv["Conversion rate"].map(lambda x: f"{x:.2%}"),
                color="Group", color_discrete_sequence=["#5d9cec", "#36d9b5"],
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", showlegend=False, yaxis_tickformat=".1%",
                yaxis_title="Share of customers who purchased",
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("What happened to average spend?")
            spend = pd.DataFrame({
                "Group": ["No email", "Email campaign"],
                "Average spend": [
                    df.loc[df.treatment == 0, "spend"].mean(),
                    df.loc[df.treatment == 1, "spend"].mean(),
                ],
            })
            fig = px.bar(
                spend, x="Group", y="Average spend", text=spend["Average spend"].map(money),
                color="Group", color_discrete_sequence=["#5d9cec", "#36d9b5"],
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                yaxis_title="Average spend per customer ($)",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### What should management do?")
        st.markdown(
            "1. **Keep email as a campaign channel worth testing.** The randomized experiment found a positive average effect.\n"
            "2. **Do not roll out model-based targeting as the winner yet.** Its holdout result has not beaten the simpler email-everyone strategy.\n"
            "3. **Run a fresh randomized test** before changing the targeting policy in a live business."
        )
        with st.expander("How confident should we be? — statistical details"):
            st.write(
                f"Conversion: email-minus-control difference {ate_conv['ate'] * 100:+.3f} percentage points; "
                f"95% confidence interval [{ate_conv['ci_lower'] * 100:.3f}, {ate_conv['ci_upper'] * 100:.3f}] pp; "
                f"p-value {ate_conv['p_value']:.6g}."
            )
            st.write(
                f"Spend: email-minus-control difference {money(ate_spend['ate'])} per customer; "
                f"95% confidence interval [{money(ate_spend['ci_lower'])}, {money(ate_spend['ci_upper'])}]; "
                f"p-value {ate_spend['p_value']:.6g}."
            )
            st.caption("These are average effects of the pooled email treatment versus no email. The two email creatives are combined in this primary comparison.")

        st.markdown("### Does personalization improve the result?")
        if policy_values and all_email and no_email and comparison:
            p1, p2, p3 = st.columns(3)
            p1.metric("Personalized model policy", money(policy_values["value"]))
            p2.metric("Email everyone", money(all_email["value"]))
            p3.metric("Email nobody", money(no_email["value"]))
            st.caption("Estimated average spend per customer on the independent randomized holdout.")
            st.warning(
                f"The model policy is {money(comparison['difference'])} per customer versus emailing everyone "
                f"(95% confidence interval: {money(comparison['ci_lower'])} to {money(comparison['ci_upper'])}). "
                "The evidence does not establish that personalization improves on emailing everyone."
            )
        else:
            st.info("Policy evaluation metrics are not available yet. Review the Data Science view after model training.")

        st.caption(
            "Important: this is a historical 2008 experiment, not a forecast of current campaign performance. "
            "The results describe this dataset and should be re-tested with current customers."
        )

    elif page == "Customer insights":
        header(
            "CUSTOMER UNDERSTANDING",
            "What customer patterns can we see?",
            "Explore customer history and model estimates. These scores are hypotheses for testing, not proven individual effects.",
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Average historical spend", money(df.history.mean()))
        c2.metric("New customers", f"{df.newbie.mean():.1%}")
        c3.metric("Multichannel customers", f"{(df.channel == 'Multichannel').mean():.1%}")

        st.subheader("Estimated response by customer history")
        chart_data = scored.copy()
        chart_data["History group"] = pd.cut(
            chart_data["history"], bins=5, duplicates="drop"
        ).astype(str)
        grouped = chart_data.groupby("History group", observed=False).agg(
            customers=("estimated_incremental_spend", "size"),
            estimated_incremental_spend=("estimated_incremental_spend", "mean"),
        ).reset_index()
        fig = px.bar(
            grouped, x="History group", y="estimated_incremental_spend",
            hover_data=["customers"], color_discrete_sequence=["#36d9b5"],
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Historical customer spend group",
            yaxis_title="Average model-estimated incremental spend ($)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "Interpretation: this chart shows patterns learned by the model. It does not prove that a particular "
            "customer will spend more because of an email. The model's individual spend prediction is weak on holdout data."
        )
        with st.expander("Explore customer-level estimates"):
            sample = scored[
                ["recency", "history", "channel", "zip_code", "predicted_spend_control",
                 "predicted_spend_treatment", "estimated_incremental_spend"]
            ].sort_values("estimated_incremental_spend", ascending=False).head(100).copy()
            sample = sample.rename(columns={
                "recency": "Months since last purchase",
                "history": "Historical spend",
                "channel": "Customer channel",
                "zip_code": "ZIP category",
                "predicted_spend_control": "Estimated spend without email",
                "predicted_spend_treatment": "Estimated spend with email",
                "estimated_incremental_spend": "Model-estimated difference",
            })
            st.dataframe(sample, use_container_width=True, hide_index=True)
            st.download_button(
                "Download sample customer estimates",
                sample.to_csv(index=False),
                "customer_estimates_exploratory.csv",
                "text/csv",
            )

    elif page == "Campaign planner":
        header(
            "CAMPAIGN PLANNING",
            "Explore a campaign budget",
            "Change the assumptions to see how a hypothetical contact plan changes. This is a scenario, not a profit forecast.",
        )
        st.warning(
            "The dataset does not contain actual email delivery costs or product contribution margins. "
            "You must supply these assumptions. The current targeting model has not proven better than emailing everyone."
        )
        col1, col2, col3 = st.columns(3)
        budget = col1.number_input("Available campaign budget ($)", min_value=0.0, value=1000.0, step=100.0)
        cost = col2.number_input("Assumed cost per email ($)", min_value=0.0, value=0.05, step=0.01)
        margin = col3.slider(
            "Assumed contribution margin",
            0.0, 1.0, 0.4, 0.05,
            format="%.0f%%",
            help="Share of incremental revenue retained after variable product costs. This is a business input, not measured by the experiment.",
        )
        result = optimize_budget(scored, budget, cost, margin)
        a, b, c = st.columns(3)
        a.metric("Customers the scenario would contact", f"{result['customers_targeted']:,}")
        b.metric("Estimated contact cost", money(result["marketing_cost"]))
        c.metric("Modelled net contribution", money(result["expected_net_contribution"]))
        st.caption(f"Unused budget under these assumptions: {money(result['unused_budget'])}")
        st.subheader("What this scenario means")
        st.write(
            "The calculation applies the assumed contribution margin to model-estimated incremental revenue, "
            "then subtracts the assumed email cost. It is useful for sensitivity analysis, but should not be "
            "treated as realized profit or a validated customer-level recommendation."
        )
        with st.expander("View the proposed allocation"):
            recommendations = result["recommendations"].copy()
            display_cols = [
                "recency", "history", "channel", "estimated_incremental_spend",
                "expected_net_contribution", "recommended_action",
            ]
            top = recommendations[display_cols].head(250).rename(columns={
                "recency": "Months since last purchase",
                "history": "Historical spend",
                "channel": "Customer channel",
                "estimated_incremental_spend": "Estimated incremental spend",
                "expected_net_contribution": "Modelled net contribution",
                "recommended_action": "Scenario action",
            })
            st.dataframe(top, use_container_width=True, hide_index=True)
            st.download_button(
                "Download full scenario allocation",
                recommendations.to_csv(index=False),
                "campaign_scenario_allocation.csv",
                "text/csv",
            )

    else:
        from causal_ai.report_generator import build_report_context, generate_executive_report
        import os

        header("GENERATIVE AI / BUSINESS INTELLIGENCE", "AI Business Reports", "Turn verified experiment evidence into a stakeholder-ready report.")
        st.write("Python remains the source of truth for all statistical calculations. Groq writes the narrative only.")
        st.info("Only aggregate experiment metrics and model diagnostics are sent to Groq. Raw customer rows and customer-level scores are not included.")
        try:
            groq_api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            groq_api_key = ""
        groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        try:
            groq_model = st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        except Exception:
            groq_model = "llama-3.3-70b-versatile"
        groq_model = os.getenv("GROQ_MODEL", groq_model)
        with st.expander("What goes into the report?"):
            st.markdown("- Randomized experiment size and average treatment effects\\n- Confidence intervals and p-values\\n- Holdout policy values\\n- Model diagnostics and limitations\\n- Business recommendation and next steps")
        if not groq_api_key:
            st.warning("Add GROQ_API_KEY in Streamlit Community Cloud → App settings → Secrets to enable report generation.")
            st.code('GROQ_API_KEY = "your_groq_api_key"\\nGROQ_MODEL = "llama-3.3-70b-versatile"', language="toml")
        if st.button("Generate Executive Report", type="primary", disabled=not bool(groq_api_key)):
            context = build_report_context(df, ate_conv, ate_spend, metrics)
            with st.spinner("Generating the executive report with Groq..."):
                try:
                    report = generate_executive_report(context, groq_api_key, groq_model)
                    st.session_state["causal_ai_executive_report"] = report
                    st.session_state["causal_ai_report_context"] = context
                except Exception as exc:
                    st.error(f"Report generation failed: {exc}")
                    st.caption("Check the API key, model availability, and network connection.")
        report = st.session_state.get("causal_ai_executive_report")
        if report:
            st.success("Executive report generated.")
            st.markdown(report)
            st.download_button("Download Executive Report (.md)", report, "causal_ai_executive_report.md", "text/markdown")
            with st.expander("Inspect aggregate evidence sent to Groq"):
                st.json(st.session_state.get("causal_ai_report_context", {}))
        st.caption("AI-generated narrative for decision support. Verify the statistical evidence before making business decisions.")

else:
    page = st.sidebar.radio(
        "DATA SCIENCE WORKSPACE",
        [
            "Dataset overview", "Data quality", "Exploratory data analysis",
            "Feature analysis", "Correlation analysis", "Statistical analysis",
            "Experiment evidence", "Uplift model diagnostics", "Budget model assumptions",
        ],
    )

    if page == "Dataset overview":
        header("DATA SCIENCE / DATA UNDERSTANDING", "Dataset overview", "Inspect the experiment dataset, schema, sample composition, and analytical roles.")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
        a,b,c,d,e = st.columns(5)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", str(df.shape[1]))
        c.metric("Numeric", str(len(numeric_cols)))
        d.metric("Categorical", str(len(categorical_cols)))
        e.metric("Exact duplicates", f"{df.duplicated().sum():,}")
        st.caption("Hillstrom / MineThatData historical randomized email experiment (2008). No unique customer identifier is supplied.")
        st.subheader("Dataset preview")
        st.dataframe(df.head(30), use_container_width=True, hide_index=True)
        st.download_button("Download complete analysis dataset", df.to_csv(index=False), "hillstrom_analysis_dataset.csv", "text/csv")
        definitions = {
            "recency":"Months since last purchase — pre-treatment numeric feature",
            "history_segment":"Binned historical spend — pre-treatment categorical feature",
            "history":"Historical customer spend — pre-treatment numeric feature",
            "mens":"Prior men's merchandise purchase indicator — pre-treatment binary feature",
            "womens":"Prior women's merchandise purchase indicator — pre-treatment binary feature",
            "zip_code":"Geographic ZIP category — pre-treatment categorical feature",
            "newbie":"New customer indicator — pre-treatment binary feature",
            "channel":"Historical acquisition channel — pre-treatment categorical feature",
            "segment":"Original randomized arm: Mens E-Mail, Womens E-Mail, No E-Mail",
            "treatment":"Derived pooled email assignment (1) versus no email (0)",
            "treatment_label":"Readable pooled treatment label",
            "visit":"Whether the customer visited — outcome",
            "conversion":"Whether the customer purchased — primary binary outcome",
            "spend":"Observed spend in USD — primary continuous outcome",
        }
        schema = pd.DataFrame([{"Column":c,"Dtype":str(df[c].dtype),"Meaning / role":definitions.get(c,"Derived analytical column"),"Unique":int(df[c].nunique(dropna=False)),"Missing":int(df[c].isna().sum())} for c in df.columns])
        st.subheader("Data dictionary")
        st.dataframe(schema, use_container_width=True, hide_index=True)
        arm_counts = df.segment.value_counts().rename_axis("Randomized arm").reset_index(name="Records")
        st.plotly_chart(px.bar(arm_counts,x="Randomized arm",y="Records",color="Randomized arm",text="Records"),use_container_width=True)

    elif page == "Data quality":
        header("DATA SCIENCE / VALIDATION", "Data quality audit", "Missingness, duplicates, ranges, binary checks, and outlier screening.")
        quality = pd.DataFrame([{"Column":c,"Dtype":str(df[c].dtype),"Missing":int(df[c].isna().sum()),"Missing %":df[c].isna().mean()*100,"Unique":int(df[c].nunique(dropna=False))} for c in df.columns])
        a,b,c = st.columns(3)
        a.metric("Missing cells", f"{validation['missing_cells']:,}")
        b.metric("Duplicate rows", f"{validation['duplicate_rows']:,}")
        c.metric("Schema checks", "Passed" if validation["valid"] else "Failed")
        st.dataframe(quality.sort_values("Missing %",ascending=False),use_container_width=True,hide_index=True)
        for issue in validation["issues"]: st.error(issue)
        st.subheader("Numeric range and IQR screening")
        records=[]
        for col in df.select_dtypes(include="number"):
            x=df[col].dropna(); q1=x.quantile(.25); q3=x.quantile(.75); iqr=q3-q1
            records.append({"Column":col,"Min":x.min(),"Q1":q1,"Median":x.median(),"Q3":q3,"Max":x.max(),"IQR flagged":int(((x<q1-1.5*iqr)|(x>q3+1.5*iqr)).sum())})
        st.dataframe(pd.DataFrame(records).round(4),use_container_width=True,hide_index=True)
        st.warning("IQR flags are investigation candidates, not proof of erroneous records. Spend and history can be strongly right-skewed.")
        st.write(f"{validation['duplicate_rows']:,} exact duplicate rows are retained transparently. Without a customer ID, identical records cannot safely be assumed to be accidental duplicates.")

    elif page == "Exploratory data analysis":
        header("DATA SCIENCE / EDA", "Exploratory data analysis", "Univariate, categorical, outcome, and experiment-arm exploration.")
        tab1,tab2,tab3,tab4=st.tabs(["Numeric distributions","Categorical distributions","Outcomes","Arm comparison"])
        with tab1:
            cols=df.select_dtypes(include="number").columns.tolist()
            var=st.selectbox("Numeric variable",cols,key="eda_num")
            left,right=st.columns(2)
            left.plotly_chart(px.histogram(df,x=var,nbins=50,marginal="box",title=f"Distribution: {var}"),use_container_width=True)
            right.dataframe(df[var].describe().to_frame("Statistics").round(5),use_container_width=True)
            st.write({"skewness":float(df[var].skew()),"kurtosis":float(df[var].kurtosis())})
        with tab2:
            cols=[c for c in df.select_dtypes(exclude="number").columns if c!="treatment_label"]
            var=st.selectbox("Categorical variable",cols,key="eda_cat")
            counts=df[var].value_counts(dropna=False).rename_axis(var).reset_index(name="Count")
            counts["Share"]=counts.Count/len(df)
            st.plotly_chart(px.bar(counts,x=var,y="Count",color=var,title=f"Frequency: {var}"),use_container_width=True)
            st.dataframe(counts,use_container_width=True,hide_index=True)
        with tab3:
            st.dataframe(df[["visit","conversion","spend"]].describe().round(5),use_container_width=True)
            st.plotly_chart(px.histogram(df,x="spend",color="treatment_label",nbins=60,barmode="overlay",opacity=.6,title="Spend distribution by treatment"),use_container_width=True)
            st.caption("Spend is zero-inflated; interpret purchase incidence separately from spend conditional on purchase.")
        with tab4:
            outcome=st.selectbox("Outcome",["visit","conversion","spend"],key="eda_arm_outcome")
            summary=df.groupby("segment")[outcome].agg(["count","mean","std","median"]).reset_index()
            st.dataframe(summary.round(5),use_container_width=True,hide_index=True)
            st.plotly_chart(px.bar(summary,x="segment",y="mean",color="segment",title=f"Mean {outcome} by randomized arm"),use_container_width=True)

    elif page == "Feature analysis":
        from causal_ai.data import FEATURES
        header("DATA SCIENCE / FEATURE ENGINEERING", "Feature analysis", "Understand model predictors, feature types, leakage boundaries, and observed outcome patterns.")
        st.warning("Model inputs are pre-treatment covariates only. Treatment assignment and post-treatment outcomes are excluded to prevent leakage.")
        info=pd.DataFrame([{"Feature":c,"Dtype":str(df[c].dtype),"Unique":int(df[c].nunique()),"Missing":int(df[c].isna().sum()),"Used by model":c in FEATURES,"Role":("Pre-treatment predictor" if c in FEATURES else "Treatment assignment" if c in ["segment","treatment","treatment_label"] else "Post-treatment outcome" if c in ["visit","conversion","spend"] else "Not currently modelled")} for c in df.columns])
        st.dataframe(info,use_container_width=True,hide_index=True)
        feature=st.selectbox("Inspect model feature",FEATURES)
        if pd.api.types.is_numeric_dtype(df[feature]):
            fig=px.histogram(df,x=feature,color="treatment_label",marginal="box",barmode="overlay",opacity=.6)
            bins=pd.qcut(df[feature],q=min(10,max(2,df[feature].nunique())),duplicates="drop")
            grouped=df.assign(feature_bin=bins.astype(str)).groupby("feature_bin",observed=False).agg(records=("conversion","size"),conversion_rate=("conversion","mean"),visit_rate=("visit","mean"),mean_spend=("spend","mean")).reset_index()
            x_name="feature_bin"
        else:
            fig=px.bar(df.groupby([feature,"treatment_label"],observed=False).size().reset_index(name="Records"),x=feature,y="Records",color="treatment_label",barmode="group")
            grouped=df.groupby(feature,observed=False).agg(records=("conversion","size"),conversion_rate=("conversion","mean"),visit_rate=("visit","mean"),mean_spend=("spend","mean")).reset_index()
            x_name=feature
        st.plotly_chart(fig,use_container_width=True)
        st.subheader("Observed outcomes across feature values")
        st.dataframe(grouped.round(5),use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(grouped,x=x_name,y="conversion_rate",title=f"Conversion rate by {feature}"),use_container_width=True)
        st.caption("These are descriptive patterns, not causal effects of changing a feature.")

    elif page == "Correlation analysis":
        header("DATA SCIENCE / DEPENDENCE", "Correlation analysis", "Compare Pearson, Spearman, and Kendall associations. Correlation is not causation.")
        numeric=df.select_dtypes(include="number")
        method=st.selectbox("Method",["pearson","spearman","kendall"])
        corr=numeric.corr(method=method)
        fig=px.imshow(corr,text_auto=".2f",aspect="auto",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,title=f"{method.title()} correlation matrix")
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(corr.round(4),use_container_width=True)
        x,y=st.columns(2)
        xvar=x.selectbox("X variable",numeric.columns,key="corr_x")
        yvar=y.selectbox("Y variable",numeric.columns,index=min(1,len(numeric.columns)-1),key="corr_y")
        st.metric(f"{method.title()} correlation",f"{numeric[xvar].corr(numeric[yvar],method=method):.4f}")
        st.plotly_chart(px.scatter(df,x=xvar,y=yvar,color="treatment_label",opacity=.4),use_container_width=True)
        st.caption("Binary indicators and zero-inflated spend can distort simple linear interpretations; compare rank correlation and inspect scatter plots.")

    elif page == "Statistical analysis":
        header("DATA SCIENCE / INFERENCE", "Statistical analysis workbench", "Descriptive moments, group summaries, hypothesis tests, p-values, and confidence intervals.")
        variable=st.selectbox("Variable",["recency","history","visit","conversion","spend"])
        desc=df[variable].describe().to_frame("Overall").round(5)
        desc.loc["skewness"]=df[variable].skew()
        desc.loc["kurtosis"]=df[variable].kurtosis()
        st.subheader("Descriptive statistics")
        st.dataframe(desc,use_container_width=True)
        st.subheader("By randomized treatment")
        st.dataframe(df.groupby("treatment_label")[variable].agg(["count","mean","std","median","min","max"]).round(5),use_container_width=True)
        if variable in ["visit","conversion"]:
            result=two_proportion_test(df[variable],df.treatment)
            st.write(f"Two-proportion z-test · difference {result['ate']:.5f} · z={result['z_statistic']:.4f} · p={result['p_value']:.6g} · 95% CI [{result['ci_lower']:.5f}, {result['ci_upper']:.5f}]")
        else:
            result=mean_difference(df[variable],df.treatment)
            st.write(f"Welch t-test · difference {result['ate']:.5f} · df={result['degrees_of_freedom']:.2f} · p={result['p_value']:.6g} · 95% CI [{result['ci_lower']:.5f}, {result['ci_upper']:.5f}]")
        st.caption("Primary comparison pools both email arms versus no email. Subgroup patterns are exploratory unless multiplicity and interactions are handled.")

    elif page == "Experiment evidence":
        header(
            "DATA SCIENCE / CAUSAL INFERENCE",
            "Experiment evidence",
            "Inspect the randomized treatment effect, uncertainty, and statistical assumptions.",
        )
        outcome = st.selectbox("Outcome to analyze", ["conversion", "visit", "spend"])
        if outcome in ["conversion", "visit"]:
            result = two_proportion_test(df[outcome], df.treatment)
            fmt = lambda value: f"{value * 100:.3f}%"
            unit = "percentage points"
            treated = result["treated_rate"]
            control = result["control_rate"]
        else:
            result = mean_difference(df[outcome], df.treatment)
            fmt = money
            unit = "USD per customer"
            treated = result["treated_mean"]
            control = result["control_mean"]

        a, b, c, d = st.columns(4)
        a.metric("Email group", fmt(treated))
        b.metric("No-email group", fmt(control))
        c.metric("Estimated average effect", fmt(result["ate"]))
        d.metric("P-value", f"{result['p_value']:.5g}")
        st.subheader("95% confidence interval")
        st.write(f"[{fmt(result['ci_lower'])}, {fmt(result['ci_upper'])}]")
        st.write(
            "Statistical conclusion:",
            "**Statistically significant at 5%**"
            if result["p_value"] < 0.05 else "**Not statistically significant at 5%**",
        )
        arms = df.groupby("segment")[outcome].mean().reset_index()
        fig = px.bar(
            arms, x="segment", y=outcome, color="segment",
            color_discrete_sequence=["#36d9b5", "#5d9cec", "#a78bfa"],
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Effect is email treatment minus no-email control. Units: {unit}. "
            "Spend uses Welch's independent-sample t-test; binary outcomes use a two-proportion z-test."
        )
        st.warning(
            "The pooled treatment combines Mens E-Mail and Womens E-Mail. The estimate is the average effect "
            "of this mixture versus no email; creative-specific effects are not the primary estimand."
        )

    elif page == "Uplift model diagnostics":
        header(
            "DATA SCIENCE / HETEROGENEOUS TREATMENT EFFECTS",
            "Uplift model diagnostics",
            "Inspect model-implied customer differences and out-of-sample policy evaluation.",
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean model-estimated uplift", money(scored.estimated_incremental_spend.mean()))
        c2.metric("Share with positive model score", f"{(scored.estimated_incremental_spend > 0).mean():.1%}")
        diff = comparison.get("difference")
        c3.metric("Model policy minus email all", money(diff) if diff is not None else "N/A")

        if policy_values and all_email and no_email:
            st.subheader("Randomized holdout policy values")
            p1, p2, p3 = st.columns(3)
            p1.metric("Model-based policy", money(policy_values["value"]))
            p2.metric("Email everyone", money(all_email["value"]))
            p3.metric("Email nobody", money(no_email["value"]))
            st.caption("Inverse-propensity-weighted estimated spend per customer on the independent randomized holdout.")
            if comparison:
                st.warning(
                    f"Model policy minus email-everyone: {money(comparison['difference'])} per customer "
                    f"(95% CI {money(comparison['ci_lower'])} to {money(comparison['ci_upper'])}). "
                    "The model policy has not demonstrated superiority to emailing everyone."
                )

        left, right = st.columns(2)
        with left:
            fig = px.histogram(
                scored, x="estimated_incremental_spend", nbins=60,
                color_discrete_sequence=["#36d9b5"],
                labels={"estimated_incremental_spend": "Model-estimated incremental spend ($)"},
            )
            fig.add_vline(x=0, line_dash="dash", line_color="#ff9d6c")
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            if metrics.get("qini_fraction"):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=metrics["qini_fraction"], y=metrics["qini_gain"],
                    mode="lines", name="Model ranking",
                ))
                if metrics.get("qini_random_gain"):
                    fig.add_trace(go.Scatter(
                        x=metrics["qini_fraction"], y=metrics["qini_random_gain"],
                        mode="lines", name="Random ranking",
                        line=dict(dash="dash"),
                    ))
                fig.update_layout(
                    title="Holdout cumulative uplift gain",
                    xaxis_title="Share of holdout customers ranked",
                    yaxis_title="Cumulative IPW gain",
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Model specification and limitations"):
            st.markdown(
                "The two-part T-learner fits separate treatment and control outcome models. "
                "Logistic regression estimates conversion probability; gradient-boosting regression "
                "estimates positive spend conditional on conversion. Expected spend is the product "
                "of those two estimates. Uplift is the difference between the two model-implied expected spends."
            )
            st.write(
                "Only pre-treatment covariates are used. The data is split into training and holdout sets; "
                "the targeting threshold is selected using training data and policy values are evaluated "
                "on the randomized holdout using inverse-propensity weighting."
            )
            st.warning(
                "Holdout spend R-squared is close to zero. Individual uplift rankings and Qini/AUUC are "
                "exploratory diagnostics, not validated individual causal effects or profit forecasts."
            )

    else:
        header(
            "DATA SCIENCE / DECISION MODEL",
            "Budget model assumptions",
            "Review the arithmetic and assumptions behind the campaign scenario.",
        )
        st.code(
            "Modelled net contribution =\n"
            "    estimated incremental revenue × assumed contribution margin\n"
            "    − assumed email contact cost",
            language="text",
        )
        st.markdown(
            "- Contact cost is supplied by the dashboard user.\n"
            "- Contribution margin is supplied by the dashboard user.\n"
            "- Neither input is observed in the Hillstrom dataset.\n"
            "- The scenario inherits the uncertainty and limitations of the uplift model."
        )
        st.warning(
            "This is a sensitivity-analysis tool, not realized profit, a financial forecast, "
            "or a validated production targeting policy."
        )

st.sidebar.divider()
st.sidebar.caption("RESEARCH DATA · HILLSTROM EMAIL EXPERIMENT")
st.sidebar.caption("MODEL · TWO-PART T-LEARNER")
st.sidebar.caption("Dashboard presentation · Executive + Data Science")
