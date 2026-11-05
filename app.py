from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parent/"src"))
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from causal_ai.data import load_data, validate_data, FEATURES
from causal_ai.modeling import load_models, score_customers
from causal_ai.statistics import two_proportion_test, mean_difference
from causal_ai.optimization import optimize_budget
from causal_ai.pipeline import run_pipeline

st.set_page_config(page_title="Causal AI | Growth Intelligence", page_icon="◈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background: radial-gradient(circle at 15% 0%, #13243b 0, #0a1220 36%, #080e18 100%); color:#e8eef8;}
section[data-testid="stSidebar"] {background:#0c1727; border-right:1px solid #20324a;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif; color:#f3f7ff;}
div[data-testid="stMetric"] {background:linear-gradient(145deg,#14243a,#101b2d); border:1px solid #263b56; padding:18px; border-radius:14px;}
div[data-testid="stMetricLabel"] {color:#9eb2ca;}
div[data-testid="stMetricValue"] {color:#f3f7ff;}
.block-container {padding-top:2rem; max-width:1500px;}
div.stButton>button {background:#39d6b4;color:#07131d;border:0;border-radius:9px;font-weight:700;}
.small-muted {color:#91a5bd;font-size:0.88rem;}
.hero {padding:26px 30px;border-radius:18px;background:linear-gradient(115deg,#142b43,#102033 65%,#153a43);border:1px solid #294b63;margin-bottom:22px;}
.hero-kicker {color:#51dfc0;letter-spacing:2px;font-size:11px;font-weight:700;}
.hero-title {font-family:'Space Grotesk';font-size:34px;font-weight:700;color:#f5f8ff;margin:6px 0;}
.hero-sub {color:#b4c5d8;font-size:14px;}
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
    st.markdown(f'<div class="hero"><div class="hero-kicker">{kicker}</div><div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div></div>', unsafe_allow_html=True)

def money(x):
    return f"${x:,.2f}"

st.sidebar.markdown("## ◈ CAUSAL AI")
st.sidebar.caption("CUSTOMER GROWTH INTELLIGENCE")
page = st.sidebar.radio("NAVIGATION", [
    "01  Executive Overview",
    "02  Customer Intelligence",
    "03  Experimentation Lab",
    "04  Uplift Intelligence",
    "05  Profit Optimizer"
])
st.sidebar.divider()
st.sidebar.caption("DATASET · HILLSTROM EMAIL EXPERIMENT")
st.sidebar.caption("MODEL · TWO-PART T-LEARNER / HURDLE SPEND")
st.sidebar.caption("VERSION · 1.1.0")

try:
    df = get_data()
except Exception as exc:
    st.error(f"Dataset unavailable: {exc}")
    st.info("Place the Hillstrom CSV at data/raw/hillstrom.csv or allow the app to download it on first launch.")
    st.stop()

validation = validate_data(df)
if not validation["valid"]:
    st.error(validation["issues"])
    st.stop()

try:
    models = get_model()
except Exception:
    st.warning("Model artifacts are not available yet.")
    if st.button("Train causal models now"):
        with st.spinner("Training two outcome models. This may take a few minutes..."):
            run_pipeline()
            st.cache_resource.clear()
        st.rerun()
    st.stop()

scored = score_customers(df, models)
metrics_path = Path("artifacts/metrics.json")
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

if page.startswith("01"):
    header("DECISION INTELLIGENCE / OVERVIEW", "Customer Growth Control Tower",
           "Measure what marketing changes — then allocate spend where incremental value is highest.")
    ate_conv = two_proportion_test(df.conversion, df.treatment)
    ate_spend = mean_difference(df.spend, df.treatment)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Conversion ATE", f"{ate_conv['ate']*100:+.3f} pp",
              f"p = {ate_conv['p_value']:.4f}")
    c3.metric("Spend ATE / customer", money(ate_spend["ate"]))
    c4.metric("Positive uplift share", f"{(scored.estimated_incremental_spend>0).mean()*100:.1f}%")
    left,right = st.columns([1.35,1])
    with left:
        st.subheader("Campaign arm performance")
        arms = df.groupby("segment").agg(customers=("segment","size"),
            conversion_rate=("conversion","mean"), visit_rate=("visit","mean"),
            avg_spend=("spend","mean")).reset_index()
        fig = px.bar(arms, x="segment", y="conversion_rate", color="segment",
                     text=arms.conversion_rate.map(lambda x:f"{x:.2%}"),
                     color_discrete_sequence=["#36d9b5","#5d9cec","#a78bfa"])
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",showlegend=False,
                          yaxis_tickformat=".1%")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader("Treatment vs control")
        comp = pd.DataFrame({"Group":["Control","Email treatment"],
          "Conversion":[df.loc[df.treatment==0,"conversion"].mean(),
                        df.loc[df.treatment==1,"conversion"].mean()],
          "Spend":[df.loc[df.treatment==0,"spend"].mean(),
                   df.loc[df.treatment==1,"spend"].mean()]})
        st.dataframe(comp.style.format({"Conversion":"{:.3%}","Spend":"${:.2f}"}),
                     use_container_width=True,hide_index=True)
        st.info("Experiment-level differences are randomized estimates. Customer-level uplift is model-estimated.")
    st.subheader("What the model estimates")
    st.markdown("The T-learner estimates expected spend under both email and no-email scenarios for each customer. Their difference is the estimated incremental spend, not an observed individual causal effect.")
    st.caption("Source: Kevin Hillstrom / MineThatData randomized email experiment. Outcomes were observed after campaign delivery.")

elif page.startswith("02"):
    header("CUSTOMER INTELLIGENCE / SEGMENTATION", "Understand the customer",
           "Explore pre-treatment behaviour and inspect model-estimated outcomes under both actions.")
    a,b,c = st.columns(3)
    a.metric("Average historical spend", money(df.history.mean()))
    b.metric("New customers", f"{df.newbie.mean():.1%}")
    c.metric("Multichannel customers", f"{(df.channel=='Multichannel').mean():.1%}")
    fig = px.scatter(scored.sample(min(3500,len(scored)),random_state=42),
        x="history",y="estimated_incremental_spend",color="channel",
        size="recency",hover_data=["zip_code","newbie","segment"],
        color_discrete_sequence=["#36d9b5","#5d9cec","#a78bfa"])
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    st.subheader("Customer-level scoring")
    st.caption("These rows are scored using pre-treatment attributes only.")
    sample = scored[["recency","history","channel","zip_code","predicted_spend_control",
        "predicted_spend_treatment","estimated_incremental_spend"]].copy()
    sample = sample.sort_values("estimated_incremental_spend",ascending=False).head(100)
    st.dataframe(sample,use_container_width=True,hide_index=True)
    st.download_button("Download top 100 customer scores",sample.to_csv(index=False),
                       "top_customer_uplift.csv","text/csv")

elif page.startswith("03"):
    header("EXPERIMENTATION / CAUSAL VALIDATION", "Experimentation Lab",
           "Quantify average campaign impact with confidence intervals and transparent statistical tests.")
    outcome = st.selectbox("Primary outcome",["conversion","visit","spend"])
    if outcome in ["conversion","visit"]:
        result = two_proportion_test(df[outcome],df.treatment)
        unit = "percentage points"
        fmt = lambda x:f"{x*100:.3f}%"
    else:
        result = mean_difference(df[outcome],df.treatment)
        unit = "USD per customer"
        fmt = money
    x1,x2,x3,x4 = st.columns(4)
    x1.metric("Treatment mean",fmt(result["treated_rate"] if outcome!="spend" else result["treated_mean"]))
    x2.metric("Control mean",fmt(result["control_rate"] if outcome!="spend" else result["control_mean"]))
    x3.metric("Estimated ATE",fmt(result["ate"]))
    x4.metric("P-value",f"{result['p_value']:.5f}")
    st.subheader("95% confidence interval")
    st.write(f"[{fmt(result['ci_lower'])}, {fmt(result['ci_upper'])}]")
    st.write("Statistical conclusion:", "**Statistically significant at 5%**" if result["p_value"]<0.05 else "**Not statistically significant at 5%**")
    st.caption(f"Welch-style mean comparison for spend; two-proportion z-test for binary outcomes. Effect is treatment minus control. Units: {unit}.")
    arms = df.groupby("segment")[outcome].mean().reset_index()
    fig=px.bar(arms,x="segment",y=outcome,color="segment",
               color_discrete_sequence=["#36d9b5","#5d9cec","#a78bfa"])
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    st.warning("The combined treatment pools two different email creatives. Interpret the estimate as the average effect of this mixture of email treatments versus no email.")

elif page.startswith("04"):
    header("UPLIFT INTELLIGENCE / MODEL INSIGHTS", "Find customers whose behaviour may change",
           "Estimate the difference between predicted spend with email and predicted spend without email.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Mean predicted uplift",money(scored.estimated_incremental_spend.mean()))
    c2.metric("Positive uplift",f"{(scored.estimated_incremental_spend>0).mean():.1%}")
    c3.metric("Holdout Qini-style score",f"{metrics.get('qini',0):,.1f}")
    if metrics.get("policy_value_ipw"):
        st.subheader("Randomized holdout policy benchmark")
        p1,p2,p3=st.columns(3)
        p1.metric("Model policy IPW value",money(metrics["policy_value_ipw"]["value"]))
        p2.metric("Email everyone",money(metrics["all_email_value_ipw"]["value"]))
        p3.metric("Email nobody",money(metrics["all_control_value_ipw"]["value"]))
        st.caption("Inverse-propensity-weighted estimated spend per customer on the randomized holdout. The model policy must beat simple baselines before it can be considered useful.")
    fig=px.histogram(scored,x="estimated_incremental_spend",nbins=60,
                     color_discrete_sequence=["#36d9b5"])
    fig.add_vline(x=0,line_dash="dash",line_color="#ff9d6c")
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    if metrics.get("qini_fraction"):
        qfig=go.Figure()
        qfig.add_trace(go.Scatter(x=metrics["qini_fraction"],y=metrics["qini_gain"],
                                  mode="lines",name="Uplift ranking",line=dict(color="#36d9b5",width=3)))
        qfig.update_layout(title="Holdout cumulative uplift ranking diagnostic",
                           xaxis_title="Fraction of ranked test customers",
                           yaxis_title="Cumulative incremental outcome proxy",
                           template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(qfig,use_container_width=True)
    st.markdown("**Model:** Two-part T-learner. Separate logistic regression models estimate conversion probability by treatment arm; regularized gradient-boosting models estimate spend conditional on conversion. Expected spend is their product. This is a baseline, not proof of individual-level causal effects.")
    st.caption("Spend is sparse and zero-inflated in this dataset; Qini/AUUC rankings are noisy. Validate policy value on randomized holdout data and compare with simple targeting baselines.")

else:
    header("DECISION SCIENCE / RESOURCE ALLOCATION", "Profit Optimization",
           "Allocate a fixed contact budget to customers with the highest estimated incremental net value.")
    col1,col2,col3=st.columns(3)
    budget=col1.number_input("Marketing budget ($)",min_value=0.0,value=1000.0,step=100.0)
    cost=col2.number_input("Email cost per customer ($)",min_value=0.0,value=0.05,step=0.01)
    margin=col3.slider("Contribution margin",0.0,1.0,0.4,0.05, help="Fraction of incremental revenue retained as contribution. For example, 0.40 means 40%. This is a business assumption, not a measured value in the dataset.")
    result=optimize_budget(scored,budget,cost,margin)
    a,b,c,d=st.columns(4)
    a.metric("Customers selected",f"{result['customers_targeted']:,}")
    b.metric("Contact spend",money(result["marketing_cost"]))
    c.metric("Modelled net contribution",money(result["expected_net_contribution"]))
    d.metric("Unused budget",money(result["unused_budget"]))
    st.subheader("Recommended customer allocation")
    top=result["recommendations"][["recency","history","channel","estimated_incremental_spend",
        "expected_net_contribution","recommended_action"]].head(250)
    st.dataframe(top,use_container_width=True,hide_index=True)
    st.download_button("Download full allocation",result["recommendations"].to_csv(index=False),
                       "campaign_allocation.csv","text/csv")
    st.warning("This is an exploratory modelled scenario, not realized profit or a validated customer-level recommendation. Net contribution = estimated incremental revenue × assumed contribution margin − assumed email cost. The dataset does not contain actual email costs or product margins. The current holdout has not shown that this targeting policy outperforms emailing everyone.")
