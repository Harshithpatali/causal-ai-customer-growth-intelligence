"""Aggregate-only Groq narrative reports for the causal experiment."""
import json
import math

def _number(value):
    if value is None: return None
    try: result = float(value)
    except (TypeError, ValueError): return None
    return result if math.isfinite(result) else None

def _block(result, keys):
    out = {key: _number(result.get(key)) for key in keys}
    for key in ("n_treated", "n_control"):
        if key in result: out[key] = int(result[key])
    return out

def build_report_context(df, conversion_result, spend_result, metrics):
    """Create a small summary; customer-level rows and scores are never included."""
    segments = {str(k): int(v) for k, v in df["segment"].value_counts().items()} if "segment" in df else {}
    policy = {}
    for key in ("policy_value_ipw", "all_email_value_ipw", "all_control_value_ipw"):
        source = metrics.get(key) or {}
        if source: policy[key] = {k: _number(source.get(k)) for k in ("value", "ci_lower", "ci_upper", "standard_error")}
    comparison = metrics.get("policy_vs_all_email_ipw") or {}
    policy["model_policy_minus_email_everyone"] = ({k: _number(comparison.get(k)) for k in ("difference", "ci_lower", "ci_upper", "standard_error")} if comparison else None)
    model_keys = ("n_train", "n_test", "test_spend_r2_control", "test_spend_r2_treated", "test_spend_rmse_control", "test_spend_rmse_treated", "test_conversion_auc_control", "test_conversion_auc_treated", "mean_predicted_uplift", "positive_uplift_share", "policy_threshold_expected_incremental_spend", "qini", "auuc")
    model = {k: _number(metrics[k]) for k in model_keys if k in metrics}
    model["model_type"] = metrics.get("model_type", "two_part_t_learner")
    model["interpretation"] = "Holdout spend R-squared is close to zero; individual predictions and uplift rankings are exploratory, not validated individual causal effects."
    return {
        "study": {"dataset": "Hillstrom / MineThatData randomized email experiment (2008)", "observations": int(len(df)), "segment_counts": segments, "treatment_definition": "Pooled email creatives versus no email", "randomized_design": True},
        "average_treatment_effects": {
            "conversion": _block(conversion_result, ("treated_rate", "control_rate", "ate", "ci_lower", "ci_upper", "p_value")),
            "spend_usd_per_customer": _block(spend_result, ("treated_mean", "control_mean", "ate", "ci_lower", "ci_upper", "p_value"))},
        "holdout_policy_evaluation": policy,
        "model_diagnostics": model,
        "budget_scenario": {"status": "Not calculated", "caveat": "Email costs and contribution margins are not in the dataset. Any net contribution scenario depends on user assumptions and is not realized profit."},
        "decision_guardrails": ["Do not claim the model policy beats emailing everyone without supporting holdout evidence.", "Do not describe modelled revenue or net contribution as realized profit.", "Do not infer validated individual treatment effects from uplift scores.", "The experiment is historical; current campaigns require a new randomized test."]
    }

def build_report_messages(context):
    evidence = json.dumps(context, indent=2, sort_keys=True, allow_nan=False)
    system = ("You are a careful senior experimentation scientist. Use only supplied JSON. Do not invent data, numbers, p-values, confidence intervals, costs, margins, or causal claims. Preserve numeric values exactly. Distinguish randomized average effects from model-estimated individual uplift. Explain uncertainty. State that holdout spend R-squared is close to zero and the model policy has not demonstrated superiority to emailing everyone when supported by the evidence. Never call a budget scenario realized profit. Mention the 2008 historical setting and recommend a fresh randomized validation. Write Markdown with headings: Executive Summary, What the Experiment Found, What the Model Can and Cannot Tell Us, Business Recommendation, Risks and Limitations, Next Steps.")
    user = "Create an executive report from this verified aggregate evidence. Do not calculate new metrics or add unsupported numbers.\n\nVERIFIED_AGGREGATE_EVIDENCE:\n" + evidence
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def build_report_chart_data(df, metrics):
    """Return safe aggregate chart data; never expose customer-level records."""
    required = {"treatment", "conversion", "spend"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot build report charts; missing columns: {sorted(missing)}")

    outcomes = []
    labels = {0: "No email (control)", 1: "Email treatment"}
    for treatment, group in df.groupby("treatment", sort=True):
        outcomes.append({
            "group": labels.get(int(treatment), f"Treatment {treatment}"),
            "customers": int(len(group)),
            "conversion_rate": _number(group["conversion"].mean()),
            "mean_spend": _number(group["spend"].mean()),
        })

    policy_specs = (
        ("policy_value_ipw", "Model targeting"),
        ("all_email_value_ipw", "Email everyone"),
        ("all_control_value_ipw", "Email nobody"),
    )
    policy_values = []
    for key, label in policy_specs:
        source = metrics.get(key) or {}
        value = _number(source.get("value"))
        if value is None:
            continue
        policy_values.append({
            "strategy": label,
            "value": value,
            "ci_lower": _number(source.get("ci_lower")),
            "ci_upper": _number(source.get("ci_upper")),
        })
    return {"outcomes_by_group": outcomes, "holdout_policy_values": policy_values}


def generate_executive_report(context, api_key, model="openai/gpt-oss-120b"):
    """Generate only the narrative; all numeric evidence and charts remain Python-owned."""
    if not api_key or not str(api_key).strip():
        raise ValueError("A Groq API key is required.")
    model = str(model or "").strip()
    if not model:
        raise ValueError("A Groq model ID is required.")
    from groq import Groq
    client = Groq(api_key=api_key, timeout=45.0, max_retries=1)
    response = client.chat.completions.create(model=model, messages=build_report_messages(context), temperature=0.2, max_tokens=1800)
    report = response.choices[0].message.content
    if not report or not report.strip(): raise RuntimeError("Groq returned an empty report.")
    return report.strip()