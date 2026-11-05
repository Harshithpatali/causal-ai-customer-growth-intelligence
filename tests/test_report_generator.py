import pandas as pd
from causal_ai.report_generator import build_report_context, build_report_messages

def test_report_context_is_aggregate_only():
    df = pd.DataFrame({"segment": ["Mens E-Mail", "No E-Mail"], "email": ["private-a", "private-b"]})
    context = build_report_context(df, {"ate": 0.01, "p_value": 0.03, "n_treated": 1, "n_control": 1}, {"ate": 2.0}, {"policy_value_ipw": {"value": 1.1}, "all_email_value_ipw": {"value": 1.3}, "all_control_value_ipw": {"value": 0.6}, "policy_vs_all_email_ipw": {"difference": -0.2}, "test_spend_r2_control": 0.002})
    assert context["study"]["observations"] == 2
    assert "private-a" not in str(context)
    assert context["average_treatment_effects"]["conversion"]["ate"] == 0.01
    assert context["holdout_policy_evaluation"]["model_policy_minus_email_everyone"]["difference"] == -0.2

def test_prompt_has_guardrails_and_embeds_evidence():
    messages = build_report_messages({"study": {"observations": 10}})
    assert "Do not invent data" in messages[0]["content"]
    assert "realized profit" in messages[0]["content"]
    assert "VERIFIED_AGGREGATE_EVIDENCE" in messages[1]["content"]
    assert "10" in messages[1]["content"]
