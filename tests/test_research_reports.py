import json

from causal_ai.research_reports import (
    load_json_report,
    validate_advanced_statistics,
    validate_research_summary,
    validate_spend_report,
)


def test_report_loader_handles_missing_file(tmp_path):
    report, error = load_json_report(tmp_path / "missing.json")
    assert report is None
    assert "missing" in error.lower()


def test_report_loader_handles_empty_and_malformed_files(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("  ", encoding="utf-8")
    assert "empty" in load_json_report(empty)[1].lower()

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    assert "valid json" in load_json_report(malformed)[1].lower()


def test_report_loader_rejects_non_object_and_incomplete_files(tmp_path):
    scalar = tmp_path / "scalar.json"
    scalar.write_text(json.dumps([]), encoding="utf-8")
    assert "expected a json object" in load_json_report(scalar)[1].lower()

    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"dataset": {}}), encoding="utf-8")
    report, error = load_json_report(partial, ("dataset", "repeated_seed_summary"))
    assert report is None
    assert "repeated_seed_summary" in error


def test_report_loader_accepts_valid_object(tmp_path):
    path = tmp_path / "valid.json"
    path.write_text(json.dumps({"dataset": {}, "summary": []}), encoding="utf-8")
    report, error = load_json_report(path, ("dataset", "summary"))
    assert error is None
    assert report == {"dataset": {}, "summary": []}


def test_research_summary_validator_flags_missing_nested_metrics():
    issues = validate_research_summary({
        "dataset": {},
        "repeated_seed_summary": {"S-learner": {"policy_value_ipw": {"mean": 1}}},
        "winner_counts": {},
        "primary_seed_policy_values": {},
        "bootstrap_vs_email_everyone": {},
        "creative_average_effects": {},
        "methodology": {},
    })
    assert any("policy_value_ipw is incomplete" in issue for issue in issues)
    assert any("primary_seed_policy_values must be a non-empty" in issue for issue in issues)


def test_optional_report_validators_flag_partial_snapshots():
    assert validate_spend_report({"baseline": {}})
    assert validate_advanced_statistics({"spend": {"ate": 1}})
