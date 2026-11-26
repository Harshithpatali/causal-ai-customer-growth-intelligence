"""Defensive readers and lightweight schema checks for generated research reports."""
import json
from pathlib import Path
from typing import Any


def load_json_report(path: str | Path, required_keys: tuple[str, ...] = ()) -> tuple[dict | None, str | None]:
    """Load a JSON object without allowing missing, empty, malformed or partial files to crash the UI."""
    report_path = Path(path)
    if not report_path.exists():
        return None, f"File is missing: {report_path}"
    try:
        raw = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"File could not be read: {report_path} ({exc})"
    if not raw.strip():
        return None, f"File is empty: {report_path}"
    try:
        payload: Any = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"File is not valid JSON: {report_path} ({exc})"
    if not isinstance(payload, dict):
        return None, f"Expected a JSON object in {report_path}, got {type(payload).__name__}"
    missing = [key for key in required_keys if key not in payload]
    if missing:
        return None, f"Report is incomplete: {report_path} is missing required sections: {', '.join(missing)}"
    return payload, None


def validate_research_summary(summary: dict) -> list[str]:
    """Return schema problems for the sections consumed by the research dashboard."""
    issues: list[str] = []
    required = (
        "dataset", "repeated_seed_summary", "winner_counts",
        "primary_seed_policy_values", "bootstrap_vs_email_everyone",
        "creative_average_effects", "methodology",
    )
    issues.extend(f"missing top-level section: {key}" for key in required if key not in summary)

    repeated = summary.get("repeated_seed_summary")
    if not isinstance(repeated, dict) or not repeated:
        issues.append("repeated_seed_summary must be a non-empty object")
    else:
        for strategy, metrics in repeated.items():
            if not isinstance(metrics, dict):
                issues.append(f"repeated_seed_summary.{strategy} must be an object")
                continue
            for metric in ("policy_value_ipw", "auuc_ht", "qini_ht"):
                values = metrics.get(metric)
                if not isinstance(values, dict) or not all(
                    key in values for key in ("mean", "standard_deviation", "minimum", "maximum")
                ):
                    issues.append(f"repeated_seed_summary.{strategy}.{metric} is incomplete")

    primary = summary.get("primary_seed_policy_values")
    if not isinstance(primary, dict) or not primary:
        issues.append("primary_seed_policy_values must be a non-empty object")
    else:
        for strategy, values in primary.items():
            if not isinstance(values, dict) or not all(
                key in values for key in ("value", "ci_lower", "ci_upper", "auuc", "qini")
            ):
                issues.append(f"primary_seed_policy_values.{strategy} is incomplete")

    contrasts = summary.get("bootstrap_vs_email_everyone")
    if not isinstance(contrasts, dict) or not contrasts:
        issues.append("bootstrap_vs_email_everyone must be a non-empty object")
    else:
        for strategy, values in contrasts.items():
            if not isinstance(values, dict) or not all(
                key in values for key in ("estimate", "bootstrap_ci_lower", "bootstrap_ci_upper")
            ):
                issues.append(f"bootstrap_vs_email_everyone.{strategy} is incomplete")

    creative = summary.get("creative_average_effects")
    if not isinstance(creative, dict):
        issues.append("creative_average_effects must be an object")
    else:
        for comparison, values in creative.items():
            effect = values.get("randomized_arm_spend_effect") if isinstance(values, dict) else None
            if not isinstance(effect, dict) or not all(
                key in effect for key in ("difference", "ci_lower", "ci_upper")
            ):
                issues.append(f"creative_average_effects.{comparison} is incomplete")
    return issues


def validate_spend_report(report: dict) -> list[str]:
    required = ("baseline", "model_metrics", "split", "feature_permutation_importance", "by_randomized_arm")
    issues = [f"missing section: {key}" for key in required if key not in report]
    for section in ("baseline", "model_metrics"):
        values = report.get(section)
        if not isinstance(values, dict) or not all(key in values for key in ("mae", "rmse", "r2")):
            issues.append(f"{section} must contain mae, rmse and r2")
    model = report.get("model_metrics")
    if isinstance(model, dict):
        interval = model.get("conformal_interval")
        if not isinstance(interval, dict) or not all(
            key in interval for key in ("nominal_coverage", "empirical_coverage", "mean_width")
        ):
            issues.append("model_metrics.conformal_interval is incomplete")
    return issues


def validate_advanced_statistics(report: dict) -> list[str]:
    required = ("spend", "spend_cuped", "spend_aipw", "conversion_power_plan", "randomization_balance",
                "visit", "conversion")
    issues = [f"missing section: {key}" for key in required if key not in report]
    for section in ("spend", "spend_cuped", "spend_aipw"):
        values = report.get(section)
        if not isinstance(values, dict) or not all(key in values for key in ("ate", "ci_lower", "ci_upper")):
            issues.append(f"{section} is incomplete")
    return issues
