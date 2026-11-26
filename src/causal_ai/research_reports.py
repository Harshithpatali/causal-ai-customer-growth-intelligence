"""Defensive readers and lightweight schema checks for generated research reports."""
import json
import math
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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def validate_research_summary(summary: dict) -> list[str]:
    """Return schema problems for the sections consumed by the research dashboard."""
    issues: list[str] = []
    required = (
        "dataset", "repeated_seed_summary", "winner_counts",
        "primary_seed_policy_values", "bootstrap_vs_email_everyone",
        "creative_average_effects", "methodology",
    )
    issues.extend(f"missing top-level section: {key}" for key in required if key not in summary)

    dataset = summary.get("dataset")
    if not isinstance(dataset, dict) or not _is_number(dataset.get("n_total")):
        issues.append("dataset.n_total must be numeric")

    if not isinstance(summary.get("winner_counts"), dict):
        issues.append("winner_counts must be an object")
    if not isinstance(summary.get("methodology"), dict):
        issues.append("methodology must be an object")

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
                    _is_number(values.get(key)) for key in ("mean", "standard_deviation", "minimum", "maximum")
                ):
                    issues.append(f"repeated_seed_summary.{strategy}.{metric} is incomplete or non-numeric")

    primary = summary.get("primary_seed_policy_values")
    if not isinstance(primary, dict) or not primary:
        issues.append("primary_seed_policy_values must be a non-empty object")
    else:
        for strategy, values in primary.items():
            if not isinstance(values, dict) or not all(
                _is_number(values.get(key)) for key in ("value", "ci_lower", "ci_upper", "auuc", "qini")
            ):
                issues.append(f"primary_seed_policy_values.{strategy} is incomplete or non-numeric")

    contrasts = summary.get("bootstrap_vs_email_everyone")
    if not isinstance(contrasts, dict) or not contrasts:
        issues.append("bootstrap_vs_email_everyone must be a non-empty object")
    else:
        for strategy, values in contrasts.items():
            if not isinstance(values, dict) or not all(
                _is_number(values.get(key)) for key in ("estimate", "bootstrap_ci_lower", "bootstrap_ci_upper")
            ):
                issues.append(f"bootstrap_vs_email_everyone.{strategy} is incomplete or non-numeric")

    creative = summary.get("creative_average_effects")
    if not isinstance(creative, dict):
        issues.append("creative_average_effects must be an object")
    else:
        for comparison, values in creative.items():
            effect = values.get("randomized_arm_spend_effect") if isinstance(values, dict) else None
            if not isinstance(effect, dict) or not all(
                _is_number(effect.get(key)) for key in ("difference", "ci_lower", "ci_upper")
            ):
                issues.append(f"creative_average_effects.{comparison} is incomplete or non-numeric")
    return issues


def validate_spend_report(report: dict) -> list[str]:
    required = ("baseline", "model_metrics", "split", "feature_permutation_importance", "by_randomized_arm")
    issues = [f"missing section: {key}" for key in required if key not in report]
    for section in ("baseline", "model_metrics"):
        values = report.get(section)
        if not isinstance(values, dict) or not all(_is_number(values.get(key)) for key in ("mae", "rmse", "r2")):
            issues.append(f"{section} must contain numeric mae, rmse and r2")
    model = report.get("model_metrics")
    if isinstance(model, dict):
        interval = model.get("conformal_interval")
        if not isinstance(interval, dict) or not all(
            _is_number(interval.get(key)) for key in ("nominal_coverage", "empirical_coverage", "mean_width")
        ):
            issues.append("model_metrics.conformal_interval is incomplete or non-numeric")
    split = report.get("split")
    if not isinstance(split, dict) or not all(
        _is_number(split.get(key)) for key in ("train", "calibration", "test")
    ):
        issues.append("split must contain numeric train, calibration and test counts")
    importance = report.get("feature_permutation_importance")
    if not isinstance(importance, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("feature"), str)
        or not _is_number(row.get("mae_increase")) for row in importance
    ):
        issues.append("feature_permutation_importance must contain feature names and numeric mae_increase values")
    arms = report.get("by_randomized_arm")
    if not isinstance(arms, dict) or any(not isinstance(row, dict) for row in arms.values()):
        issues.append("by_randomized_arm must be an object of arm-level metric objects")
    return issues


def validate_advanced_statistics(report: dict) -> list[str]:
    required = ("spend", "spend_cuped", "spend_aipw", "conversion_power_plan", "randomization_balance",
                "visit", "conversion")
    issues = [f"missing section: {key}" for key in required if key not in report]
    for section in ("spend", "spend_cuped", "spend_aipw"):
        values = report.get(section)
        if not isinstance(values, dict) or not all(_is_number(values.get(key)) for key in ("ate", "ci_lower", "ci_upper")):
            issues.append(f"{section} is incomplete or non-numeric")
    power = report.get("conversion_power_plan")
    if not isinstance(power, dict) or not all(
        _is_number(power.get(key)) for key in ("absolute_effect", "control_rate", "n_control", "n_treatment")
    ):
        issues.append("conversion_power_plan is incomplete or non-numeric")
    balance = report.get("randomization_balance")
    if not isinstance(balance, dict) or not _is_number(balance.get("max_abs_smd")):
        issues.append("randomization_balance is incomplete or non-numeric")
    for outcome in ("visit", "conversion"):
        values = report.get(outcome)
        if not isinstance(values, dict) or not _is_number(values.get("p_value")):
            issues.append(f"{outcome} statistics are incomplete or non-numeric")
    spend = report.get("spend")
    if isinstance(spend, dict) and not _is_number(spend.get("p_value")):
        issues.append("spend p-value is missing or non-numeric")
    return issues
