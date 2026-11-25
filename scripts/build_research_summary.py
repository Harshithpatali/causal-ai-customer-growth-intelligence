"""Create a compact, dashboard-safe snapshot from benchmark JSON outputs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", default="reports/uplift_benchmark.json")
    parser.add_argument("--repeated", default="reports/repeated_seed_benchmark.json")
    parser.add_argument("--output", default="reports/research_summary.json")
    args = parser.parse_args()

    primary = json.loads(Path(args.primary).read_text(encoding="utf-8"))
    repeated = json.loads(Path(args.repeated).read_text(encoding="utf-8"))
    binary = primary["binary_uplift_benchmark"]
    summary = repeated["summary"]
    compact = {
        "dataset": primary["design"],
        "repeated_seed_summary": summary["metric_summary"],
        "winner_counts": summary["winner_counts"],
        "primary_seed_policy_values": {
            name: {
                "value": metrics["policy_value_ipw"]["value"],
                "ci_lower": metrics["policy_value_ipw"]["ci_lower"],
                "ci_upper": metrics["policy_value_ipw"]["ci_upper"],
                "auuc": metrics["auuc_ht"],
                "qini": metrics["qini_ht"],
            }
            for name, metrics in binary.items()
        },
        "bootstrap_vs_email_everyone": primary["bootstrap_policy_contrasts_vs_email_everyone"],
        "creative_average_effects": primary["multi_treatment_average_effects"],
        "methodology": {
            "repeated_splits": summary["interpretation"],
            "bootstrap": primary["estimators"]["bootstrap"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote compact research summary: {output}")


if __name__ == "__main__":
    main()
