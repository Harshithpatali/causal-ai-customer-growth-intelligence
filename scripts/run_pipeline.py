import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from causal_ai.pipeline import run_pipeline

if __name__ == "__main__":
    result = run_pipeline()
    print("Pipeline completed.")
    print("Rows:", result["validation"]["rows"])
    print("Test-set ATE on spend:", result["metrics"]["test_ate_spend"]["ate"])
    print("Qini:", result["metrics"]["qini"])
