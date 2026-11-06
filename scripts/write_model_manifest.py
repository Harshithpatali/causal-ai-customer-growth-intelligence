"""Write reproducibility metadata for the persisted uplift model."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json
import os
import platform

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATASET = ROOT / "data" / "raw" / "hillstrom.csv"
MODEL = ARTIFACTS / "uplift_model.joblib"
METRICS = ARTIFACTS / "metrics.json"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    for path in (DATASET, MODEL, METRICS):
        if not path.is_file():
            raise FileNotFoundError(f"Required training output is missing: {path}")

    manifest = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.getenv("GITHUB_SHA", "local"),
        "python_version": platform.python_version(),
        "package_versions": {
            name: package_version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
        },
        "dataset": {
            "file": "data/raw/hillstrom.csv",
            "sha256": file_sha256(DATASET),
        },
        "model": {
            "file": "artifacts/uplift_model.joblib",
            "sha256": file_sha256(MODEL),
            "size_bytes": MODEL.stat().st_size,
        },
        "metrics": {
            "file": "artifacts/metrics.json",
            "sha256": file_sha256(METRICS),
        },
    }
    output = ARTIFACTS / "model_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote reproducibility manifest: {output}")


if __name__ == "__main__":
    main()
