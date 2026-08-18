import subprocess
import sys
from pathlib import Path

from src.config import RAW_DATA_GLOB
from src.logging_utils import get_logger

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_STEPS = [
    (
        "Build selected-model subset",
        "src.ingestion.build_subset",
    ),
    (
        "Build 7-day failure target",
        "src.features.build_target",
    ),
    (
        "Build temporal SMART features",
        "src.features.build_features",
    ),
    (
        "Export final Random Forest model",
        "src.modeling.export_final_model",
    ),
    (
        "Export full demo predictions",
        "src.modeling.export_demo_data",
    ),
    (
        "Export compact demo dataset",
        "src.modeling.export_demo_compact",
    ),
]


def validate_raw_data() -> None:
    raw_pattern = PROJECT_ROOT / RAW_DATA_GLOB

    raw_files = sorted(raw_pattern.parent.glob(raw_pattern.name))

    if not raw_files:
        raise FileNotFoundError(
            "No raw Backblaze CSV files were found. "
            f"Expected files matching: {RAW_DATA_GLOB}"
        )

    logger.info(
        "Raw data files found: %s",
        len(raw_files),
    )


def run_step(
    name: str,
    module: str,
) -> None:
    print()
    print("=" * 70)
    print(name)
    print(f"Module: {module}")
    print("=" * 70)

    subprocess.run(
        [
            sys.executable,
            "-m",
            module,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    print("Backblaze Disk Failure — reproducible pipeline")

    print(f"Project root: {PROJECT_ROOT}")

    validate_raw_data()

    for name, module in PIPELINE_STEPS:
        run_step(
            name,
            module,
        )

    print()
    print("=" * 70)
    logger.info("Pipeline completed successfully.")
    print("=" * 70)

    print("\nGenerated artifacts:")

    artifacts = [
        ("data/interim/q1_2026_selected_models.parquet"),
        ("data/processed/q1_2026_failure_target.parquet"),
        ("data/processed/q1_2026_features.parquet"),
        "models/final_model.joblib",
        "models/feature_schema.json",
        ("data/processed/demo_predictions.parquet"),
        ("data/processed/demo_predictions_compact.parquet"),
    ]

    for artifact in artifacts:
        path = PROJECT_ROOT / artifact

        status = "OK" if path.exists() else "MISSING"

        print(f"[{status}] {artifact}")


if __name__ == "__main__":
    main()
