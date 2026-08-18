from pathlib import Path

import pandas as pd

from src.config import RANDOM_STATE

INPUT_PATH = Path("data/processed/demo_predictions.parquet")
OUTPUT_PATH = Path("data/processed/demo_predictions_compact.parquet")

NORMAL_DRIVE_SAMPLE = 500


def main() -> None:
    print("Loading full demo dataset...")

    demo = pd.read_parquet(INPUT_PATH)

    print(f"Rows: {len(demo)}")
    print(f"Unique drives: {demo['drive_id'].nunique()}")

    failed_drives = set(
        demo.loc[
            demo["failure_date"].notna(),
            "drive_id",
        ]
    )

    alerted_drives = set(
        demo.loc[
            demo["top_1pct_alert"],
            "drive_id",
        ]
    )

    interesting_drives = failed_drives | alerted_drives

    normal_drive_candidates = demo.loc[
        ~demo["drive_id"].isin(interesting_drives),
        "drive_id",
    ].drop_duplicates()

    normal_sample_size = min(
        NORMAL_DRIVE_SAMPLE,
        len(normal_drive_candidates),
    )

    sampled_normal_drives = set(
        normal_drive_candidates.sample(
            n=normal_sample_size,
            random_state=RANDOM_STATE,
        )
    )

    selected_drives = interesting_drives | sampled_normal_drives

    compact = demo[demo["drive_id"].isin(selected_drives)].copy()

    compact["demo_category"] = "normal_sample"

    compact.loc[
        compact["drive_id"].isin(alerted_drives),
        "demo_category",
    ] = "alerted"

    compact.loc[
        compact["drive_id"].isin(failed_drives),
        "demo_category",
    ] = "failed"

    compact.loc[
        compact["drive_id"].isin(failed_drives & alerted_drives),
        "demo_category",
    ] = "failed_and_alerted"

    compact = compact.sort_values(
        [
            "demo_category",
            "drive_id",
            "date",
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    compact.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    print("\nCompact demo dataset created:")
    print(f"Rows: {len(compact)}")
    print(f"Unique drives: {compact['drive_id'].nunique()}")
    print(f"Failed drives: {len(failed_drives)}")
    print(f"Alerted drives: {len(alerted_drives)}")
    print(f"Normal sampled drives: {len(sampled_normal_drives)}")

    print("\nCategory breakdown:")
    print(
        compact[
            [
                "drive_id",
                "demo_category",
            ]
        ]
        .drop_duplicates()["demo_category"]
        .value_counts()
        .to_string()
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
