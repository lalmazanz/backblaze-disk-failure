import duckdb

from src.config import (
    FEATURE_COLUMNS,
    TEST_END,
    TEST_START,
)
from src.modeling.data import load_failure_dates, load_period
from src.modeling.inference import predict_risk

OUTPUT_PATH = "data/processed/demo_predictions.parquet"


def main() -> None:
    con = duckdb.connect()

    print("Loading test period...")

    test = load_period(
        con,
        TEST_START,
        TEST_END,
    ).copy()

    print(f"Test rows: {len(test)}")

    print("Scoring test period with exported model...")

    test["risk_score"] = predict_risk(test[FEATURE_COLUMNS])

    test["drive_id"] = test["model"] + ":" + test["serial_number"]

    failure_dates = load_failure_dates(con)

    test = test.merge(
        failure_dates,
        on="drive_id",
        how="left",
    )

    test["days_to_failure"] = (test["failure_date"] - test["date"]).dt.days

    test["within_failure_horizon"] = test["failure_next_7d"] == 1

    daily_rank = test.groupby("date")["risk_score"].rank(
        method="first",
        ascending=False,
    )

    daily_count = test.groupby("date")["risk_score"].transform("size")

    test["risk_rank"] = daily_rank.astype(int)

    test["risk_percentile"] = daily_rank / daily_count

    test["top_1pct_alert"] = test["risk_percentile"] <= 0.01

    output_columns = [
        "date",
        "serial_number",
        "model",
        "drive_id",
        "risk_score",
        "risk_rank",
        "risk_percentile",
        "top_1pct_alert",
        "failure_next_7d",
        "within_failure_horizon",
        "failure_date",
        "days_to_failure",
        *FEATURE_COLUMNS,
    ]

    demo = test[output_columns].sort_values(
        [
            "date",
            "risk_rank",
        ]
    )

    demo.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    print("\nDemo dataset created:")
    print(f"Rows: {len(demo)}")
    print(f"Unique drives: {demo['drive_id'].nunique()}")
    print(f"Top 1% alert rows: {demo['top_1pct_alert'].sum()}")
    print(f"Positive rows: {demo['failure_next_7d'].sum()}")
    print(
        "Drives with known failure: "
        f"{
            demo.loc[
                demo['failure_date'].notna(),
                'drive_id',
            ].nunique()
        }"
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
