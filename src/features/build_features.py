from pathlib import Path

import duckdb

INPUT_PATH = Path("data/processed/q1_2026_failure_target.parquet")
OUTPUT_PATH = Path("data/processed/q1_2026_features.parquet")

SMART_COLUMNS = [
    "smart_1_raw",
    "smart_5_raw",
    "smart_7_raw",
    "smart_9_raw",
    "smart_194_raw",
    "smart_197_raw",
    "smart_198_raw",
    "smart_199_raw",
]


def main() -> None:
    con = duckdb.connect()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lag_expressions = []
    rolling_expressions = []
    feature_expressions = []

    for column in SMART_COLUMNS:
        lag_expressions.extend(
            [
                f"LAG(date, 1) OVER drive_window AS {column}_date_lag_1",
                f"LAG({column}, 1) OVER drive_window AS {column}_lag_1",
                f"LAG(date, 7) OVER drive_window AS {column}_date_lag_7",
                f"LAG({column}, 7) OVER drive_window AS {column}_lag_7",
            ]
        )

        rolling_expressions.extend(
            [
                f"""
                AVG({column}) OVER (
                    PARTITION BY model, serial_number
                    ORDER BY date
                    RANGE BETWEEN INTERVAL '6 days' PRECEDING
                    AND CURRENT ROW
                ) AS {column}_mean_7d
                """,
                f"""
                MAX({column}) OVER (
                    PARTITION BY model, serial_number
                    ORDER BY date
                    RANGE BETWEEN INTERVAL '6 days' PRECEDING
                    AND CURRENT ROW
                ) AS {column}_max_7d
                """,
            ]
        )

        feature_expressions.extend(
            [
                column,
                f"""
                CASE
                    WHEN date - {column}_date_lag_1 = 1
                    THEN {column} - {column}_lag_1
                    ELSE NULL
                END AS {column}_delta_1d
                """,
                f"""
                CASE
                    WHEN date - {column}_date_lag_7 = 7
                    THEN {column} - {column}_lag_7
                    ELSE NULL
                END AS {column}_delta_7d
                """,
                f"{column}_mean_7d",
                f"{column}_max_7d",
            ]
        )

    lag_sql = ",\n".join(lag_expressions)
    rolling_sql = ",\n".join(rolling_expressions)
    features_sql = ",\n".join(feature_expressions)

    query = f"""
    COPY (
        WITH lagged AS (
            SELECT
                *,
                {lag_sql},
                {rolling_sql}
            FROM read_parquet('{INPUT_PATH}')
            WINDOW drive_window AS (
                PARTITION BY model, serial_number
                ORDER BY date
            )
        )

        SELECT
            date,
            serial_number,
            model,
            {features_sql},
            failure_next_7d
        FROM lagged
    )
    TO '{OUTPUT_PATH}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    );
    """

    print("Building temporal SMART features...")
    con.execute(query)

    summary = con.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT model || ':' || serial_number) AS unique_drives,
            SUM(failure_next_7d) AS positive_rows,
            COUNT(*) FILTER (
                WHERE smart_5_raw_delta_7d IS NOT NULL
            ) AS rows_with_7d_history,
            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE smart_5_raw_delta_7d IS NOT NULL
                ) / COUNT(*),
                2
            ) AS pct_with_7d_history,
            MIN(date) AS min_date,
            MAX(date) AS max_date
        FROM read_parquet('{OUTPUT_PATH}');
        """
    ).fetchdf()

    print("\nFeature dataset created:")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
