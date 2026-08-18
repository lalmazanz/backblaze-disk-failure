from pathlib import Path

import duckdb

DATA_PATH = Path("data/processed/q1_2026_features.parquet")

FEATURES_TO_CHECK = [
    "smart_5_raw_delta_1d",
    "smart_5_raw_delta_7d",
    "smart_5_raw_mean_7d",
    "smart_197_raw_delta_1d",
    "smart_197_raw_delta_7d",
    "smart_197_raw_mean_7d",
    "smart_198_raw_delta_1d",
    "smart_198_raw_delta_7d",
    "smart_198_raw_mean_7d",
]


def main() -> None:
    con = duckdb.connect()

    null_expressions = ",\n".join(
        f"""
        ROUND(
            100.0 * COUNT(*) FILTER (
                WHERE {feature} IS NULL
            ) / COUNT(*),
            2
        ) AS {feature}_null_pct
        """
        for feature in FEATURES_TO_CHECK
    )

    result = con.execute(
        f"""
        SELECT
            failure_next_7d,
            COUNT(*) AS rows,
            {null_expressions}
        FROM read_parquet('{DATA_PATH}')
        GROUP BY failure_next_7d
        ORDER BY failure_next_7d;
        """
    ).fetchdf()

    print("Feature null rates by target:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
