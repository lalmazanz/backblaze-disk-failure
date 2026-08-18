from pathlib import Path

import duckdb

DATA_PATH = Path("data/processed/q1_2026_features.parquet")


def main() -> None:
    con = duckdb.connect()

    result = con.execute(
        f"""
        SELECT
            DATE_TRUNC('week', date) AS week_start,
            COUNT(*) AS rows,
            SUM(failure_next_7d) AS positive_rows,
            COUNT(
                DISTINCT CASE
                    WHEN failure_next_7d = 1
                    THEN model || ':' || serial_number
                END
            ) AS positive_drives,
            ROUND(
                100.0 * SUM(failure_next_7d) / COUNT(*),
                4
            ) AS positive_rate_pct
        FROM read_parquet('{DATA_PATH}')
        GROUP BY week_start
        ORDER BY week_start;
        """
    ).fetchdf()

    print("Target distribution by week:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
