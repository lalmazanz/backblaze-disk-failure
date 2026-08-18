from pathlib import Path

import duckdb

DATA_PATH = Path("data/processed/q1_2026_features.parquet")


def main() -> None:
    con = duckdb.connect()

    result = con.execute(
        f"""
        WITH labeled AS (
            SELECT
                *,
                CASE
                    WHEN date <= DATE '2026-02-28'
                        THEN 'train'
                    WHEN date <= DATE '2026-03-10'
                        THEN 'validation'
                    ELSE 'test'
                END AS split
            FROM read_parquet('{DATA_PATH}')
        )

        SELECT
            split,
            MIN(date) AS min_date,
            MAX(date) AS max_date,
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
        FROM labeled
        GROUP BY split
        ORDER BY
            CASE split
                WHEN 'train' THEN 1
                WHEN 'validation' THEN 2
                WHEN 'test' THEN 3
            END;
        """
    ).fetchdf()

    print("Proposed temporal split:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
