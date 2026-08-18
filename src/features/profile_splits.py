import duckdb

from src.config import (
    FEATURES_PATH,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)


def main() -> None:
    con = duckdb.connect()

    result = con.execute(
        f"""
        WITH labeled AS (
            SELECT
                *,
                CASE
                    WHEN date BETWEEN
                        DATE '{TRAIN_START}'
                        AND DATE '{TRAIN_END}'
                    THEN 'train'

                    WHEN date BETWEEN
                        DATE '{VALIDATION_START}'
                        AND DATE '{VALIDATION_END}'
                    THEN 'validation'

                    WHEN date BETWEEN
                        DATE '{TEST_START}'
                        AND DATE '{TEST_END}'
                    THEN 'test'

                    ELSE NULL
                END AS split
            FROM read_parquet('{FEATURES_PATH}')
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
                100.0
                * SUM(failure_next_7d)
                / COUNT(*),
                4
            ) AS positive_rate_pct
        FROM labeled
        WHERE split IS NOT NULL
        GROUP BY split
        ORDER BY
            CASE split
                WHEN 'train' THEN 1
                WHEN 'validation' THEN 2
                WHEN 'test' THEN 3
            END;
        """
    ).fetchdf()

    print("Purged temporal split:")
    print(result.to_string(index=False))

    print("\nExcluded purge periods:")
    print(f"Train -> validation: {TRAIN_END} to {VALIDATION_START}")
    print(f"Final-train -> test handled separately: test begins {TEST_START}")


if __name__ == "__main__":
    main()
