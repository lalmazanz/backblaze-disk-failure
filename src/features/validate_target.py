from pathlib import Path

import duckdb

SOURCE_PATH = Path("data/interim/q1_2026_selected_models.parquet")
TARGET_PATH = Path("data/processed/q1_2026_failure_target.parquet")


def main() -> None:
    con = duckdb.connect()

    result = con.execute(
        f"""
        WITH failures AS (
            SELECT
                serial_number,
                model,
                MIN(date) AS failure_date
            FROM read_parquet('{SOURCE_PATH}')
            WHERE failure = 1
            GROUP BY serial_number, model
        ),

        represented AS (
            SELECT DISTINCT
                serial_number,
                model
            FROM read_parquet('{TARGET_PATH}')
            WHERE failure_next_7d = 1
        )

        SELECT
            f.model,
            COUNT(*) AS failures,
            COUNT(r.serial_number) AS represented_failures,
            COUNT(*) - COUNT(r.serial_number) AS unrepresented_failures
        FROM failures AS f
        LEFT JOIN represented AS r
            ON f.serial_number = r.serial_number
            AND f.model = r.model
        GROUP BY f.model
        ORDER BY f.model;
        """
    ).fetchdf()

    print("Failure coverage by model:")
    print(result.to_string(index=False))

    missing = con.execute(
        f"""
        WITH failures AS (
            SELECT
                serial_number,
                model,
                MIN(date) AS failure_date
            FROM read_parquet('{SOURCE_PATH}')
            WHERE failure = 1
            GROUP BY serial_number, model
        ),

        represented AS (
            SELECT DISTINCT
                serial_number,
                model
            FROM read_parquet('{TARGET_PATH}')
            WHERE failure_next_7d = 1
        )

        SELECT
            f.model,
            f.failure_date,
            COUNT(*) AS failures
        FROM failures AS f
        LEFT JOIN represented AS r
            ON f.serial_number = r.serial_number
            AND f.model = r.model
        WHERE r.serial_number IS NULL
        GROUP BY f.model, f.failure_date
        ORDER BY f.failure_date, f.model;
        """
    ).fetchdf()

    print("\nUnrepresented failures by date:")
    print(missing.to_string(index=False))

    window_check = con.execute(
        f"""
        WITH failures AS (
            SELECT
                serial_number,
                model,
                MIN(date) AS failure_date
            FROM read_parquet('{SOURCE_PATH}')
            WHERE failure = 1
            GROUP BY serial_number, model
        ),
    
        represented AS (
            SELECT DISTINCT
                serial_number,
                model
            FROM read_parquet('{TARGET_PATH}')
            WHERE failure_next_7d = 1
        ),
    
        missing AS (
            SELECT
                f.serial_number,
                f.model,
                f.failure_date
            FROM failures AS f
            LEFT JOIN represented AS r
                ON f.serial_number = r.serial_number
                AND f.model = r.model
            WHERE r.serial_number IS NULL
              AND f.failure_date <= DATE '2026-03-24'
        )
    
        SELECT
            m.model,
            m.failure_date,
            m.serial_number,
            COUNT(s.date) AS observations_previous_7d
        FROM missing AS m
        LEFT JOIN read_parquet('{SOURCE_PATH}') AS s
            ON m.serial_number = s.serial_number
            AND m.model = s.model
            AND s.date >= m.failure_date - INTERVAL '7 days'
            AND s.date < m.failure_date
            AND s.failure = 0
        GROUP BY
            m.model,
            m.failure_date,
            m.serial_number
        ORDER BY
            observations_previous_7d,
            m.failure_date;
        """
    ).fetchdf()

    print("\nPrevious observations for unrepresented failures:")
    print(window_check.to_string(index=False))


if __name__ == "__main__":
    main()
