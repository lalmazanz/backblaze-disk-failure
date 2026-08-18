import duckdb

from src.config import (
    LAST_OBSERVABLE_DATE,
    PREDICTION_HORIZON_DAYS,
    SUBSET_PATH,
    TARGET_PATH,
)


def main() -> None:
    con = duckdb.connect()

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)

    query = f"""
    COPY (
        WITH base AS (
            SELECT
                date,
                serial_number,
                model,
                failure,
                smart_1_raw,
                smart_5_raw,
                smart_7_raw,
                smart_9_raw,
                smart_194_raw,
                smart_197_raw,
                smart_198_raw,
                smart_199_raw
            FROM read_parquet('{SUBSET_PATH}')
        ),

        failure_dates AS (
            SELECT
                serial_number,
                model,
                MIN(date) AS failure_date
            FROM base
            WHERE failure = 1
            GROUP BY
                serial_number,
                model
        )

        SELECT
            b.date,
            b.serial_number,
            b.model,
            b.smart_1_raw,
            b.smart_5_raw,
            b.smart_7_raw,
            b.smart_9_raw,
            b.smart_194_raw,
            b.smart_197_raw,
            b.smart_198_raw,
            b.smart_199_raw,

            CASE
                WHEN f.failure_date > b.date
                 AND f.failure_date
                     <= b.date + INTERVAL '{PREDICTION_HORIZON_DAYS} days'
                THEN 1
                ELSE 0
            END AS failure_next_7d

        FROM base AS b

        LEFT JOIN failure_dates AS f
            ON b.serial_number = f.serial_number
            AND b.model = f.model

        WHERE b.date <= DATE '{LAST_OBSERVABLE_DATE}'
          AND b.failure = 0
    )
    TO '{TARGET_PATH}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    );
    """

    print("Building 7-day failure target...")
    con.execute(query)

    summary = con.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            SUM(failure_next_7d) AS positive_rows,
            COUNT(*) - SUM(failure_next_7d) AS negative_rows,
            ROUND(
                100.0 * SUM(failure_next_7d) / COUNT(*),
                4
            ) AS positive_rate_pct,
            COUNT(
                DISTINCT model || ':' || serial_number
            ) AS unique_drives,
            MIN(date) AS min_date,
            MAX(date) AS max_date
        FROM read_parquet('{TARGET_PATH}');
        """
    ).fetchdf()

    event_summary = con.execute(
        f"""
        SELECT
            COUNT(
                DISTINCT CASE
                    WHEN failure_next_7d = 1
                    THEN model || ':' || serial_number
                END
            ) AS drives_with_positive_window,

            MIN(
                CASE
                    WHEN failure_next_7d = 1 THEN date
                END
            ) AS first_positive_date,

            MAX(
                CASE
                    WHEN failure_next_7d = 1 THEN date
                END
            ) AS last_positive_date

        FROM read_parquet('{TARGET_PATH}');
        """
    ).fetchdf()

    print("\nPositive-event summary:")
    print(event_summary.to_string(index=False))

    print("\nTarget dataset created:")
    print(summary.to_string(index=False))

    print(f"\nSaved to: {TARGET_PATH}")


if __name__ == "__main__":
    main()
