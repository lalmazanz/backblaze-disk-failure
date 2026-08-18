import duckdb
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    FEATURES_PATH,
    NEGATIVE_RATIO,
    RANDOM_STATE,
    SUBSET_PATH,
)


def load_undersampled_train(
    con: duckdb.DuckDBPyConnection,
    start_date: str,
    end_date: str,
    negative_ratio: int = NEGATIVE_RATIO,
) -> pd.DataFrame:
    columns = ", ".join(FEATURE_COLUMNS)

    positive_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{FEATURES_PATH}')
        WHERE date BETWEEN
              DATE '{start_date}'
              AND DATE '{end_date}'
          AND failure_next_7d = 1;
        """
    ).fetchone()[0]

    negative_sample_size = positive_count * negative_ratio

    query = f"""
    WITH positives AS (
        SELECT
            date,
            model,
            serial_number,
            {columns},
            failure_next_7d
        FROM read_parquet(
            '{FEATURES_PATH}'
        )
        WHERE date BETWEEN
              DATE '{start_date}'
              AND DATE '{end_date}'
          AND failure_next_7d = 1
    ),

    negative_candidates AS (
        SELECT
            date,
            model,
            serial_number,
            {columns},
            failure_next_7d,
            HASH(
                model,
                serial_number,
                date,
                {RANDOM_STATE}
            ) AS sample_hash
        FROM read_parquet(
            '{FEATURES_PATH}'
        )
        WHERE date BETWEEN
              DATE '{start_date}'
              AND DATE '{end_date}'
          AND failure_next_7d = 0
    ),

    negatives AS (
        SELECT
            date,
            model,
            serial_number,
            {columns},
            failure_next_7d
        FROM negative_candidates
        ORDER BY
            sample_hash,
            model,
            serial_number,
            date
        LIMIT {negative_sample_size}
    ),

    combined AS (
        SELECT * FROM positives

        UNION ALL

        SELECT * FROM negatives
    )

    SELECT
        {columns},
        failure_next_7d
    FROM combined
    ORDER BY
        model,
        serial_number,
        date,
        failure_next_7d;
    """

    return con.execute(query).fetchdf()


def load_period(
    con: duckdb.DuckDBPyConnection,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    columns = ", ".join(FEATURE_COLUMNS)

    query = f"""
    SELECT
        date,
        serial_number,
        model,
        {columns},
        failure_next_7d
    FROM read_parquet(
        '{FEATURES_PATH}'
    )
    WHERE date BETWEEN
          DATE '{start_date}'
          AND DATE '{end_date}'
    ORDER BY
        date,
        model,
        serial_number;
    """

    return con.execute(query).fetchdf()


def load_failure_dates(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    query = f"""
    SELECT
        model
            || ':'
            || serial_number
            AS drive_id,
        MIN(date) AS failure_date
    FROM read_parquet(
        '{SUBSET_PATH}'
    )
    WHERE failure = 1
    GROUP BY
        model,
        serial_number
    ORDER BY
        model,
        serial_number;
    """

    return con.execute(query).fetchdf()
