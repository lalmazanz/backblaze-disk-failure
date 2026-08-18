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
        WHERE date BETWEEN DATE '{start_date}'
                       AND DATE '{end_date}'
          AND failure_next_7d = 1;
        """
    ).fetchone()[0]

    negative_sample_size = positive_count * negative_ratio

    query = f"""
    WITH positives AS (
        SELECT
            {columns},
            failure_next_7d
        FROM read_parquet('{FEATURES_PATH}')
        WHERE date BETWEEN DATE '{start_date}'
                       AND DATE '{end_date}'
          AND failure_next_7d = 1
    ),

    negatives AS (
        SELECT
            {columns},
            failure_next_7d
        FROM read_parquet('{FEATURES_PATH}')
        WHERE date BETWEEN DATE '{start_date}'
                       AND DATE '{end_date}'
          AND failure_next_7d = 0
        ORDER BY HASH(
            model,
            serial_number,
            date,
            {RANDOM_STATE}
        )
        LIMIT {negative_sample_size}
    )

    SELECT * FROM positives
    UNION ALL
    SELECT * FROM negatives;
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
    FROM read_parquet('{FEATURES_PATH}')
    WHERE date BETWEEN DATE '{start_date}'
                   AND DATE '{end_date}';
    """

    return con.execute(query).fetchdf()


def load_failure_dates(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    query = f"""
    SELECT
        model || ':' || serial_number AS drive_id,
        MIN(date) AS failure_date
    FROM read_parquet('{SUBSET_PATH}')
    WHERE failure = 1
    GROUP BY model, serial_number;
    """

    return con.execute(query).fetchdf()
