from pathlib import Path

import duckdb

DATA_PATH = Path("data/interim/q1_2026_selected_models.parquet")

SMART_COLUMNS = [
    "smart_1_raw",
    "smart_5_raw",
    "smart_7_raw",
    "smart_9_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_194_raw",
    "smart_197_raw",
    "smart_198_raw",
    "smart_199_raw",
]


def main() -> None:
    con = duckdb.connect()

    availability = ",\n".join(
        f"""
        ROUND(
            100.0 * COUNT({column}) / COUNT(*),
            2
        ) AS {column}_available_pct
        """
        for column in SMART_COLUMNS
    )

    query = f"""
    SELECT
        model,
        COUNT(*) AS drive_days,
        {availability}
    FROM read_parquet('{DATA_PATH}')
    GROUP BY model
    ORDER BY model;
    """

    result = con.execute(query).fetchdf()

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
