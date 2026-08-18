from pathlib import Path

import duckdb

RAW_GLOB = Path("data/raw/*.csv")

def main() -> None:
    con = duckdb.connect()

    query = f"""
    SELECT
        model,
        COUNT(DISTINCT serial_number) AS unique_drives,
        COUNT(*) AS drive_days,
        SUM(failure) AS failures,
        ROUND(100.0 * SUM(failure) / COUNT(*), 6) AS failure_rate_pct
    FROM read_csv_auto(
        '{RAW_GLOB}',
        union_by_name = true
    )
    WHERE model IS NOT NULL
    GROUP BY model
    ORDER BY failures DESC, drive_days DESC;
    """

    result = con.execute(query).fetchdf()

    print(result.head(30).to_string(index=False))


if __name__ == "__main__":
    main()