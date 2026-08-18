from pathlib import Path

import duckdb

RAW_GLOB = Path("data/raw/*.csv")
OUTPUT_PATH = Path("data/interim/q1_2026_selected_models.parquet")

SELECTED_MODELS = [
    "ST12000NM0008",
    "TOSHIBA MG08ACA16TA",
    "HGST HUH721212ALN604",
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    models_sql = ", ".join(f"'{model}'" for model in SELECTED_MODELS)

    con = duckdb.connect()

    query = f"""
    COPY (
        SELECT *
        FROM read_csv_auto(
            '{RAW_GLOB}',
            union_by_name = true
        )
        WHERE model IN ({models_sql})
    )
    TO '{OUTPUT_PATH}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    );
    """

    print("Building Q1 2026 subset...")
    con.execute(query)

    result = con.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT serial_number) AS unique_drives,
            SUM(failure) AS failures,
            MIN(date) AS min_date,
            MAX(date) AS max_date
        FROM read_parquet('{OUTPUT_PATH}');
        """
    ).fetchdf()

    print("\nSubset created:")
    print(result.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
