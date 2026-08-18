from pathlib import Path

import duckdb
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path("data/processed/q1_2026_features.parquet")

FEATURE_COLUMNS = [
    "has_1d_history",
    "has_7d_history",
    "smart_1_raw",
    "smart_1_raw_delta_1d",
    "smart_1_raw_delta_7d",
    "smart_1_raw_mean_7d",
    "smart_1_raw_max_7d",
    "smart_5_raw",
    "smart_5_raw_delta_1d",
    "smart_5_raw_delta_7d",
    "smart_5_raw_mean_7d",
    "smart_5_raw_max_7d",
    "smart_7_raw",
    "smart_7_raw_delta_1d",
    "smart_7_raw_delta_7d",
    "smart_7_raw_mean_7d",
    "smart_7_raw_max_7d",
    "smart_9_raw",
    "smart_9_raw_delta_1d",
    "smart_9_raw_delta_7d",
    "smart_9_raw_mean_7d",
    "smart_9_raw_max_7d",
    "smart_194_raw",
    "smart_194_raw_delta_1d",
    "smart_194_raw_delta_7d",
    "smart_194_raw_mean_7d",
    "smart_194_raw_max_7d",
    "smart_197_raw",
    "smart_197_raw_delta_1d",
    "smart_197_raw_delta_7d",
    "smart_197_raw_mean_7d",
    "smart_197_raw_max_7d",
    "smart_198_raw",
    "smart_198_raw_delta_1d",
    "smart_198_raw_delta_7d",
    "smart_198_raw_mean_7d",
    "smart_198_raw_max_7d",
    "smart_199_raw",
    "smart_199_raw_delta_1d",
    "smart_199_raw_delta_7d",
    "smart_199_raw_mean_7d",
    "smart_199_raw_max_7d",
]

TRAIN_START = "2026-01-01"
TRAIN_END = "2026-03-10"
NEGATIVE_RATIO = 50
RANDOM_STATE = 42


def load_train(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    columns = ", ".join(FEATURE_COLUMNS)

    positive_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{DATA_PATH}')
        WHERE date BETWEEN DATE '{TRAIN_START}' AND DATE '{TRAIN_END}'
          AND failure_next_7d = 1
        """
    ).fetchone()[0]

    negative_sample_size = positive_count * NEGATIVE_RATIO

    query = f"""
    WITH positives AS (
        SELECT
            {columns},
            failure_next_7d
        FROM read_parquet('{DATA_PATH}')
        WHERE date BETWEEN DATE '{TRAIN_START}' AND DATE '{TRAIN_END}'
          AND failure_next_7d = 1
    ),

    negatives AS (
        SELECT
            {columns},
            failure_next_7d
        FROM read_parquet('{DATA_PATH}')
        WHERE date BETWEEN DATE '{TRAIN_START}' AND DATE '{TRAIN_END}'
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
    SELECT * FROM negatives
    """

    return con.execute(query).fetchdf()


def print_importance(
    name: str,
    values,
) -> None:
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": values,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(f"\n{name} — Top 15 features")
    print(importance.head(15).to_string(index=False))


def main() -> None:
    con = duckdb.connect()

    print("Loading final training sample...")
    train = load_train(con)

    x_train = train[FEATURE_COLUMNS]
    y_train = train["failure_next_7d"]

    print("Training LightGBM...")
    lightgbm = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    lightgbm.fit(
        x_train,
        y_train,
    )

    print_importance(
        "LightGBM",
        lightgbm.feature_importances_,
    )

    print("\nTraining Random Forest...")
    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    random_forest.fit(
        x_train,
        y_train,
    )

    print_importance(
        "Random Forest",
        random_forest.feature_importances_,
    )


if __name__ == "__main__":
    main()
