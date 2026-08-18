from pathlib import Path

import duckdb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

NEGATIVE_RATIO = 50
RANDOM_STATE = 42


def load_split(
    con: duckdb.DuckDBPyConnection,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    columns = ", ".join(FEATURE_COLUMNS)

    query = f"""
    SELECT
        {columns},
        failure_next_7d
    FROM read_parquet('{DATA_PATH}')
    WHERE date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
    """

    return con.execute(query).fetchdf()


def build_undersampled_train(
    train: pd.DataFrame,
) -> pd.DataFrame:
    positives = train[train["failure_next_7d"] == 1]

    negatives = train[train["failure_next_7d"] == 0]

    sampled_negatives = negatives.sample(
        n=len(positives) * NEGATIVE_RATIO,
        random_state=RANDOM_STATE,
        replace=False,
    )

    return (
        pd.concat(
            [positives, sampled_negatives],
            ignore_index=True,
        )
        .sample(
            frac=1,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )


def main() -> None:
    con = duckdb.connect()

    print("Loading train...")
    train = load_split(
        con,
        "2026-01-01",
        "2026-02-28",
    )

    print("Loading validation...")
    validation = load_split(
        con,
        "2026-03-01",
        "2026-03-10",
    )

    sampled_train = build_undersampled_train(train)

    x_train = sampled_train[FEATURE_COLUMNS]
    y_train = sampled_train["failure_next_7d"]

    x_validation = validation[FEATURE_COLUMNS]
    y_validation = validation["failure_next_7d"]

    print(f"Training rows: {len(sampled_train)}")

    print(f"Train positives: {int((y_train == 1).sum())}")

    print(f"Train negatives: {int((y_train == 0).sum())}")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training Random Forest...")
    model.fit(
        x_train,
        y_train,
    )

    probabilities = model.predict_proba(x_validation)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    print("\nRANDOM FOREST — VALIDATION")
    print(f"ROC-AUC:   {roc_auc_score(y_validation, probabilities):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_validation, probabilities):.4f}")
    print(
        f"Precision: {precision_score(y_validation, predictions, zero_division=0):.4f}"
    )
    print(f"Recall:    {recall_score(y_validation, predictions, zero_division=0):.4f}")


if __name__ == "__main__":
    main()
