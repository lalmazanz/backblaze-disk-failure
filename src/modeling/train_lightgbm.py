from pathlib import Path

import duckdb
import pandas as pd
from lightgbm import LGBMClassifier
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


def evaluate(
    name: str,
    model: LGBMClassifier,
    x: pd.DataFrame,
    y: pd.Series,
) -> None:
    probabilities = model.predict_proba(x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n{name}")
    print(f"ROC-AUC:   {roc_auc_score(y, probabilities):.4f}")
    print(f"PR-AUC:    {average_precision_score(y, probabilities):.4f}")
    print(f"Precision: {precision_score(y, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y, predictions, zero_division=0):.4f}")


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

    x_train = train[FEATURE_COLUMNS]
    y_train = train["failure_next_7d"]

    x_validation = validation[FEATURE_COLUMNS]
    y_validation = validation["failure_next_7d"]

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    print(f"Train positives: {positive_count}")
    print(f"Train negatives: {negative_count}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    print("Training LightGBM...")
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        eval_metric="average_precision",
    )

    evaluate(
        "TRAIN",
        model,
        x_train,
        y_train,
    )

    evaluate(
        "VALIDATION",
        model,
        x_validation,
        y_validation,
    )


if __name__ == "__main__":
    main()
