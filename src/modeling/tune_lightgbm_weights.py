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

WEIGHTS = [1, 10, 50, 100, 500]


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
    weight: int,
    model: LGBMClassifier,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, float]:
    probabilities = model.predict_proba(x_validation)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "scale_pos_weight": weight,
        "roc_auc": roc_auc_score(
            y_validation,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_validation,
            probabilities,
        ),
        "precision": precision_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
    }


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

    results = []

    for weight in WEIGHTS:
        print(f"\nTraining scale_pos_weight={weight}...")

        model = LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=weight,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

        model.fit(
            x_train,
            y_train,
        )

        metrics = evaluate(
            weight,
            model,
            x_validation,
            y_validation,
        )

        results.append(metrics)

        print(
            f"ROC-AUC={metrics['roc_auc']:.4f} | "
            f"PR-AUC={metrics['pr_auc']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f}"
        )

    results_df = pd.DataFrame(results)

    print("\nValidation results:")
    print(
        results_df.sort_values(
            "pr_auc",
            ascending=False,
        ).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()
