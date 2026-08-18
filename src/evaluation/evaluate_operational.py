from pathlib import Path

import duckdb
import pandas as pd
from lightgbm import LGBMClassifier

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

TOP_PCTS = [0.001, 0.005, 0.01, 0.02]
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
        date,
        serial_number,
        model,
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

    sampled_train = pd.concat(
        [
            positives,
            sampled_negatives,
        ],
        ignore_index=True,
    )

    return sampled_train.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)


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

    model = LGBMClassifier(
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

    print("Training LightGBM 1:50...")
    model.fit(
        x_train,
        y_train,
    )

    validation = validation.copy()

    validation["risk_score"] = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]

    validation["drive_id"] = validation["model"] + ":" + validation["serial_number"]

    positive_drives = set(
        validation.loc[
            validation["failure_next_7d"] == 1,
            "drive_id",
        ]
    )

    print(f"\nPositive drives in validation: {len(positive_drives)}")

    results = []

    for top_pct in TOP_PCTS:
        all_alerted_drives = set()
        positive_alerted_drives = set()
        total_alert_rows = 0

        for _, day_data in validation.groupby("date"):
            n_alerts = max(
                1,
                int(len(day_data) * top_pct),
            )

            daily_alerts = day_data.nlargest(
                n_alerts,
                "risk_score",
            )

            all_alerted_drives.update(daily_alerts["drive_id"])

            positive_daily_alerts = daily_alerts[daily_alerts["failure_next_7d"] == 1]

            positive_alerted_drives.update(positive_daily_alerts["drive_id"])

            total_alert_rows += len(daily_alerts)

        detected_positive_drives = positive_alerted_drives

        recall_by_drive = len(detected_positive_drives) / len(positive_drives)

        results.append(
            {
                "top_pct": top_pct * 100,
                "alert_rows": total_alert_rows,
                "unique_alerted_drives": len(all_alerted_drives),
                "detected_positive_drives": len(detected_positive_drives),
                "total_positive_drives": len(positive_drives),
                "drive_recall": recall_by_drive,
            }
        )

    results_df = pd.DataFrame(results)

    print("\nOperational validation results:")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()
