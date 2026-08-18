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

FEATURES_PATH = Path("data/processed/q1_2026_features.parquet")
SOURCE_PATH = Path("data/interim/q1_2026_selected_models.parquet")

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

TEST_START = "2026-03-11"
TEST_END = "2026-03-24"

NEGATIVE_RATIO = 50
TOP_PCT = 0.01
RANDOM_STATE = 42


def load_undersampled_train(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    columns = ", ".join(FEATURE_COLUMNS)

    positive_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{FEATURES_PATH}')
        WHERE date BETWEEN DATE '{TRAIN_START}' AND DATE '{TRAIN_END}'
          AND failure_next_7d = 1;
        """
    ).fetchone()[0]

    negative_sample_size = positive_count * NEGATIVE_RATIO

    print(f"Train positives: {positive_count}")
    print(f"Sampled train negatives: {negative_sample_size}")

    query = f"""
    WITH positives AS (
        SELECT
            {columns},
            failure_next_7d
        FROM read_parquet('{FEATURES_PATH}')
        WHERE date BETWEEN DATE '{TRAIN_START}' AND DATE '{TRAIN_END}'
          AND failure_next_7d = 1
    ),

    negatives AS (
        SELECT
            {columns},
            failure_next_7d
        FROM read_parquet('{FEATURES_PATH}')
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
    SELECT * FROM negatives;
    """

    return con.execute(query).fetchdf()


def load_test(
    con: duckdb.DuckDBPyConnection,
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
    WHERE date BETWEEN DATE '{TEST_START}' AND DATE '{TEST_END}';
    """

    return con.execute(query).fetchdf()


def load_failure_dates(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT
            model || ':' || serial_number AS drive_id,
            MIN(date) AS failure_date
        FROM read_parquet('{SOURCE_PATH}')
        WHERE failure = 1
        GROUP BY model, serial_number;
        """
    ).fetchdf()


def main() -> None:
    con = duckdb.connect()

    print("Building final 1:50 training sample...")
    train = load_undersampled_train(con)

    print("Loading untouched test period...")
    test = load_test(con)

    x_train = train[FEATURE_COLUMNS]
    y_train = train["failure_next_7d"]

    x_test = test[FEATURE_COLUMNS]
    y_test = test["failure_next_7d"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training final Random Forest...")
    model.fit(x_train, y_train)

    test = test.copy()

    test["risk_score"] = model.predict_proba(x_test)[:, 1]

    test["prediction_05"] = (test["risk_score"] >= 0.5).astype(int)

    test["drive_id"] = test["model"] + ":" + test["serial_number"]

    print("\nRANDOM FOREST FINAL TEST — row-level metrics")
    print(f"ROC-AUC:   {roc_auc_score(y_test, test['risk_score']):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_test, test['risk_score']):.4f}")
    print(
        f"Precision: "
        f"{precision_score(y_test, test['prediction_05'], zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(y_test, test['prediction_05'], zero_division=0):.4f}"
    )

    positive_drives = set(
        test.loc[
            test["failure_next_7d"] == 1,
            "drive_id",
        ]
    )

    daily_alerts = []

    for _, day_data in test.groupby("date"):
        n_alerts = max(
            1,
            int(len(day_data) * TOP_PCT),
        )

        alerts = day_data.nlargest(
            n_alerts,
            "risk_score",
        ).copy()

        daily_alerts.append(alerts)

    alerts = pd.concat(
        daily_alerts,
        ignore_index=True,
    )

    positive_alerts = alerts[alerts["failure_next_7d"] == 1].copy()

    detected_positive_drives = set(positive_alerts["drive_id"])

    drive_recall = len(detected_positive_drives) / len(positive_drives)

    print("\nRANDOM FOREST FINAL TEST — operational metrics")
    print(f"Daily inspection budget: {TOP_PCT * 100:.1f}%")
    print(f"Alert rows: {len(alerts)}")
    print(f"Unique alerted drives: {alerts['drive_id'].nunique()}")
    print(f"Positive drives: {len(positive_drives)}")
    print(f"Detected positive drives: {len(detected_positive_drives)}")
    print(f"Drive recall: {drive_recall:.4f}")

    failure_dates = load_failure_dates(con)

    first_alerts = (
        positive_alerts.groupby("drive_id")["date"]
        .min()
        .reset_index(name="first_alert_date")
    )

    lead_time = first_alerts.merge(
        failure_dates,
        on="drive_id",
        how="inner",
    )

    lead_time["lead_days"] = (
        lead_time["failure_date"] - lead_time["first_alert_date"]
    ).dt.days

    lead_time = lead_time[
        lead_time["lead_days"].between(
            1,
            7,
            inclusive="both",
        )
    ].copy()

    print("\nRANDOM FOREST FINAL TEST — lead time")

    if lead_time.empty:
        print("No valid lead-time observations.")
        return

    print(lead_time["lead_days"].describe().to_string())

    distribution = (
        lead_time["lead_days"]
        .value_counts()
        .sort_index()
        .rename_axis("lead_days")
        .reset_index(name="drives")
    )

    print("\nLead-time distribution:")
    print(distribution.to_string(index=False))


if __name__ == "__main__":
    main()
