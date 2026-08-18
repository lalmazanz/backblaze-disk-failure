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

NEGATIVE_RATIO = 50
TOP_PCT = 0.01
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


def build_undersampled_train(train: pd.DataFrame) -> pd.DataFrame:
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
    model.fit(x_train, y_train)

    validation = validation.copy()

    validation["risk_score"] = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]

    validation["drive_id"] = validation["model"] + ":" + validation["serial_number"]

    positive_rows = validation[validation["failure_next_7d"] == 1].copy()

    failure_dates = (
        positive_rows.groupby("drive_id")["date"]
        .max()
        .add(pd.Timedelta(days=1))
        .rename("failure_date")
    )

    alerts = []

    for _, day_data in validation.groupby("date"):
        n_alerts = max(
            1,
            int(len(day_data) * TOP_PCT),
        )

        daily_alerts = day_data.nlargest(
            n_alerts,
            "risk_score",
        )

        alerts.append(daily_alerts[["date", "drive_id", "failure_next_7d"]])

    alerts_df = pd.concat(
        alerts,
        ignore_index=True,
    )

    positive_alerts = alerts_df[alerts_df["failure_next_7d"] == 1].copy()

    first_alerts = (
        positive_alerts.groupby("drive_id")["date"].min().rename("first_alert_date")
    )

    lead_time = pd.concat(
        [failure_dates, first_alerts],
        axis=1,
        join="inner",
    )

    lead_time["lead_days"] = (
        lead_time["failure_date"] - lead_time["first_alert_date"]
    ).dt.days

    total_positive_drives = len(failure_dates)
    detected_drives = len(lead_time)

    print(f"\nTop daily percentage: {TOP_PCT * 100:.1f}%")
    print(f"Positive drives: {total_positive_drives}")
    print(f"Detected drives: {detected_drives}")
    print(f"Drive recall: {detected_drives / total_positive_drives:.4f}")

    print("\nLead-time summary:")
    print(lead_time["lead_days"].describe().to_string())

    print("\nLead-time distribution:")
    distribution = (
        lead_time["lead_days"]
        .value_counts()
        .sort_index()
        .rename_axis("lead_days")
        .reset_index(name="drives")
    )

    print(distribution.to_string(index=False))


if __name__ == "__main__":
    main()
