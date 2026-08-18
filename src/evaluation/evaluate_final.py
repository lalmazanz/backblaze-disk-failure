import duckdb
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    FEATURE_COLUMNS,
    FINAL_TRAIN_END,
    FINAL_TRAIN_START,
    LIGHTGBM_PARAMS,
    TEST_END,
    TEST_START,
    TOP_PCT,
)
from src.modeling.data import (
    load_failure_dates,
    load_period,
    load_undersampled_train,
)


def build_daily_alerts(
    test: pd.DataFrame,
) -> pd.DataFrame:
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

    return pd.concat(
        daily_alerts,
        ignore_index=True,
    )


def evaluate_row_level(
    test: pd.DataFrame,
) -> None:
    y_test = test["failure_next_7d"]

    print("\nFINAL TEST — row-level metrics")

    print(f"ROC-AUC:   {roc_auc_score(y_test, test['risk_score']):.4f}")

    print(f"PR-AUC:    {average_precision_score(y_test, test['risk_score']):.4f}")

    print(
        "Precision: "
        f"{
            precision_score(
                y_test,
                test['prediction_05'],
                zero_division=0,
            ):.4f}"
    )

    print(
        "Recall:    "
        f"{
            recall_score(
                y_test,
                test['prediction_05'],
                zero_division=0,
            ):.4f}"
    )


def evaluate_operational(
    test: pd.DataFrame,
    alerts: pd.DataFrame,
) -> set[str]:
    positive_drives = set(
        test.loc[
            test["failure_next_7d"] == 1,
            "drive_id",
        ]
    )

    positive_alerts = alerts[alerts["failure_next_7d"] == 1].copy()

    detected_positive_drives = set(positive_alerts["drive_id"])

    drive_recall = len(detected_positive_drives) / len(positive_drives)

    print("\nFINAL TEST — operational metrics")
    print(f"Daily inspection budget: {TOP_PCT * 100:.1f}%")
    print(f"Alert rows: {len(alerts)}")
    print(f"Unique alerted drives: {alerts['drive_id'].nunique()}")
    print(f"Positive drives: {len(positive_drives)}")
    print(f"Detected positive drives: {len(detected_positive_drives)}")
    print(f"Drive recall: {drive_recall:.4f}")

    return detected_positive_drives


def evaluate_lead_time(
    con: duckdb.DuckDBPyConnection,
    alerts: pd.DataFrame,
) -> None:
    positive_alerts = alerts[alerts["failure_next_7d"] == 1].copy()

    first_alerts = (
        positive_alerts.groupby("drive_id")["date"]
        .min()
        .reset_index(name="first_alert_date")
    )

    failure_dates = load_failure_dates(con)

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

    print("\nFINAL TEST — lead time")

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


def main() -> None:
    con = duckdb.connect()

    print("Building final 1:50 training sample...")
    train = load_undersampled_train(
        con,
        FINAL_TRAIN_START,
        FINAL_TRAIN_END,
    )

    print("Loading untouched test period...")
    test = load_period(
        con,
        TEST_START,
        TEST_END,
    )

    x_train = train[FEATURE_COLUMNS]
    y_train = train["failure_next_7d"]

    model = LGBMClassifier(
        **LIGHTGBM_PARAMS,
    )

    print("Training final LightGBM...")
    model.fit(
        x_train,
        y_train,
    )

    test = test.copy()

    test["risk_score"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    test["prediction_05"] = (test["risk_score"] >= 0.5).astype(int)

    test["drive_id"] = test["model"] + ":" + test["serial_number"]

    evaluate_row_level(test)

    alerts = build_daily_alerts(test)

    evaluate_operational(
        test,
        alerts,
    )

    evaluate_lead_time(
        con,
        alerts,
    )


if __name__ == "__main__":
    main()
