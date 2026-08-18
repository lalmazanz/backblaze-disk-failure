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
    NEGATIVE_RATIO,
    TEST_END,
    TEST_START,
    TOP_PCT,
)
from src.evaluation.policy import (
    get_daily_alerts,
)
from src.logging_utils import get_logger
from src.modeling.data import (
    load_failure_dates,
    load_period,
    load_undersampled_train,
)

logger = get_logger(__name__)


def evaluate_row_level(
    y_true: pd.Series,
    probabilities: pd.Series,
) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": (
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }


def evaluate_operational(
    test: pd.DataFrame,
) -> dict[str, float | int]:
    alerts = get_daily_alerts(
        test,
        top_pct=TOP_PCT,
    )

    positive_drives = test.loc[
        test["failure_next_7d"] == 1,
        "drive_id",
    ].nunique()

    positive_alerts = alerts[alerts["failure_next_7d"] == 1]

    detected_positive_drives = positive_alerts["drive_id"].nunique()

    drive_recall = (
        detected_positive_drives / positive_drives if positive_drives else 0.0
    )

    return {
        "alert_rows": len(alerts),
        "unique_alerted_drives": (alerts["drive_id"].nunique()),
        "positive_drives": (positive_drives),
        "detected_positive_drives": (detected_positive_drives),
        "drive_recall": drive_recall,
    }


def evaluate_lead_time(
    test: pd.DataFrame,
    failure_dates: pd.DataFrame,
) -> pd.Series:
    alerts = get_daily_alerts(
        test,
        top_pct=TOP_PCT,
    )

    positive_alerts = alerts[alerts["failure_next_7d"] == 1].copy()

    positive_alerts = positive_alerts.merge(
        failure_dates,
        on="drive_id",
        how="left",
    )

    positive_alerts["lead_days"] = (
        positive_alerts["failure_date"] - positive_alerts["date"]
    ).dt.days

    lead_times = (
        positive_alerts.groupby("drive_id")["lead_days"].max().dropna().astype(int)
    )

    return lead_times


def main() -> None:
    con = duckdb.connect()

    logger.info(
        "Building final 1:%s training sample...",
        NEGATIVE_RATIO,
    )

    train = load_undersampled_train(
        con,
        FINAL_TRAIN_START,
        FINAL_TRAIN_END,
        negative_ratio=NEGATIVE_RATIO,
    )

    logger.info("Loading final test period...")

    test = load_period(
        con,
        TEST_START,
        TEST_END,
    ).copy()

    logger.info("Training final LightGBM...")

    model = LGBMClassifier(**LIGHTGBM_PARAMS)

    model.fit(
        train[FEATURE_COLUMNS],
        train["failure_next_7d"],
    )

    logger.info("Scoring final test period...")

    test["risk_score"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    test["drive_id"] = test["model"] + ":" + test["serial_number"]

    row_metrics = evaluate_row_level(
        test["failure_next_7d"],
        test["risk_score"],
    )

    operational_metrics = evaluate_operational(test)

    logger.info("Loading failure dates for lead-time evaluation...")

    failure_dates = load_failure_dates(con)

    lead_times = evaluate_lead_time(
        test,
        failure_dates,
    )

    logger.info("Final evaluation completed.")

    print("\nFINAL TEST — row-level metrics")
    print(f"ROC-AUC:   {row_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {row_metrics['pr_auc']:.4f}")
    print(f"Precision: {row_metrics['precision']:.4f}")
    print(f"Recall:    {row_metrics['recall']:.4f}")

    print("\nFINAL TEST — operational metrics")
    print(f"Daily inspection budget: {TOP_PCT * 100:.1f}%")
    print(f"Alert rows: {operational_metrics['alert_rows']}")
    print(f"Unique alerted drives: {operational_metrics['unique_alerted_drives']}")
    print(f"Positive drives: {operational_metrics['positive_drives']}")
    print(
        f"Detected positive drives: {operational_metrics['detected_positive_drives']}"
    )
    print(f"Drive recall: {operational_metrics['drive_recall']:.4f}")

    print("\nFINAL TEST — lead time")

    print(lead_times.describe())

    distribution = (
        lead_times.value_counts()
        .sort_index()
        .rename_axis("lead_days")
        .reset_index(name="drives")
    )

    print("\nLead-time distribution:")
    print(distribution.to_string(index=False))


if __name__ == "__main__":
    main()
