import duckdb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_COLUMNS,
    FINAL_TRAIN_END,
    FINAL_TRAIN_START,
    NEGATIVE_RATIO,
    RANDOM_FOREST_PARAMS,
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
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
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
) -> tuple[
    dict[str, float | int],
    pd.DataFrame,
]:
    alerts = get_daily_alerts(
        test,
        top_pct=TOP_PCT,
    )

    positive_drives = set(
        test.loc[
            test["failure_next_7d"] == 1,
            "drive_id",
        ]
    )

    positive_alerts = alerts[alerts["failure_next_7d"] == 1].copy()

    detected_drives = set(positive_alerts["drive_id"])

    drive_recall = (
        len(detected_drives) / len(positive_drives) if positive_drives else 0.0
    )

    metrics = {
        "alert_rows": len(alerts),
        "unique_alerted_drives": (alerts["drive_id"].nunique()),
        "positive_drives": (len(positive_drives)),
        "detected_positive_drives": (len(detected_drives)),
        "drive_recall": drive_recall,
    }

    return metrics, positive_alerts


def evaluate_lead_time(
    positive_alerts: pd.DataFrame,
    failure_dates: pd.DataFrame,
) -> pd.Series:
    alerts_with_failure = positive_alerts.merge(
        failure_dates,
        on="drive_id",
        how="inner",
    )

    alerts_with_failure["lead_days"] = (
        alerts_with_failure["failure_date"] - alerts_with_failure["date"]
    ).dt.days

    valid_alerts = alerts_with_failure[
        alerts_with_failure["lead_days"].between(
            1,
            7,
            inclusive="both",
        )
    ].copy()

    lead_times = (
        valid_alerts.groupby("drive_id")["lead_days"].max().dropna().astype(int)
    )

    return lead_times


def main() -> None:
    con = duckdb.connect()

    logger.info(
        "Building final 1:%s purged training sample...",
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

    x_train = train[FEATURE_COLUMNS]
    y_train = train["failure_next_7d"]

    x_test = test[FEATURE_COLUMNS]
    y_test = test["failure_next_7d"]

    print(f"Final train: {FINAL_TRAIN_START} -> {FINAL_TRAIN_END}")
    print(f"Test:        {TEST_START} -> {TEST_END}")
    print(f"Training rows: {len(train)}")
    print(f"Training positives: {int(y_train.sum())}")
    print(f"Training negatives: {len(train) - int(y_train.sum())}")
    print(f"Test rows: {len(test)}")
    print(f"Test positives: {int(y_test.sum())}")

    logger.info("Training final Random Forest...")

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(**RANDOM_FOREST_PARAMS),
            ),
        ]
    )

    model.fit(
        x_train,
        y_train,
    )

    logger.info("Scoring final test period...")

    test["risk_score"] = model.predict_proba(x_test)[:, 1]

    test["drive_id"] = test["model"] + ":" + test["serial_number"]

    row_metrics = evaluate_row_level(
        y_test,
        test["risk_score"],
    )

    (
        operational_metrics,
        positive_alerts,
    ) = evaluate_operational(test)

    logger.info("Loading failure dates for lead-time evaluation...")

    failure_dates = load_failure_dates(con)

    lead_times = evaluate_lead_time(
        positive_alerts,
        failure_dates,
    )

    logger.info("Final Random Forest evaluation completed.")

    print("\nRANDOM FOREST FINAL TEST — row-level metrics")
    print(f"ROC-AUC:   {row_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {row_metrics['pr_auc']:.4f}")
    print(f"Precision: {row_metrics['precision']:.4f}")
    print(f"Recall:    {row_metrics['recall']:.4f}")

    print("\nRANDOM FOREST FINAL TEST — operational metrics")
    print(f"Daily inspection budget: {TOP_PCT * 100:.1f}%")
    print(f"Alert rows: {operational_metrics['alert_rows']}")
    print(f"Unique alerted drives: {operational_metrics['unique_alerted_drives']}")
    print(f"Positive drives: {operational_metrics['positive_drives']}")
    print(
        f"Detected positive drives: {operational_metrics['detected_positive_drives']}"
    )
    print(f"Drive recall: {operational_metrics['drive_recall']:.4f}")

    print("\nRANDOM FOREST FINAL TEST — lead time")

    if lead_times.empty:
        print("No valid lead-time observations.")
        return

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
