import duckdb
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    FEATURE_COLUMNS,
    LIGHTGBM_PARAMS,
    NEGATIVE_RATIO,
    TOP_PCT,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)
from src.evaluation.policy import (
    get_daily_alerts,
)
from src.modeling.data import (
    load_period,
    load_undersampled_train,
)


def main() -> None:
    con = duckdb.connect()

    print(f"Loading 1:{NEGATIVE_RATIO} purged training sample...")

    train = load_undersampled_train(
        con,
        TRAIN_START,
        TRAIN_END,
        negative_ratio=NEGATIVE_RATIO,
    )

    print("Loading validation period...")

    validation = load_period(
        con,
        VALIDATION_START,
        VALIDATION_END,
    ).copy()

    x_train = train[FEATURE_COLUMNS]
    y_train = train["failure_next_7d"]

    x_val = validation[FEATURE_COLUMNS]
    y_val = validation["failure_next_7d"]

    print(f"Training rows: {len(train)}")
    print(f"Training positives: {int(y_train.sum())}")
    print(f"Training negatives: {len(train) - int(y_train.sum())}")
    print(f"Validation rows: {len(validation)}")
    print(f"Validation positives: {int(y_val.sum())}")

    model = LGBMClassifier(**LIGHTGBM_PARAMS)

    print("Training LightGBM...")

    model.fit(
        x_train,
        y_train,
    )

    validation["risk_score"] = model.predict_proba(x_val)[:, 1]

    validation["drive_id"] = validation["model"] + ":" + validation["serial_number"]

    predictions = (validation["risk_score"] >= 0.5).astype(int)

    roc_auc = roc_auc_score(
        y_val,
        validation["risk_score"],
    )

    pr_auc = average_precision_score(
        y_val,
        validation["risk_score"],
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )

    positive_drives = set(
        validation.loc[
            validation["failure_next_7d"] == 1,
            "drive_id",
        ]
    )

    alerts = get_daily_alerts(
        validation,
        top_pct=TOP_PCT,
    )

    positive_alerts = alerts[alerts["failure_next_7d"] == 1]

    detected_drives = set(positive_alerts["drive_id"])

    operational_recall = (
        len(detected_drives) / len(positive_drives) if positive_drives else 0.0
    )

    print(f"\nVALIDATION — LightGBM 1:{NEGATIVE_RATIO}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    print(f"\nOperational top {TOP_PCT * 100:.1f}%")
    print(f"Positive drives: {len(positive_drives)}")
    print(f"Detected drives: {len(detected_drives)}")
    print(f"Drive recall: {operational_recall:.4f}")


if __name__ == "__main__":
    main()
