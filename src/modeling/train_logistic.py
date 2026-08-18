import duckdb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_COLUMNS,
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
)


def main() -> None:
    con = duckdb.connect()

    print("Loading purged train period...")

    train = load_period(
        con,
        TRAIN_START,
        TRAIN_END,
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
    print(f"Validation rows: {len(validation)}")
    print(f"Validation positives: {int(y_val.sum())}")

    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training Logistic Regression...")

    pipeline.fit(
        x_train,
        y_train,
    )

    probabilities = pipeline.predict_proba(x_val)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    roc_auc = roc_auc_score(
        y_val,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_val,
        probabilities,
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

    validation["risk_score"] = probabilities

    validation["drive_id"] = validation["model"] + ":" + validation["serial_number"]

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

    print("\nVALIDATION — Logistic Regression")
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
