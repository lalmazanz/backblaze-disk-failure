import duckdb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_COLUMNS,
    FINAL_TRAIN_END,
    FINAL_TRAIN_START,
    NEGATIVE_RATIO,
    RANDOM_FOREST_PARAMS,
    REPORTS_DIR,
    SELECTED_MODELS,
    TEST_END,
    TEST_START,
    TOP_PCT,
)
from src.evaluation.policy import add_daily_alert_policy
from src.modeling.data import (
    load_period,
    load_undersampled_train,
)


def build_model() -> Pipeline:
    return Pipeline(
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


def calculate_operational_recall(
    data: pd.DataFrame,
) -> tuple[int, int, float]:
    positive_drives = set(
        data.loc[
            data["failure_next_7d"] == 1,
            "drive_id",
        ]
    )

    detected_drives = set(
        data.loc[
            (data["failure_next_7d"] == 1) & data["top_1pct_alert"],
            "drive_id",
        ]
    )

    recall = len(detected_drives) / len(positive_drives) if positive_drives else 0.0

    return (
        len(positive_drives),
        len(detected_drives),
        recall,
    )


def evaluate_by_model(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model_name in SELECTED_MODELS:
        subset = test[test["model"] == model_name].copy()

        if subset.empty:
            continue

        y_true = subset["failure_next_7d"]

        scores = subset["risk_score"]

        positive_rows = int(y_true.sum())

        roc_auc = (
            roc_auc_score(
                y_true,
                scores,
            )
            if y_true.nunique() > 1
            else float("nan")
        )

        pr_auc = (
            average_precision_score(
                y_true,
                scores,
            )
            if positive_rows > 0
            else float("nan")
        )

        (
            positive_drives,
            detected_drives,
            drive_recall,
        ) = calculate_operational_recall(subset)

        alert_rows = int(subset["top_1pct_alert"].sum())

        rows.append(
            {
                "model": model_name,
                "rows": len(subset),
                "positive_rows": (positive_rows),
                "alert_rows": alert_rows,
                "positive_drives": (positive_drives),
                "detected_drives": (detected_drives),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "global_top_1pct_drive_recall": (drive_recall),
            }
        )

    return pd.DataFrame(rows)


def evaluate_by_date(
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for date, subset in test.groupby("date"):
        y_true = subset["failure_next_7d"]

        scores = subset["risk_score"]

        positive_rows = int(y_true.sum())

        roc_auc = (
            roc_auc_score(
                y_true,
                scores,
            )
            if y_true.nunique() > 1
            else float("nan")
        )

        pr_auc = (
            average_precision_score(
                y_true,
                scores,
            )
            if positive_rows > 0
            else float("nan")
        )

        alerts = subset[subset["top_1pct_alert"]]

        positive_alert_rows = int(alerts["failure_next_7d"].sum())

        rows.append(
            {
                "date": date,
                "rows": len(subset),
                "positive_rows": (positive_rows),
                "alert_rows": len(alerts),
                "positive_alert_rows": (positive_alert_rows),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    con = duckdb.connect()

    train = load_undersampled_train(
        con,
        FINAL_TRAIN_START,
        FINAL_TRAIN_END,
        negative_ratio=NEGATIVE_RATIO,
    )

    test = load_period(
        con,
        TEST_START,
        TEST_END,
    ).copy()

    x_train = train[FEATURE_COLUMNS]

    y_train = train["failure_next_7d"]

    x_test = test[FEATURE_COLUMNS]

    model = build_model()

    print("Training final Random Forest pipeline...")

    model.fit(
        x_train,
        y_train,
    )

    test["risk_score"] = model.predict_proba(x_test)[:, 1]

    test["drive_id"] = test["model"] + ":" + test["serial_number"]

    # Apply the operational policy once,
    # globally across all drive models
    # for each observation date.
    test = add_daily_alert_policy(
        test,
        top_pct=TOP_PCT,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nPER-MODEL EVALUATION (GLOBAL TOP 1% POLICY)")

    model_results = evaluate_by_model(test)

    model_output_path = REPORTS_DIR / "model_breakdown.csv"

    model_results.to_csv(
        model_output_path,
        index=False,
    )

    print(model_results.to_string(index=False))

    print("\nPER-DATE EVALUATION")

    date_results = evaluate_by_date(test)

    date_output_path = REPORTS_DIR / "daily_breakdown.csv"

    date_results.to_csv(
        date_output_path,
        index=False,
    )

    print(date_results.to_string(index=False))

    print("\nPER-DATE SUMMARY")

    print(
        date_results[
            [
                "roc_auc",
                "pr_auc",
            ]
        ].agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )

    print(f"\nModel breakdown saved to: {model_output_path}")

    print(f"Daily breakdown saved to: {date_output_path}")


if __name__ == "__main__":
    main()
