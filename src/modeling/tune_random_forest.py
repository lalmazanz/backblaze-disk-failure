import duckdb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from src.config import (
    FEATURE_COLUMNS,
    NEGATIVE_RATIO,
    RANDOM_STATE,
    TOP_PCT,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)
from src.evaluation.policy import get_daily_alerts
from src.modeling.data import (
    load_period,
    load_undersampled_train,
)

CONFIGS = [
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 500,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": 0.5,
    },
    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": 0.5,
    },
]


def evaluate_config(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict,
) -> dict:
    model = RandomForestClassifier(
        **config,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        train[FEATURE_COLUMNS],
        train["failure_next_7d"],
    )

    scored = validation.copy()

    scored["risk_score"] = model.predict_proba(scored[FEATURE_COLUMNS])[:, 1]

    scored["drive_id"] = scored["model"] + ":" + scored["serial_number"]

    y_true = scored["failure_next_7d"]

    roc_auc = roc_auc_score(
        y_true,
        scored["risk_score"],
    )

    pr_auc = average_precision_score(
        y_true,
        scored["risk_score"],
    )

    positive_drives = set(
        scored.loc[
            scored["failure_next_7d"] == 1,
            "drive_id",
        ]
    )

    alerts = get_daily_alerts(
        scored,
        top_pct=TOP_PCT,
    )

    detected_drives = set(
        alerts.loc[
            alerts["failure_next_7d"] == 1,
            "drive_id",
        ]
    )

    drive_recall = (
        len(detected_drives) / len(positive_drives) if positive_drives else 0.0
    )

    return {
        **config,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "positive_drives": len(positive_drives),
        "detected_drives": len(detected_drives),
        "top_1pct_drive_recall": (drive_recall),
    }


def main() -> None:
    con = duckdb.connect()

    print("Loading purged Random Forest training sample...")

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
    )

    print(f"Train: {TRAIN_START} -> {TRAIN_END}")
    print(f"Validation: {VALIDATION_START} -> {VALIDATION_END}")
    print(f"Training rows: {len(train)}")
    print(f"Training positives: {int(train['failure_next_7d'].sum())}")
    print(f"Configurations: {len(CONFIGS)}")

    results = []

    for index, config in enumerate(
        CONFIGS,
        start=1,
    ):
        print(f"\n[{index}/{len(CONFIGS)}] {config}")

        result = evaluate_config(
            train,
            validation,
            config,
        )

        results.append(result)

        print(f"ROC-AUC: {result['roc_auc']:.4f}")
        print(f"PR-AUC: {result['pr_auc']:.4f}")
        print(f"Detected drives: {result['detected_drives']}")
        print(f"Top 1% drive recall: {result['top_1pct_drive_recall']:.4f}")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        [
            "top_1pct_drive_recall",
            "pr_auc",
            "roc_auc",
        ],
        ascending=False,
    ).reset_index(drop=True)

    print("\nRANDOM FOREST TUNING RESULTS")

    print(results_df.to_string(index=False))

    best = results_df.iloc[0]

    print("\nSELECTED CONFIGURATION")
    print(f"n_estimators: {best['n_estimators']}")
    print(f"max_depth: {best['max_depth']}")
    print(f"min_samples_leaf: {best['min_samples_leaf']}")
    print(f"max_features: {best['max_features']}")
    print(f"Top 1% drive recall: {best['top_1pct_drive_recall']:.4f}")
    print(f"PR-AUC: {best['pr_auc']:.4f}")
    print(f"ROC-AUC: {best['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
