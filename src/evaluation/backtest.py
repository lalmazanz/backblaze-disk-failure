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
    RANDOM_FOREST_PARAMS,
    TOP_PCT,
)
from src.evaluation.policy import get_daily_alerts
from src.modeling.data import (
    load_period,
    load_undersampled_train,
)

BACKTEST_WINDOWS = [
    {
        "name": "Backtest 1",
        "train_start": "2026-01-01",
        "train_end": "2026-01-24",
        "eval_start": "2026-02-01",
        "eval_end": "2026-02-14",
    },
    {
        "name": "Backtest 2",
        "train_start": "2026-01-01",
        "train_end": "2026-02-07",
        "eval_start": "2026-02-15",
        "eval_end": "2026-02-28",
    },
    {
        "name": "Backtest 3",
        "train_start": "2026-01-01",
        "train_end": "2026-02-21",
        "eval_start": "2026-03-01",
        "eval_end": "2026-03-10",
    },
]


def evaluate_window(
    con: duckdb.DuckDBPyConnection,
    window: dict,
) -> dict:
    print(f"\n{window['name']}")
    print(f"Train: {window['train_start']} -> {window['train_end']}")
    print(f"Eval:  {window['eval_start']} -> {window['eval_end']}")

    train = load_undersampled_train(
        con,
        window["train_start"],
        window["train_end"],
        negative_ratio=NEGATIVE_RATIO,
    )

    evaluation = load_period(
        con,
        window["eval_start"],
        window["eval_end"],
    ).copy()

    x_train = train[FEATURE_COLUMNS]
    y_train = train["failure_next_7d"]

    x_eval = evaluation[FEATURE_COLUMNS]
    y_eval = evaluation["failure_next_7d"]

    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)

    model.fit(
        x_train,
        y_train,
    )

    evaluation["risk_score"] = model.predict_proba(x_eval)[:, 1]

    evaluation["drive_id"] = evaluation["model"] + ":" + evaluation["serial_number"]

    roc_auc = roc_auc_score(
        y_eval,
        evaluation["risk_score"],
    )

    pr_auc = average_precision_score(
        y_eval,
        evaluation["risk_score"],
    )

    positive_drives = set(
        evaluation.loc[
            evaluation["failure_next_7d"] == 1,
            "drive_id",
        ]
    )

    alerts = get_daily_alerts(
        evaluation,
        top_pct=TOP_PCT,
    )

    positive_alerts = alerts[alerts["failure_next_7d"] == 1]

    detected_drives = set(positive_alerts["drive_id"])

    operational_recall = (
        len(detected_drives) / len(positive_drives) if positive_drives else 0.0
    )

    print(f"Training rows: {len(train)}")
    print(f"Training positives: {int(y_train.sum())}")
    print(f"Eval rows: {len(evaluation)}")
    print(f"Positive rows: {int(y_eval.sum())}")
    print(f"Positive drives: {len(positive_drives)}")
    print(f"Detected drives: {len(detected_drives)}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Top 1% drive recall: {operational_recall:.4f}")

    return {
        "backtest": window["name"],
        "train_start": window["train_start"],
        "train_end": window["train_end"],
        "eval_start": window["eval_start"],
        "eval_end": window["eval_end"],
        "train_rows": len(train),
        "training_positive_rows": int(y_train.sum()),
        "eval_rows": len(evaluation),
        "positive_rows": int(y_eval.sum()),
        "positive_drives": len(positive_drives),
        "detected_drives": len(detected_drives),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "top_1pct_drive_recall": (operational_recall),
    }


def main() -> None:
    con = duckdb.connect()

    results = []

    print("Running purged Random Forest temporal backtests...")

    for window in BACKTEST_WINDOWS:
        result = evaluate_window(
            con,
            window,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("\nRANDOM FOREST BACKTEST SUMMARY")

    print(
        results_df[
            [
                "backtest",
                "train_end",
                "eval_start",
                "roc_auc",
                "pr_auc",
                "positive_drives",
                "detected_drives",
                "top_1pct_drive_recall",
            ]
        ].to_string(index=False)
    )

    print("\nAverage metrics:")
    print(f"ROC-AUC: {results_df['roc_auc'].mean():.4f}")
    print(f"PR-AUC: {results_df['pr_auc'].mean():.4f}")
    print(f"Top 1% drive recall: {results_df['top_1pct_drive_recall'].mean():.4f}")


if __name__ == "__main__":
    main()
