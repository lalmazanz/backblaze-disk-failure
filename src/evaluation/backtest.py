import duckdb
import numpy as np
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
    NEGATIVE_RATIO,
    RANDOM_FOREST_PARAMS,
    RANDOM_STATE,
    REPORTS_DIR,
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


def calculate_drive_recall(
    data: pd.DataFrame,
    score_column: str,
) -> float:
    scored = data.copy()

    scored["risk_score"] = scored[score_column]

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

    positive_alerts = alerts[alerts["failure_next_7d"] == 1]

    detected_drives = set(positive_alerts["drive_id"])

    if not positive_drives:
        return 0.0

    return len(detected_drives) / len(positive_drives)


def add_baseline_scores(
    evaluation: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    result = evaluation.copy()

    rng = np.random.default_rng(seed)

    result["random_score"] = rng.random(len(result))

    smart_197 = result["smart_197_raw"].fillna(0)
    smart_198 = result["smart_198_raw"].fillna(0)

    result["smart_baseline_score"] = smart_197 + smart_198

    return result


def evaluate_window(
    con: duckdb.DuckDBPyConnection,
    window: dict,
    window_index: int,
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

    evaluation["model_score"] = model.predict_proba(x_eval)[:, 1]

    evaluation["drive_id"] = evaluation["model"] + ":" + evaluation["serial_number"]

    evaluation = add_baseline_scores(
        evaluation,
        seed=RANDOM_STATE + window_index,
    )

    roc_auc = roc_auc_score(
        y_eval,
        evaluation["model_score"],
    )

    pr_auc = average_precision_score(
        y_eval,
        evaluation["model_score"],
    )

    positive_drives = set(
        evaluation.loc[
            evaluation["failure_next_7d"] == 1,
            "drive_id",
        ]
    )

    model_recall = calculate_drive_recall(
        evaluation,
        "model_score",
    )

    random_recall = calculate_drive_recall(
        evaluation,
        "random_score",
    )

    smart_recall = calculate_drive_recall(
        evaluation,
        "smart_baseline_score",
    )

    print(f"Training rows: {len(train)}")
    print(f"Training positives: {int(y_train.sum())}")
    print(f"Eval rows: {len(evaluation)}")
    print(f"Positive rows: {int(y_eval.sum())}")
    print(f"Positive drives: {len(positive_drives)}")

    print("\nModel:")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Top 1% drive recall: {model_recall:.4f}")

    print("\nBaselines:")
    print(f"Random top 1% drive recall: {random_recall:.4f}")
    print(f"SMART 197/198 top 1% drive recall: {smart_recall:.4f}")

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
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "model_drive_recall": model_recall,
        "random_drive_recall": random_recall,
        "smart_drive_recall": smart_recall,
    }


def main() -> None:
    con = duckdb.connect()

    results = []

    print("Running purged Random Forest temporal backtests...")

    for index, window in enumerate(BACKTEST_WINDOWS):
        result = evaluate_window(
            con,
            window,
            window_index=index,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = REPORTS_DIR / "backtest_results.csv"

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(f"\nBacktest results saved to: {output_path}")
    print("\nRANDOM FOREST BACKTEST SUMMARY")

    print(
        results_df[
            [
                "backtest",
                "train_end",
                "eval_start",
                "roc_auc",
                "pr_auc",
                "model_drive_recall",
                "random_drive_recall",
                "smart_drive_recall",
            ]
        ].to_string(index=False)
    )

    print("\nMean metrics:")
    print(f"ROC-AUC: {results_df['roc_auc'].mean():.4f}")
    print(f"PR-AUC: {results_df['pr_auc'].mean():.4f}")
    print(f"Model top 1% drive recall: {results_df['model_drive_recall'].mean():.4f}")
    print(f"Random top 1% drive recall: {results_df['random_drive_recall'].mean():.4f}")
    print(f"SMART top 1% drive recall: {results_df['smart_drive_recall'].mean():.4f}")

    print("\nStandard deviation:")
    print(f"ROC-AUC: {results_df['roc_auc'].std():.4f}")
    print(f"PR-AUC: {results_df['pr_auc'].std():.4f}")
    print(f"Model top 1% drive recall: {results_df['model_drive_recall'].std():.4f}")
    print(f"Random top 1% drive recall: {results_df['random_drive_recall'].std():.4f}")
    print(f"SMART top 1% drive recall: {results_df['smart_drive_recall'].std():.4f}")


if __name__ == "__main__":
    main()
