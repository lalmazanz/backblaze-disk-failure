from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import shap
from lightgbm import LGBMClassifier

FEATURES_PATH = Path("data/processed/q1_2026_features.parquet")

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

TRAIN_START = "2026-01-01"
TRAIN_END = "2026-03-10"

TEST_START = "2026-03-11"
TEST_END = "2026-03-24"

NEGATIVE_RATIO = 50
SHAP_SAMPLE_SIZE = 5000
RANDOM_STATE = 42


def load_undersampled_train(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    columns = ", ".join(FEATURE_COLUMNS)

    positive_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{FEATURES_PATH}')
        WHERE date BETWEEN DATE '{TRAIN_START}' AND DATE '{TRAIN_END}'
          AND failure_next_7d = 1
        """
    ).fetchone()[0]

    negative_sample_size = positive_count * NEGATIVE_RATIO

    query = f"""
    WITH positives AS (
        SELECT
            {columns},
            failure_next_7d
        FROM read_parquet('{FEATURES_PATH}')
        WHERE date BETWEEN DATE '{TRAIN_START}' AND DATE '{TRAIN_END}'
          AND failure_next_7d = 1
    ),

    negatives AS (
        SELECT
            {columns},
            failure_next_7d
        FROM read_parquet('{FEATURES_PATH}')
        WHERE date BETWEEN DATE '{TRAIN_START}' AND DATE '{TRAIN_END}'
          AND failure_next_7d = 0
        ORDER BY HASH(
            model,
            serial_number,
            date,
            {RANDOM_STATE}
        )
        LIMIT {negative_sample_size}
    )

    SELECT * FROM positives
    UNION ALL
    SELECT * FROM negatives
    """

    return con.execute(query).fetchdf()


def load_test_sample(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    columns = ", ".join(FEATURE_COLUMNS)

    query = f"""
    SELECT
        {columns},
        failure_next_7d
    FROM read_parquet('{FEATURES_PATH}')
    WHERE date BETWEEN DATE '{TEST_START}' AND DATE '{TEST_END}'
    ORDER BY HASH(
        model,
        serial_number,
        date,
        {RANDOM_STATE}
    )
    LIMIT {SHAP_SAMPLE_SIZE}
    """

    return con.execute(query).fetchdf()


def main() -> None:
    con = duckdb.connect()

    print("Loading training sample...")
    train = load_undersampled_train(con)

    print("Loading SHAP test sample...")
    test_sample = load_test_sample(con)

    x_train = train[FEATURE_COLUMNS]
    y_train = train["failure_next_7d"]

    x_sample = test_sample[FEATURE_COLUMNS]

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

    print("Training LightGBM...")
    model.fit(
        x_train,
        y_train,
    )

    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x_sample)

    mean_abs_shap = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "mean_abs_shap": abs(shap_values.values).mean(axis=0),
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    print("\nTop 15 SHAP features:")
    print(mean_abs_shap.head(15).to_string(index=False))

    ranking_path = OUTPUT_DIR / "lightgbm_shap_importance.csv"

    mean_abs_shap.to_csv(
        ranking_path,
        index=False,
    )

    shap.plots.beeswarm(
        shap_values,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    beeswarm_path = OUTPUT_DIR / "lightgbm_shap_beeswarm.png"

    plt.savefig(
        beeswarm_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nSaved ranking to: {ranking_path}")
    print(f"Saved beeswarm to: {beeswarm_path}")


if __name__ == "__main__":
    main()
