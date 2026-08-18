from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import shap
from lightgbm import LGBMClassifier

FEATURES_PATH = Path("data/processed/q1_2026_features.parquet")
SOURCE_PATH = Path("data/interim/q1_2026_selected_models.parquet")

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
TOP_PCT = 0.01
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


def load_test(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    columns = ", ".join(FEATURE_COLUMNS)

    query = f"""
    SELECT
        date,
        serial_number,
        model,
        {columns},
        failure_next_7d
    FROM read_parquet('{FEATURES_PATH}')
    WHERE date BETWEEN DATE '{TEST_START}' AND DATE '{TEST_END}'
    """

    return con.execute(query).fetchdf()


def load_failure_dates(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT
            model || ':' || serial_number AS drive_id,
            MIN(date) AS failure_date
        FROM read_parquet('{SOURCE_PATH}')
        WHERE failure = 1
        GROUP BY model, serial_number
        """
    ).fetchdf()


def main() -> None:
    con = duckdb.connect()

    print("Loading training sample...")
    train = load_undersampled_train(con)

    print("Loading test...")
    test = load_test(con)

    x_train = train[FEATURE_COLUMNS]
    y_train = train["failure_next_7d"]

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
    model.fit(x_train, y_train)

    test = test.copy()

    test["risk_score"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    test["drive_id"] = test["model"] + ":" + test["serial_number"]

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

    alerts = pd.concat(
        daily_alerts,
        ignore_index=True,
    )

    positive_alerts = alerts[alerts["failure_next_7d"] == 1].copy()

    failure_dates = load_failure_dates(con)

    first_alerts = (
        positive_alerts.groupby("drive_id")["date"]
        .min()
        .reset_index(name="first_alert_date")
    )

    detected = first_alerts.merge(
        failure_dates,
        on="drive_id",
        how="inner",
    )

    detected["lead_days"] = (
        detected["failure_date"] - detected["first_alert_date"]
    ).dt.days

    candidates = detected[detected["lead_days"] == 7].copy()

    if candidates.empty:
        raise RuntimeError("No drive detected exactly 7 days before failure.")

    selected = candidates.iloc[0]

    drive_id = selected["drive_id"]
    alert_date = selected["first_alert_date"]
    failure_date = selected["failure_date"]

    print("\nSelected drive:")
    print(f"Drive: {drive_id}")
    print(f"First alert: {alert_date}")
    print(f"Failure date: {failure_date}")
    print("Lead time: 7 days")

    selected_row = test[
        (test["drive_id"] == drive_id) & (test["date"] == alert_date)
    ].copy()

    if selected_row.empty:
        raise RuntimeError("Could not find selected drive observation.")

    x_selected = selected_row[FEATURE_COLUMNS]

    risk_score = selected_row["risk_score"].iloc[0]

    print(f"Risk score: {risk_score:.4f}")

    print("Calculating local SHAP explanation...")

    explainer = shap.TreeExplainer(model)
    explanation = explainer(x_selected)

    local_values = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "value": x_selected.iloc[0].values,
            "shap_value": explanation.values[0],
        }
    )

    local_values["abs_shap"] = local_values["shap_value"].abs()

    local_values = local_values.sort_values(
        "abs_shap",
        ascending=False,
    )

    print("\nTop local contributors:")
    print(
        local_values.head(10)[
            [
                "feature",
                "value",
                "shap_value",
            ]
        ].to_string(index=False)
    )

    output_csv = OUTPUT_DIR / "lightgbm_detected_drive_shap.csv"

    local_values.to_csv(
        output_csv,
        index=False,
    )

    shap.plots.waterfall(
        explanation[0],
        max_display=12,
        show=False,
    )

    plt.tight_layout()

    output_plot = OUTPUT_DIR / "lightgbm_detected_drive_waterfall.png"

    plt.savefig(
        output_plot,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nSaved local SHAP values to: {output_csv}")
    print(f"Saved waterfall to: {output_plot}")


if __name__ == "__main__":
    main()
