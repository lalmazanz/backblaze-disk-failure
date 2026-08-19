import numpy as np
import pandas as pd
import shap

from src.config import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
)
from src.logging_utils import get_logger
from src.modeling.inference import load_model

logger = get_logger(__name__)

DEMO_PATH = "data/processed/demo_predictions.parquet"

CSV_OUTPUT_PATH = FIGURES_DIR / "final_model_detected_drive_shap.csv"

WATERFALL_OUTPUT_PATH = FIGURES_DIR / "final_model_detected_drive_waterfall.png"


def extract_class_one_values(
    explanation: shap.Explanation,
) -> tuple[
    np.ndarray,
    float,
]:
    values = explanation.values
    base_values = explanation.base_values

    if values.ndim == 3:
        shap_values = values[0, :, 1]

        if np.ndim(base_values) == 2:
            base_value = float(base_values[0, 1])
        else:
            base_value = float(np.asarray(base_values).reshape(-1)[1])

        return (
            shap_values,
            base_value,
        )

    if values.ndim == 2:
        shap_values = values[0]

        base_value = float(np.asarray(base_values).reshape(-1)[0])

        return (
            shap_values,
            base_value,
        )

    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")


def select_detected_case(
    data: pd.DataFrame,
) -> pd.Series:
    detected = data[
        (data["top_1pct_alert"])
        & (data["failure_next_7d"] == 1)
        & (data["failure_date"].notna())
    ].copy()

    if detected.empty:
        raise ValueError("No detected failure case was found in demo data.")

    detected["lead_days"] = (detected["failure_date"] - detected["date"]).dt.days

    detected = detected[
        detected["lead_days"].between(
            1,
            7,
            inclusive="both",
        )
    ].copy()

    if detected.empty:
        raise ValueError(
            "Detected alerts exist, but none fall inside the 1-7 day failure horizon."
        )

    earliest_alerts = (
        detected.sort_values(
            [
                "drive_id",
                "date",
            ]
        )
        .groupby(
            "drive_id",
            as_index=False,
        )
        .first()
    )

    case = earliest_alerts.sort_values(
        [
            "lead_days",
            "risk_score",
        ],
        ascending=[
            False,
            False,
        ],
    ).iloc[0]

    return case


def main() -> None:
    import matplotlib.pyplot as plt

    logger.info("Loading demo predictions...")

    data = pd.read_parquet(DEMO_PATH)

    data["date"] = pd.to_datetime(data["date"])

    data["failure_date"] = pd.to_datetime(data["failure_date"])

    logger.info("Selecting a detected failure case...")

    case = select_detected_case(data)

    x_selected = pd.DataFrame(
        [case[FEATURE_COLUMNS].to_dict()],
        columns=FEATURE_COLUMNS,
    )

    logger.info("Loading final model...")

    model = load_model()

    logger.info("Computing local SHAP values...")

    explainer = shap.TreeExplainer(model)

    explanation = explainer(x_selected)

    (
        shap_values,
        base_value,
    ) = extract_class_one_values(explanation)

    shap_table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "feature_value": (x_selected.iloc[0].values),
            "shap_value": shap_values,
        }
    )

    shap_table["abs_shap"] = shap_table["shap_value"].abs()

    shap_table = shap_table.sort_values(
        "abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shap_table.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
    )

    local_explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=(x_selected.iloc[0].to_numpy()),
        feature_names=(FEATURE_COLUMNS),
    )

    logger.info("Saving local SHAP waterfall...")

    shap.plots.waterfall(
        local_explanation,
        max_display=12,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        WATERFALL_OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    logger.info("Detected-drive explanation export completed.")

    print("\nFINAL MODEL — detected drive case")

    print(f"Drive: {case['drive_id']}")

    print(f"Observation date: {case['date'].date()}")

    print(f"Failure date: {case['failure_date'].date()}")

    print(f"Lead time: {int(case['lead_days'])} days")

    print(f"Risk score: {case['risk_score']:.6f}")

    print("\nTop local SHAP contributors:")

    print(shap_table.head(15).to_string(index=False))

    print(f"\nSHAP table saved to: {CSV_OUTPUT_PATH}")

    print(f"Waterfall saved to: {WATERFALL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
