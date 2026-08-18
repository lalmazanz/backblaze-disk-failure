import numpy as np
import pandas as pd
import shap

from src.config import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
    TEST_END,
    TEST_START,
)
from src.logging_utils import get_logger
from src.modeling.data import load_period
from src.modeling.inference import load_model

logger = get_logger(__name__)

SAMPLE_SIZE = 5000

IMPORTANCE_PATH = FIGURES_DIR / "final_model_shap_importance.csv"

BEESWARM_PATH = FIGURES_DIR / "final_model_shap_beeswarm.png"


def extract_class_one_shap_values(
    explanation: shap.Explanation,
) -> np.ndarray:
    values = explanation.values

    if values.ndim == 3:
        return values[:, :, 1]

    if values.ndim == 2:
        return values

    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")


def main() -> None:
    import duckdb
    import matplotlib.pyplot as plt

    con = duckdb.connect()

    logger.info("Loading final model artifact...")

    model = load_model()

    logger.info("Loading final test period...")

    test = load_period(
        con,
        TEST_START,
        TEST_END,
    )

    if len(test) > SAMPLE_SIZE:
        sample = test.sample(
            n=SAMPLE_SIZE,
            random_state=42,
        ).copy()
    else:
        sample = test.copy()

    x_sample = sample[FEATURE_COLUMNS]

    print(f"Test rows: {len(test)}")
    print(f"SHAP sample rows: {len(sample)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    logger.info("Building TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    logger.info("Computing SHAP values...")

    explanation = explainer(x_sample)

    shap_values = extract_class_one_shap_values(explanation)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "mean_abs_shap": (mean_abs_shap),
        }
    )

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    logger.info("Saving SHAP beeswarm...")

    class_one_explanation = shap.Explanation(
        values=shap_values,
        base_values=np.zeros(len(x_sample)),
        data=x_sample.to_numpy(),
        feature_names=(FEATURE_COLUMNS),
    )

    shap.plots.beeswarm(
        class_one_explanation,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        BEESWARM_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    logger.info("Global SHAP export completed.")

    print("\nFINAL MODEL — global SHAP importance")

    print(importance.head(20).to_string(index=False))

    print(f"\nImportance saved to: {IMPORTANCE_PATH}")

    print(f"Beeswarm saved to: {BEESWARM_PATH}")


if __name__ == "__main__":
    main()
