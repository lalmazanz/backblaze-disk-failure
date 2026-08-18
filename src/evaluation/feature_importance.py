import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
)
from src.logging_utils import get_logger
from src.modeling.inference import load_model

logger = get_logger(__name__)

OUTPUT_PATH = FIGURES_DIR / "final_model_feature_importance.csv"


def main() -> None:
    logger.info("Loading final model artifact...")

    model = load_model()

    if not hasattr(
        model,
        "feature_importances_",
    ):
        raise AttributeError("Final model does not expose 'feature_importances_'.")

    importances = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": (model.feature_importances_),
        }
    )

    importances = importances.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    importances.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    logger.info("Feature importance export completed.")

    print("\nFINAL MODEL — feature importance")

    print(importances.head(20).to_string(index=False))

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
