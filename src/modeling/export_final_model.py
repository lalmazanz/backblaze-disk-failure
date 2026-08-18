import json
from datetime import datetime, timezone

import duckdb
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    FEATURE_COLUMNS,
    FINAL_MODEL_PATH,
    FINAL_TRAIN_END,
    FINAL_TRAIN_START,
    MODEL_SCHEMA_PATH,
    MODELS_DIR,
    NEGATIVE_RATIO,
    PREDICTION_HORIZON_DAYS,
    RANDOM_FOREST_PARAMS,
    RANDOM_STATE,
)
from src.logging_utils import get_logger
from src.modeling.data import load_undersampled_train

logger = get_logger(__name__)


def main() -> None:
    con = duckdb.connect()

    logger.info("Loading final purged training sample...")

    train = load_undersampled_train(
        con,
        FINAL_TRAIN_START,
        FINAL_TRAIN_END,
        negative_ratio=NEGATIVE_RATIO,
    )

    positive_rows = int(train["failure_next_7d"].sum())
    negative_rows = len(train) - positive_rows

    print(f"Training period: {FINAL_TRAIN_START} -> {FINAL_TRAIN_END}")
    print(f"Training rows: {len(train)}")
    print(f"Training positives: {positive_rows}")
    print(f"Training negatives: {negative_rows}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    x_train = train[FEATURE_COLUMNS]
    y_train = train["failure_next_7d"]

    logger.info("Training final Random Forest...")

    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)

    model.fit(
        x_train,
        y_train,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Saving final model artifact...")

    joblib.dump(
        model,
        FINAL_MODEL_PATH,
    )

    schema = {
        "model_type": "RandomForestClassifier",
        "target": "failure_next_7d",
        "prediction_horizon_days": (PREDICTION_HORIZON_DAYS),
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "training_period": {
            "start": FINAL_TRAIN_START,
            "end": FINAL_TRAIN_END,
        },
        "negative_ratio": NEGATIVE_RATIO,
        "random_state": RANDOM_STATE,
        "training_rows": len(train),
        "training_positive_rows": (positive_rows),
        "training_negative_rows": (negative_rows),
        "model_params": (RANDOM_FOREST_PARAMS),
        "exported_at_utc": (datetime.now(timezone.utc).isoformat()),
    }

    logger.info("Saving feature schema...")

    with MODEL_SCHEMA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            schema,
            file,
            indent=2,
        )

    logger.info("Final Random Forest export completed.")

    print(f"\nModel saved to: {FINAL_MODEL_PATH}")
    print(f"Schema saved to: {MODEL_SCHEMA_PATH}")


if __name__ == "__main__":
    main()
