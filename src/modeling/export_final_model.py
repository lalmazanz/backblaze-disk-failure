import json
from datetime import UTC, datetime

import duckdb
import joblib
from lightgbm import LGBMClassifier

from src.config import (
    FEATURE_COLUMNS,
    FINAL_TRAIN_END,
    FINAL_TRAIN_START,
    LIGHTGBM_PARAMS,
    MODELS_DIR,
    NEGATIVE_RATIO,
    PREDICTION_HORIZON_DAYS,
    RANDOM_STATE,
)
from src.modeling.data import load_undersampled_train

MODEL_PATH = MODELS_DIR / "lightgbm_final.joblib"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"


def main() -> None:
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = duckdb.connect()

    print("Loading final training sample...")

    train = load_undersampled_train(
        con,
        FINAL_TRAIN_START,
        FINAL_TRAIN_END,
    )

    x_train = train[FEATURE_COLUMNS]
    y_train = train["failure_next_7d"]

    positive_rows = int(y_train.sum())
    negative_rows = len(train) - positive_rows

    print(f"Training rows: {len(train)}")
    print(f"Training positives: {positive_rows}")
    print(f"Training negatives: {negative_rows}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    model = LGBMClassifier(
        **LIGHTGBM_PARAMS,
    )

    print("Training final LightGBM...")

    model.fit(
        x_train,
        y_train,
    )

    print("Saving model...")

    joblib.dump(
        model,
        MODEL_PATH,
    )

    schema = {
        "model_type": "LightGBM",
        "target": "failure_next_7d",
        "prediction_horizon_days": (PREDICTION_HORIZON_DAYS),
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "training": {
            "start_date": FINAL_TRAIN_START,
            "end_date": FINAL_TRAIN_END,
            "negative_ratio": NEGATIVE_RATIO,
            "random_state": RANDOM_STATE,
            "rows": len(train),
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
        },
        "model_params": LIGHTGBM_PARAMS,
        "exported_at_utc": datetime.now(UTC).isoformat(),
    }

    with SCHEMA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            schema,
            file,
            indent=2,
        )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Schema saved to: {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
