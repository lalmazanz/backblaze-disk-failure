import json
from collections.abc import Mapping

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    FINAL_MODEL_PATH,
    MODEL_SCHEMA_PATH,
)


class FeatureSchemaError(ValueError):
    """Raised when inference features do not match the model schema."""


def load_model():
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Final model artifact not found: {FINAL_MODEL_PATH}")

    return joblib.load(FINAL_MODEL_PATH)


def load_schema() -> dict:
    if not MODEL_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Model schema not found: {MODEL_SCHEMA_PATH}")

    with MODEL_SCHEMA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_schema(
    schema: dict,
) -> list[str]:
    if "features" not in schema:
        raise FeatureSchemaError("Schema does not contain a 'features' field.")

    features = schema["features"]

    if not isinstance(features, list):
        raise FeatureSchemaError("Schema 'features' field must be a list.")

    if features != FEATURE_COLUMNS:
        raise FeatureSchemaError(
            "Saved model feature schema does not match current config."
        )

    feature_count = schema.get("feature_count")

    if feature_count != len(features):
        raise FeatureSchemaError(
            "Schema feature_count does not match the number of features."
        )

    return features


def validate_features(
    features: pd.DataFrame,
    expected_features: list[str],
) -> None:
    if features.empty:
        raise FeatureSchemaError("Inference input is empty.")

    actual_columns = list(features.columns)

    missing = [column for column in expected_features if column not in actual_columns]

    unexpected = [
        column for column in actual_columns if column not in expected_features
    ]

    if missing:
        raise FeatureSchemaError("Missing required features: " + ", ".join(missing))

    if unexpected:
        raise FeatureSchemaError("Unexpected features: " + ", ".join(unexpected))

    if actual_columns != expected_features:
        raise FeatureSchemaError("Feature columns are not in the expected order.")

    non_numeric = [
        column
        for column in expected_features
        if not pd.api.types.is_numeric_dtype(features[column])
    ]

    if non_numeric:
        raise FeatureSchemaError("Non-numeric features: " + ", ".join(non_numeric))

    values = features[expected_features].to_numpy(
        dtype=float,
        copy=False,
    )

    if np.isinf(values).any():
        raise FeatureSchemaError("Input contains infinite values.")

    # NaN values are intentionally allowed.
    # The exported model pipeline applies median
    # imputation before Random Forest inference.


def predict_risk(
    features: pd.DataFrame,
) -> pd.Series:
    schema = load_schema()

    expected_features = validate_schema(schema)

    validate_features(
        features,
        expected_features,
    )

    model = load_model()

    model_feature_count = getattr(
        model,
        "n_features_in_",
        None,
    )

    if model_feature_count is not None and model_feature_count != len(
        expected_features
    ):
        raise FeatureSchemaError("Model feature count does not match saved schema.")

    probabilities = model.predict_proba(features[expected_features])[:, 1]

    return pd.Series(
        probabilities,
        index=features.index,
        name="risk_score",
    )


def predict_risk_from_row(
    row: Mapping[str, object],
) -> float:
    missing = [feature for feature in FEATURE_COLUMNS if feature not in row]

    if missing:
        raise FeatureSchemaError("Missing required features: " + ", ".join(missing))

    features = pd.DataFrame(
        [{feature: row[feature] for feature in FEATURE_COLUMNS}],
        columns=FEATURE_COLUMNS,
    )

    return float(predict_risk(features).iloc[0])
