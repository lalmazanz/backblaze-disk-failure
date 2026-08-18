import json

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, MODELS_DIR

MODEL_PATH = MODELS_DIR / "lightgbm_final.joblib"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"


class FeatureSchemaError(ValueError):
    pass


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def load_schema() -> dict:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Feature schema not found: {SCHEMA_PATH}")

    with SCHEMA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_schema(
    schema: dict,
) -> list[str]:
    if "features" not in schema:
        raise FeatureSchemaError("Schema does not contain a 'features' field.")

    expected_features = schema["features"]

    if expected_features != FEATURE_COLUMNS:
        raise FeatureSchemaError(
            "Saved feature schema does not match the current project configuration."
        )

    if schema.get("feature_count") != len(expected_features):
        raise FeatureSchemaError(
            "Schema feature_count does not match the number of saved features."
        )

    return expected_features


def validate_features(
    data: pd.DataFrame,
    expected_features: list[str],
) -> None:
    if data.empty:
        raise FeatureSchemaError("Input data contains no rows.")

    missing_features = [
        feature for feature in expected_features if feature not in data.columns
    ]

    if missing_features:
        raise FeatureSchemaError(
            "Missing required features: " + ", ".join(missing_features)
        )

    unexpected_features = [
        column for column in data.columns if column not in expected_features
    ]

    if unexpected_features:
        raise FeatureSchemaError(
            "Unexpected features: " + ", ".join(unexpected_features)
        )

    if list(data.columns) != expected_features:
        raise FeatureSchemaError("Feature columns are not in the expected order.")

    non_numeric = [
        column
        for column in expected_features
        if not pd.api.types.is_numeric_dtype(data[column])
    ]

    if non_numeric:
        raise FeatureSchemaError(
            "Non-numeric feature columns: " + ", ".join(non_numeric)
        )

    numeric_values = data[expected_features].to_numpy(
        dtype=float,
        copy=False,
    )

    if np.isinf(numeric_values).any():
        raise FeatureSchemaError("Input contains infinite values.")


def predict_risk(
    data: pd.DataFrame,
) -> pd.Series:
    schema = load_schema()

    expected_features = validate_schema(schema)

    validate_features(
        data,
        expected_features,
    )

    model = load_model()

    if getattr(
        model,
        "n_features_in_",
        None,
    ) != len(expected_features):
        raise FeatureSchemaError(
            "Model feature count does not match the saved feature schema."
        )

    risk_scores = model.predict_proba(data[expected_features])[:, 1]

    return pd.Series(
        risk_scores,
        index=data.index,
        name="risk_score",
    )


def predict_risk_from_row(
    row: dict,
) -> float:
    schema = load_schema()

    expected_features = validate_schema(schema)

    missing_features = [feature for feature in expected_features if feature not in row]

    if missing_features:
        raise FeatureSchemaError(
            "Missing required features: " + ", ".join(missing_features)
        )

    data = pd.DataFrame([{feature: row[feature] for feature in expected_features}])

    return float(predict_risk(data).iloc[0])
