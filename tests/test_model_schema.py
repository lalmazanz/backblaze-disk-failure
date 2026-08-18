from src.config import FEATURE_COLUMNS
from src.modeling.inference import (
    load_model,
    load_schema,
    validate_schema,
)


def test_saved_schema_matches_config() -> None:
    schema = load_schema()

    expected_features = validate_schema(schema)

    assert expected_features == FEATURE_COLUMNS
    assert schema["feature_count"] == 42


def test_model_matches_feature_schema() -> None:
    model = load_model()

    assert model.n_features_in_ == len(FEATURE_COLUMNS)
