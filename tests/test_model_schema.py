from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import (
    FEATURE_COLUMNS,
    FINAL_MODEL_PATH,
    MODEL_SCHEMA_PATH,
)
from src.modeling.inference import (
    load_model,
    load_schema,
    validate_schema,
)


def test_model_artifacts_exist() -> None:
    assert FINAL_MODEL_PATH.exists()
    assert MODEL_SCHEMA_PATH.exists()


def test_saved_schema_matches_config() -> None:
    schema = load_schema()

    expected_features = validate_schema(schema)

    assert expected_features == FEATURE_COLUMNS
    assert schema["feature_count"] == len(FEATURE_COLUMNS)


def test_saved_schema_describes_deployed_pipeline() -> None:
    schema = load_schema()

    assert schema["model_type"] == "RandomForestClassifier"

    preprocessing = schema["preprocessing"]

    assert preprocessing["imputer"] == "SimpleImputer"
    assert preprocessing["strategy"] == "median"


def test_model_is_expected_pipeline() -> None:
    model = load_model()

    assert isinstance(model, Pipeline)

    assert list(model.named_steps) == [
        "imputer",
        "classifier",
    ]

    assert isinstance(
        model.named_steps["imputer"],
        SimpleImputer,
    )

    assert model.named_steps["imputer"].strategy == "median"

    assert isinstance(
        model.named_steps["classifier"],
        RandomForestClassifier,
    )


def test_model_matches_feature_schema() -> None:
    model = load_model()
    schema = load_schema()

    assert model.n_features_in_ == len(FEATURE_COLUMNS)

    assert schema["feature_count"] == (model.n_features_in_)

    assert schema["features"] == FEATURE_COLUMNS
