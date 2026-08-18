from src.config import (
    FEATURE_COLUMNS,
    SMART_COLUMNS,
)


def test_feature_schema_has_expected_size() -> None:
    assert len(FEATURE_COLUMNS) == 42


def test_feature_names_are_unique() -> None:
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_all_smart_raw_columns_are_features() -> None:
    for column in SMART_COLUMNS:
        assert column in FEATURE_COLUMNS
