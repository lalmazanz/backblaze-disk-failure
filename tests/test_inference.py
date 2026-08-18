import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.modeling.inference import (
    FeatureSchemaError,
    predict_risk,
)


def make_valid_input() -> pd.DataFrame:
    return pd.DataFrame(
        [{feature: 0.0 for feature in FEATURE_COLUMNS}],
        columns=FEATURE_COLUMNS,
    )


def test_valid_inference_returns_probability() -> None:
    data = make_valid_input()

    prediction = predict_risk(data)

    assert len(prediction) == 1
    assert 0.0 <= prediction.iloc[0] <= 1.0


def test_missing_feature_is_rejected() -> None:
    data = make_valid_input()

    data = data.drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(
        FeatureSchemaError,
        match="Missing required features",
    ):
        predict_risk(data)


def test_infinite_value_is_rejected() -> None:
    data = make_valid_input()

    data.loc[
        0,
        FEATURE_COLUMNS[0],
    ] = np.inf

    with pytest.raises(
        FeatureSchemaError,
        match="Input contains infinite values",
    ):
        predict_risk(data)


def test_nan_is_allowed() -> None:
    data = make_valid_input()

    data.loc[
        0,
        FEATURE_COLUMNS[3],
    ] = np.nan

    prediction = predict_risk(data)

    assert len(prediction) == 1
    assert 0.0 <= prediction.iloc[0] <= 1.0


def test_wrong_column_order_is_rejected() -> None:
    data = make_valid_input()

    reordered = data[list(reversed(FEATURE_COLUMNS))]

    with pytest.raises(
        FeatureSchemaError,
        match="not in the expected order",
    ):
        predict_risk(reordered)
