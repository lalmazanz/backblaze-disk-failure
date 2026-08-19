import pandas as pd

from src.config import FEATURE_COLUMNS
from src.modeling import data as modeling_data


def test_training_sample_is_deterministic(
    tmp_path,
    monkeypatch,
) -> None:
    rows = []

    # Two positive observations.
    for index in range(2):
        row = {
            "date": pd.Timestamp("2026-01-10") + pd.Timedelta(days=index),
            "model": "TEST_MODEL",
            "serial_number": f"positive_{index}",
            "failure_next_7d": 1,
        }

        for feature in FEATURE_COLUMNS:
            row[feature] = float(index)

        rows.append(row)

    # 120 negatives are enough for the default
    # 1:50 sampling ratio with 2 positives.
    for index in range(120):
        row = {
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index % 20),
            "model": "TEST_MODEL",
            "serial_number": f"negative_{index:03d}",
            "failure_next_7d": 0,
        }

        for feature_index, feature in enumerate(FEATURE_COLUMNS):
            row[feature] = float(index + feature_index)

        rows.append(row)

    synthetic_data = pd.DataFrame(rows)

    test_path = tmp_path / "synthetic_features.parquet"

    synthetic_data.to_parquet(
        test_path,
        index=False,
    )

    monkeypatch.setattr(
        modeling_data,
        "FEATURES_PATH",
        test_path,
    )

    con = modeling_data.duckdb.connect()

    first = modeling_data.load_undersampled_train(
        con,
        "2026-01-01",
        "2026-01-31",
    )

    second = modeling_data.load_undersampled_train(
        con,
        "2026-01-01",
        "2026-01-31",
    )

    assert first.equals(second)

    assert len(first) == 102
    assert int(first["failure_next_7d"].sum()) == 2
