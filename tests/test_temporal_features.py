from pathlib import Path

import pandas as pd

from src.config import SMART_COLUMNS
from src.features import build_features, build_target


def make_smart_row(
    date: str,
    serial_number: str = "TEST_DRIVE",
    model: str = "TEST_MODEL",
    failure: int = 0,
    value: float = 1.0,
) -> dict:
    row = {
        "date": pd.Timestamp(date),
        "serial_number": serial_number,
        "model": model,
        "failure": failure,
    }

    for column in SMART_COLUMNS:
        row[column] = value

    return row


def test_target_marks_only_previous_seven_days(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dates = pd.date_range(
        "2026-01-01",
        "2026-01-10",
        freq="D",
    )

    rows = []

    for date in dates:
        rows.append(
            make_smart_row(
                date=str(date.date()),
                failure=int(date == pd.Timestamp("2026-01-10")),
            )
        )

    source_path = tmp_path / "source.parquet"
    target_path = tmp_path / "target.parquet"

    pd.DataFrame(rows).to_parquet(
        source_path,
        index=False,
    )

    monkeypatch.setattr(
        build_target,
        "SUBSET_PATH",
        source_path,
    )
    monkeypatch.setattr(
        build_target,
        "TARGET_PATH",
        target_path,
    )

    build_target.main()

    result = pd.read_parquet(target_path).sort_values("date")

    positive_dates = set(
        result.loc[
            result["failure_next_7d"] == 1,
            "date",
        ]
    )

    expected_positive_dates = set(
        pd.date_range(
            "2026-01-03",
            "2026-01-09",
            freq="D",
        )
    )

    assert positive_dates == expected_positive_dates

    assert pd.Timestamp("2026-01-02") not in positive_dates

    assert pd.Timestamp("2026-01-10") not in set(result["date"])


def build_feature_dataset(
    tmp_path: Path,
    monkeypatch,
    future_value: float,
    suffix: str,
) -> pd.DataFrame:
    dates = pd.date_range(
        "2026-01-01",
        "2026-01-10",
        freq="D",
    )

    rows = []

    for index, date in enumerate(
        dates,
        start=1,
    ):
        value = float(index)

        if date > pd.Timestamp("2026-01-08"):
            value = future_value

        row = make_smart_row(
            date=str(date.date()),
            value=value,
        )

        row["failure_next_7d"] = 0

        rows.append(row)

    target_path = tmp_path / f"target_{suffix}.parquet"

    features_path = tmp_path / f"features_{suffix}.parquet"

    pd.DataFrame(rows).to_parquet(
        target_path,
        index=False,
    )

    monkeypatch.setattr(
        build_features,
        "TARGET_PATH",
        target_path,
    )
    monkeypatch.setattr(
        build_features,
        "FEATURES_PATH",
        features_path,
    )

    build_features.main()

    return pd.read_parquet(features_path)


def test_features_do_not_use_future_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    baseline = build_feature_dataset(
        tmp_path,
        monkeypatch,
        future_value=9.0,
        suffix="baseline",
    )

    altered_future = build_feature_dataset(
        tmp_path,
        monkeypatch,
        future_value=999999.0,
        suffix="altered",
    )

    cutoff = pd.Timestamp("2026-01-08")

    baseline_past = (
        baseline[baseline["date"] <= cutoff].sort_values("date").reset_index(drop=True)
    )

    altered_past = (
        altered_future[altered_future["date"] <= cutoff]
        .sort_values("date")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        baseline_past,
        altered_past,
    )
