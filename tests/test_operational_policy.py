import pandas as pd

from src.evaluation.policy import (
    add_daily_alert_policy,
    get_daily_alerts,
)


def make_test_data(
    rows: int = 1000,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-11"] * rows),
            "model": ["TEST_MODEL"] * rows,
            "serial_number": [f"drive_{index:04d}" for index in range(rows)],
            "risk_score": [value / rows for value in range(rows)],
        }
    )


def test_daily_alerts_select_top_one_percent() -> None:
    data = make_test_data()

    alerts = get_daily_alerts(data)

    assert len(alerts) == 10

    assert alerts["risk_score"].min() >= 0.99


def test_daily_policy_assigns_unique_ranks() -> None:
    data = make_test_data()

    scored = add_daily_alert_policy(data)

    assert scored["risk_rank"].nunique() == len(scored)


def test_ties_are_deterministic() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-11"] * 100),
            "model": ["MODEL"] * 100,
            "serial_number": [f"drive_{index:03d}" for index in range(100)],
            "risk_score": [0.5] * 100,
        }
    )

    alerts = get_daily_alerts(data)

    assert len(alerts) == 1

    assert alerts.iloc[0]["serial_number"] == "drive_000"
