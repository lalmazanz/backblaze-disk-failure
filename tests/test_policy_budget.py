import pandas as pd

from src.evaluation.policy import add_daily_alert_policy


def test_daily_alert_policy_respects_budget() -> None:
    rows = []

    for index in range(250):
        rows.append(
            {
                "date": pd.Timestamp("2026-03-01"),
                "model": "MODEL_A",
                "serial_number": f"DRIVE_{index:03d}",
                "risk_score": float(index),
            }
        )

    data = pd.DataFrame(rows)

    scored = add_daily_alert_policy(
        data,
        top_pct=0.01,
    )

    alerts = scored[scored["top_1pct_alert"]]

    assert len(alerts) == 2


def test_daily_alert_policy_has_minimum_one_alert() -> None:
    rows = []

    for index in range(50):
        rows.append(
            {
                "date": pd.Timestamp("2026-03-01"),
                "model": "MODEL_A",
                "serial_number": f"DRIVE_{index:03d}",
                "risk_score": float(index),
            }
        )

    data = pd.DataFrame(rows)

    scored = add_daily_alert_policy(
        data,
        top_pct=0.01,
    )

    alerts = scored[scored["top_1pct_alert"]]

    assert len(alerts) == 1


def test_daily_budget_is_applied_independently_per_date() -> None:
    rows = []

    for date in [
        "2026-03-01",
        "2026-03-02",
    ]:
        for index in range(200):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "model": "MODEL_A",
                    "serial_number": f"{date}_{index:03d}",
                    "risk_score": float(index),
                }
            )

    data = pd.DataFrame(rows)

    scored = add_daily_alert_policy(
        data,
        top_pct=0.01,
    )

    alerts_per_date = scored[scored["top_1pct_alert"]].groupby("date").size()

    assert alerts_per_date.to_dict() == {
        pd.Timestamp("2026-03-01"): 2,
        pd.Timestamp("2026-03-02"): 2,
    }
