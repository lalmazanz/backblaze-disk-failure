import pandas as pd

from src.evaluation.evaluate_final import (
    build_daily_alerts,
)


def test_daily_alerts_select_top_one_percent() -> None:
    rows = 1000

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-11"] * rows),
            "risk_score": [value / rows for value in range(rows)],
        }
    )

    alerts = build_daily_alerts(data)

    assert len(alerts) == 10

    assert alerts["risk_score"].min() >= 0.99
