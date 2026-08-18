import pandas as pd

from src.config import TOP_PCT


def add_daily_alert_policy(
    data: pd.DataFrame,
    top_pct: float = TOP_PCT,
) -> pd.DataFrame:
    required_columns = {
        "date",
        "model",
        "serial_number",
        "risk_score",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "Missing columns required "
            "for alert policy: " + ", ".join(sorted(missing_columns))
        )

    result = data.copy()

    result["_original_index"] = range(len(result))

    result = result.sort_values(
        [
            "date",
            "risk_score",
            "model",
            "serial_number",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
    )

    result["risk_rank"] = result.groupby("date").cumcount() + 1

    daily_count = result.groupby("date")["risk_score"].transform("size")

    daily_budget = (daily_count * top_pct).astype(int)

    daily_budget = daily_budget.clip(lower=1)

    result["risk_percentile"] = result["risk_rank"] / daily_count

    result["top_1pct_alert"] = result["risk_rank"] <= daily_budget

    result = (
        result.sort_values("_original_index")
        .drop(columns="_original_index")
        .reset_index(drop=True)
    )

    return result


def get_daily_alerts(
    data: pd.DataFrame,
    top_pct: float = TOP_PCT,
) -> pd.DataFrame:
    scored = add_daily_alert_policy(
        data,
        top_pct=top_pct,
    )

    return scored[scored["top_1pct_alert"]].copy().reset_index(drop=True)
