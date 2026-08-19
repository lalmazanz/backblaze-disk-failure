import sys
from pathlib import Path

import altair as alt
import pandas as pd
import shap
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.config import FEATURE_COLUMNS  # noqa: E402
from src.modeling.inference import load_model  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data/processed/demo_predictions_compact.parquet"

CATEGORY_LABELS = {
    "failed_and_alerted": "Failed and detected",
    "failed": "Failed but missed",
    "alerted": "Alerted, no observed failure",
    "normal_sample": "Normal sample",
}

SMART_LABELS = {
    "smart_5_raw": "SMART 5 — Reallocated sectors",
    "smart_197_raw": "SMART 197 — Current pending sectors",
    "smart_198_raw": "SMART 198 — Offline uncorrectable sectors",
}

SMART_BASE_LABELS = {
    "smart_1_raw": "SMART 1 — Raw read error rate",
    "smart_5_raw": "SMART 5 — Reallocated sectors",
    "smart_7_raw": "SMART 7 — Seek error rate",
    "smart_9_raw": "SMART 9 — Power-on hours",
    "smart_194_raw": "SMART 194 — Temperature",
    "smart_197_raw": "SMART 197 — Current pending sectors",
    "smart_198_raw": "SMART 198 — Offline uncorrectable sectors",
    "smart_199_raw": "SMART 199 — UDMA CRC errors",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Demo dataset not found: {DATA_PATH}")

    data = pd.read_parquet(DATA_PATH)

    data["date"] = pd.to_datetime(data["date"])
    data["failure_date"] = pd.to_datetime(data["failure_date"])

    return data


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_explainer():
    model = get_model()

    classifier = model.named_steps["classifier"]

    return shap.TreeExplainer(classifier)


def format_risk(
    score: float,
) -> str:
    return f"{score * 100:.2f}%"


def friendly_feature_name(
    feature: str,
) -> str:
    if feature == "has_1d_history":
        return "1-day history available"

    if feature == "has_7d_history":
        return "7-day history available"

    suffix_labels = {
        "_delta_1d": " — 1-day change",
        "_delta_7d": " — 7-day change",
        "_mean_7d": " — 7-day mean",
        "_max_7d": " — 7-day maximum",
    }

    for (
        base_feature,
        base_label,
    ) in SMART_BASE_LABELS.items():
        if feature == base_feature:
            return f"{base_label} — current value"

        for (
            suffix,
            suffix_label,
        ) in suffix_labels.items():
            if feature == (base_feature + suffix):
                return base_label + suffix_label

    return feature


def get_default_date_index(
    drive_data: pd.DataFrame,
    category: str,
) -> int:
    if category in {
        "alerted",
        "failed_and_alerted",
    }:
        alerted_indexes = drive_data.index[drive_data["top_1pct_alert"]].tolist()

        if alerted_indexes:
            return alerted_indexes[0]

    return len(drive_data) - 1


def build_risk_chart(
    drive_data: pd.DataFrame,
    selected_row: pd.Series,
) -> alt.LayerChart:
    chart_data = drive_data[
        [
            "date",
            "risk_score",
            "top_1pct_alert",
        ]
    ].copy()

    selected_data = pd.DataFrame(
        {
            "date": [selected_row["date"]],
            "risk_score": [selected_row["risk_score"]],
        }
    )

    alerts_data = chart_data[chart_data["top_1pct_alert"]].copy()

    base = alt.Chart(chart_data).encode(
        x=alt.X(
            "date:T",
            title="Date",
            axis=alt.Axis(
                format="%b %d",
                labelAngle=0,
            ),
        ),
        y=alt.Y(
            "risk_score:Q",
            title="Risk score",
            axis=alt.Axis(
                format=".0%",
            ),
            scale=alt.Scale(
                zero=True,
            ),
        ),
    )

    line = base.mark_line(
        strokeWidth=2,
    ).encode(
        tooltip=[
            alt.Tooltip(
                "date:T",
                title="Date",
                format="%Y-%m-%d",
            ),
            alt.Tooltip(
                "risk_score:Q",
                title="Risk score",
                format=".2%",
            ),
        ]
    )

    selected_rule = (
        alt.Chart(selected_data)
        .mark_rule(
            strokeDash=[
                5,
                5,
            ],
        )
        .encode(
            x="date:T",
        )
    )

    selected_point = (
        alt.Chart(selected_data)
        .mark_point(
            size=220,
            filled=True,
            shape="circle",
        )
        .encode(
            x="date:T",
            y="risk_score:Q",
        )
    )

    chart = line + selected_rule + selected_point

    if not alerts_data.empty:
        alert_points = (
            alt.Chart(alerts_data)
            .mark_point(
                size=130,
                filled=True,
                shape="diamond",
            )
            .encode(
                x="date:T",
                y="risk_score:Q",
            )
        )

        chart += alert_points

    failure_date = selected_row["failure_date"]

    if pd.notna(failure_date):
        failure_data = pd.DataFrame({"date": [failure_date]})

        failure_rule = (
            alt.Chart(failure_data)
            .mark_rule(
                strokeWidth=2,
                strokeDash=[
                    2,
                    3,
                ],
            )
            .encode(
                x="date:T",
            )
        )

        chart += failure_rule

    return chart.properties(
        height=400,
    ).interactive()


def build_smart_chart(
    drive_data: pd.DataFrame,
    selected_row: pd.Series,
    feature: str,
) -> alt.LayerChart:
    chart_data = drive_data[
        [
            "date",
            feature,
        ]
    ].copy()

    selected_data = pd.DataFrame(
        {
            "date": [selected_row["date"]],
            "value": [selected_row[feature]],
        }
    )

    line = (
        alt.Chart(chart_data)
        .mark_line(
            strokeWidth=2,
        )
        .encode(
            x=alt.X(
                "date:T",
                title="Date",
                axis=alt.Axis(
                    format="%b %d",
                    labelAngle=0,
                ),
            ),
            y=alt.Y(
                f"{feature}:Q",
                title="Raw value",
                scale=alt.Scale(
                    zero=False,
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "date:T",
                    title="Date",
                    format="%Y-%m-%d",
                ),
                alt.Tooltip(
                    f"{feature}:Q",
                    title="Value",
                    format=".2f",
                ),
            ],
        )
    )

    selected_rule = (
        alt.Chart(selected_data)
        .mark_rule(
            strokeDash=[
                5,
                5,
            ],
        )
        .encode(
            x="date:T",
        )
    )

    selected_point = (
        alt.Chart(selected_data)
        .mark_point(
            size=140,
            filled=True,
        )
        .encode(
            x="date:T",
            y="value:Q",
        )
    )

    chart = line + selected_rule + selected_point

    failure_date = selected_row["failure_date"]

    if pd.notna(failure_date):
        failure_data = pd.DataFrame({"date": [failure_date]})

        failure_rule = (
            alt.Chart(failure_data)
            .mark_rule(
                strokeWidth=2,
                strokeDash=[
                    2,
                    3,
                ],
            )
            .encode(
                x="date:T",
            )
        )

        chart += failure_rule

    return chart.properties(
        height=240,
    ).interactive()


def get_smart_summary(
    drive_data: pd.DataFrame,
    selected_row: pd.Series,
    feature: str,
) -> dict:
    series = drive_data[feature].dropna()

    if series.empty:
        return {
            "current": None,
            "change_7d": None,
            "trend": "Unavailable",
            "is_constant": True,
            "all_zero": False,
        }

    current_value = selected_row[feature]
    selected_date = selected_row["date"]

    target_date = selected_date - pd.Timedelta(days=7)

    previous_rows = drive_data[drive_data["date"] <= target_date]

    if previous_rows.empty:
        change_7d = None
    else:
        previous_value = previous_rows.iloc[-1][feature]

        if pd.isna(previous_value):
            change_7d = None
        else:
            change_7d = current_value - previous_value

    is_constant = series.max() == series.min()

    all_zero = is_constant and series.iloc[0] == 0

    if all_zero:
        trend = "No signal observed"
    elif is_constant:
        trend = "Stable"
    elif change_7d is None:
        trend = "Variable"
    elif change_7d > 0:
        trend = "Increasing"
    elif change_7d < 0:
        trend = "Decreasing"
    else:
        trend = "Stable recently"

    return {
        "current": current_value,
        "change_7d": change_7d,
        "trend": trend,
        "is_constant": is_constant,
        "all_zero": all_zero,
    }


def render_smart_signal(
    drive_data: pd.DataFrame,
    selected_row: pd.Series,
    feature: str,
) -> None:
    summary = get_smart_summary(
        drive_data,
        selected_row,
        feature,
    )

    st.markdown(f"**{SMART_LABELS[feature]}**")

    if summary["all_zero"]:
        st.metric(
            "Current value",
            "0",
        )

        st.success("No degradation signal observed during the displayed period.")

        return

    st.altair_chart(
        build_smart_chart(
            drive_data,
            selected_row,
            feature,
        ),
        width="stretch",
    )

    (
        metric_col1,
        metric_col2,
    ) = st.columns(2)

    if summary["current"] is None:
        current_display = "N/A"
    else:
        current_display = f"{summary['current']:.0f}"

    metric_col1.metric(
        "Current value",
        current_display,
    )

    metric_col2.metric(
        "Trend",
        summary["trend"],
    )

    if summary["change_7d"] is None:
        st.caption("7-day change: unavailable")
    else:
        st.caption(f"7-day change: {summary['change_7d']:+.0f}")

    if summary["is_constant"]:
        st.caption("Persistent non-zero signal during the observed period.")


def get_local_shap(
    selected_row: pd.Series,
) -> pd.DataFrame:
    x_selected = pd.DataFrame(
        [selected_row[FEATURE_COLUMNS].to_dict()],
        columns=FEATURE_COLUMNS,
    )

    model = get_model()

    imputer = model.named_steps["imputer"]

    x_imputed = imputer.transform(x_selected)

    x_imputed = pd.DataFrame(
        x_imputed,
        columns=FEATURE_COLUMNS,
        index=x_selected.index,
    )

    explainer = get_explainer()

    explanation = explainer(x_imputed)

    values = explanation.values

    if values.ndim == 3:
        shap_values = values[0, :, 1]
    elif values.ndim == 2:
        shap_values = values[0]
    else:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

    shap_data = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "feature_value": (x_imputed.iloc[0].values),
            "shap_value": shap_values,
        }
    )

    shap_data["abs_shap"] = shap_data["shap_value"].abs()

    shap_data["feature_label"] = shap_data["feature"].map(friendly_feature_name)

    shap_data["direction"] = shap_data["shap_value"].apply(
        lambda value: "Increases risk" if value > 0 else "Reduces risk"
    )

    return shap_data.sort_values(
        "abs_shap",
        ascending=False,
    )


def build_shap_chart(
    shap_data: pd.DataFrame,
) -> alt.LayerChart:
    top_features = shap_data.head(8).sort_values("shap_value").copy()

    bars = (
        alt.Chart(top_features)
        .mark_bar()
        .encode(
            x=alt.X(
                "shap_value:Q",
                title=("SHAP contribution to failure-risk score"),
            ),
            y=alt.Y(
                "feature_label:N",
                title=None,
                sort=None,
                axis=alt.Axis(
                    labelLimit=420,
                    labelPadding=8,
                ),
            ),
            color=alt.Color(
                "direction:N",
                title=None,
                scale=alt.Scale(
                    domain=[
                        "Increases risk",
                        "Reduces risk",
                    ],
                    range=[
                        "#ef4444",
                        "#3b82f6",
                    ],
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "feature_label:N",
                    title="Feature",
                ),
                alt.Tooltip(
                    "feature_value:Q",
                    title="Observed value",
                    format=".2f",
                ),
                alt.Tooltip(
                    "shap_value:Q",
                    title="SHAP value",
                    format="+.3f",
                ),
                alt.Tooltip(
                    "direction:N",
                    title="Effect",
                ),
            ],
        )
        .properties(
            height=330,
        )
    )

    zero_line = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule()
        .encode(
            x="x:Q",
        )
    )

    return (bars + zero_line).interactive()


def render_shap_summary(
    shap_data: pd.DataFrame,
) -> None:
    positive = shap_data[shap_data["shap_value"] > 0].head(3)

    negative = shap_data[shap_data["shap_value"] < 0].head(3)

    left, right = st.columns(2)

    with left:
        st.markdown("**Main factors increasing risk**")

        if positive.empty:
            st.caption("No strong positive contributors.")
        else:
            for _, row in positive.iterrows():
                st.write(f"• {row['feature_label']} ({row['shap_value']:+.3f})")

    with right:
        st.markdown("**Main factors reducing risk**")

        if negative.empty:
            st.caption("No strong negative contributors.")
        else:
            for _, row in negative.iterrows():
                st.write(f"• {row['feature_label']} ({row['shap_value']:+.3f})")


def main() -> None:
    st.set_page_config(
        page_title="Disk Failure Early Warning",
        page_icon="💾",
        layout="wide",
    )

    st.title(
        "Disk Failure Early Warning System",
        anchor=False,
    )

    st.caption(
        "Predictive maintenance demo "
        "using Backblaze SMART telemetry "
        "and a Random Forest model. "
        "The system ranks hard drives "
        "according to their estimated "
        "risk of failure within the "
        "next 7 days."
    )

    data = load_data()

    st.sidebar.header(
        "Drive selection",
        anchor=False,
    )

    available_categories = data["demo_category"].drop_duplicates().tolist()

    selected_category = st.sidebar.selectbox(
        "Case type",
        options=available_categories,
        format_func=lambda value: CATEGORY_LABELS[value],
    )

    category_data = data[data["demo_category"] == selected_category]

    available_drives = sorted(category_data["drive_id"].drop_duplicates().tolist())

    selected_drive = st.sidebar.selectbox(
        "Drive",
        options=available_drives,
    )

    drive_data = (
        data[data["drive_id"] == selected_drive]
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    available_dates = drive_data["date"].dt.date.tolist()

    default_date_index = get_default_date_index(
        drive_data,
        selected_category,
    )

    selected_date = st.sidebar.selectbox(
        "Observation date",
        options=available_dates,
        index=default_date_index,
    )

    selected_rows = drive_data[drive_data["date"].dt.date == selected_date]

    if selected_rows.empty:
        st.error("No observation was found for the selected date.")
        return

    selected_row = selected_rows.iloc[0]

    st.subheader(
        "Current observation",
        anchor=False,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Risk score",
        format_risk(selected_row["risk_score"]),
    )

    col2.metric(
        "Daily risk rank",
        f"#{int(selected_row['risk_rank'])}",
    )

    alert_status = "High risk" if selected_row["top_1pct_alert"] else "Not alerted"

    col3.metric(
        "Operational status",
        alert_status,
    )

    horizon_status = "Yes" if selected_row["within_failure_horizon"] else "No"

    col4.metric(
        "Failure within 7 days",
        horizon_status,
    )

    st.caption(
        "The displayed risk score "
        "is used for ranking drives. "
        "Because the model was trained "
        "with 1:50 negative undersampling, "
        "it should not be interpreted "
        "as a calibrated real-world "
        "failure probability."
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader(
            "Drive information",
            anchor=False,
        )

        st.write(f"**Model:** {selected_row['model']}")

        st.write(f"**Serial:** {selected_row['serial_number']}")

        st.write(f"**Observation date:** {selected_row['date'].date()}")

        if pd.notna(selected_row["failure_date"]):
            st.write(
                f"**Observed failure date:** {selected_row['failure_date'].date()}"
            )

            if pd.notna(selected_row["days_to_failure"]):
                st.write(f"**Days to failure:** {int(selected_row['days_to_failure'])}")
        else:
            st.write("**Observed failure date:** No failure observed in Q1 2026")

    with right:
        st.subheader(
            "Operational interpretation",
            anchor=False,
        )

        if selected_row["top_1pct_alert"]:
            st.warning(
                "This observation belongs "
                "to the daily top 1% "
                "highest-risk drives. "
                "Under the inspection "
                "policy, it would be "
                "flagged for maintenance "
                "review."
            )
        else:
            st.success(
                "This observation is outside the daily top 1% inspection budget."
            )

        if selected_row["within_failure_horizon"]:
            st.error("The drive actually failed within the following 7 days.")

    st.divider()

    st.subheader(
        "Risk evolution",
        anchor=False,
    )

    st.altair_chart(
        build_risk_chart(
            drive_data,
            selected_row,
        ),
        width="stretch",
    )

    (
        legend_col1,
        legend_col2,
        legend_col3,
    ) = st.columns(3)

    legend_col1.caption("● Selected observation")

    legend_col2.caption("◆ Daily top 1% alert")

    if pd.notna(selected_row["failure_date"]):
        legend_col3.caption("┆ Observed failure date")

    st.caption(
        "Operational alerts are "
        "based on the daily top 1% "
        "risk ranking rather than "
        "a fixed score threshold."
    )

    st.divider()

    st.subheader(
        "SMART degradation signals",
        anchor=False,
    )

    st.caption(
        "Sector-health indicators "
        "used by the model. "
        "Zero-only signals are "
        "summarized compactly; "
        "persistent or changing "
        "signals retain their "
        "temporal chart."
    )

    (
        smart_5_col,
        smart_197_col,
        smart_198_col,
    ) = st.columns(3)

    with smart_5_col:
        render_smart_signal(
            drive_data,
            selected_row,
            "smart_5_raw",
        )

    with smart_197_col:
        render_smart_signal(
            drive_data,
            selected_row,
            "smart_197_raw",
        )

    with smart_198_col:
        render_smart_signal(
            drive_data,
            selected_row,
            "smart_198_raw",
        )

    st.divider()

    st.subheader(
        "Why this prediction?",
        anchor=False,
    )

    st.caption(
        "Local SHAP values explain "
        "how each feature contributed "
        "to the Random Forest output "
        "for this observation. "
        "Positive values push the "
        "failure-risk score upward; "
        "negative values push it "
        "downward."
    )

    shap_data = get_local_shap(selected_row)

    st.altair_chart(
        build_shap_chart(shap_data),
        width="stretch",
    )

    render_shap_summary(shap_data)

    st.caption(
        "SHAP values explain the "
        "model's prediction for this "
        "observation. They describe "
        "model behavior, not causal "
        "effects or calibrated "
        "probability changes."
    )


if __name__ == "__main__":
    main()
