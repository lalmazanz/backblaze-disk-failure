from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_BREAKDOWN_PATH = REPORTS_DIR / "model_breakdown.csv"

DETECTION_OUTPUT = FIGURES_DIR / "failure_detection_by_model.png"
LEAD_TIME_OUTPUT = FIGURES_DIR / "failure_warning_lead_time.png"


def plot_detection_by_model() -> None:
    data = pd.read_csv(MODEL_BREAKDOWN_PATH)

    data["missed_drives"] = data["positive_drives"] - data["detected_drives"]

    _, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        data["model"],
        data["detected_drives"],
        label="Detected",
    )

    ax.bar(
        data["model"],
        data["missed_drives"],
        bottom=data["detected_drives"],
        label="Missed",
    )

    for index, row in data.iterrows():
        recall = row["global_top_1pct_drive_recall"] * 100

        ax.text(
            index,
            row["positive_drives"] + 0.5,
            f"{int(row['detected_drives'])}/{int(row['positive_drives'])}\n"
            f"{recall:.1f}% recall",
            ha="center",
            va="bottom",
        )

    ax.set_title("Failure Detection by Drive Model\nDaily Top-1% Inspection Budget")
    ax.set_xlabel("Drive model")
    ax.set_ylabel("Failing drives")
    ax.legend()

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    plt.savefig(
        DETECTION_OUTPUT,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def plot_lead_time() -> None:
    lead_time = pd.DataFrame(
        {
            "lead_days": [1, 2, 3, 4, 5, 6, 7],
            "drives": [8, 3, 1, 2, 1, 2, 14],
        }
    )

    _, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        lead_time["lead_days"],
        lead_time["drives"],
    )

    for bar, count in zip(
        bars,
        lead_time["drives"],
        strict=True,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            str(count),
            ha="center",
            va="bottom",
        )

    ax.set_title("Warning Lead Time for Detected Failures\n31 failing drives detected")
    ax.set_xlabel("Days before failure")
    ax.set_ylabel("Detected drives")

    ax.set_xticks(range(1, 8))

    plt.tight_layout()

    plt.savefig(
        LEAD_TIME_OUTPUT,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_detection_by_model()
    plot_lead_time()

    print(f"Saved: {DETECTION_OUTPUT}")
    print(f"Saved: {LEAD_TIME_OUTPUT}")


if __name__ == "__main__":
    main()
