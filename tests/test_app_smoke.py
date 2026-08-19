from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.config import (
    FINAL_MODEL_PATH,
    MODEL_SCHEMA_PATH,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

APP_PATH = PROJECT_ROOT / "app" / "app.py"

DEMO_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "demo_predictions_compact.parquet"
)


def test_app_required_artifacts_exist() -> None:
    assert APP_PATH.exists()
    assert FINAL_MODEL_PATH.exists()
    assert MODEL_SCHEMA_PATH.exists()
    assert DEMO_DATA_PATH.exists()


def test_streamlit_app_starts_without_exception() -> None:
    app = AppTest.from_file(
        APP_PATH,
        default_timeout=30,
    )

    app.run()

    assert not app.exception
