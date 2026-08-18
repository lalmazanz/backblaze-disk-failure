import duckdb

from src.config import (
    FINAL_TRAIN_END,
    FINAL_TRAIN_START,
)
from src.modeling.data import (
    load_undersampled_train,
)


def test_training_sample_is_deterministic() -> None:
    con = duckdb.connect()

    first = load_undersampled_train(
        con,
        FINAL_TRAIN_START,
        FINAL_TRAIN_END,
    )

    second = load_undersampled_train(
        con,
        FINAL_TRAIN_START,
        FINAL_TRAIN_END,
    )

    assert first.equals(second)
