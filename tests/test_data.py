import numpy as np
import pandas as pd
import pytest

from src.data import split_data


@pytest.fixture
def dummy_data():
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        rng.normal(size=(1000, 4)),
        columns=["f1", "f2", "f3", "f4"],
    )
    df["default"] = rng.choice([0, 1], size=1000, p=[0.7, 0.3])
    return df

@pytest.fixture
def dummy_config():
    return {
        "seed": 42,
        "data": {"target": "default"},
        "split": {"train_size": 0.7, "stratify": True},
    }


def test_split_sizes(dummy_data, dummy_config):
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(
        dummy_data,
        dummy_config,
    )

    assert len(X_train) == 700
    assert len(X_val) == 150
    assert len(X_test) == 150


def test_split_default_rate(dummy_data, dummy_config):
    _, y_train, _, y_test, _, y_val = split_data(
        dummy_data,
        dummy_config,
    )

    original_rate = dummy_data["default"].mean()

    for split in (y_train, y_test, y_val):
        assert abs(split.mean() - original_rate) < 0.01



