# tests/test_model.py
import pandas as pd
import pytest

from air_pollution.model import ModelFactory


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})


@pytest.fixture
def sample_target() -> pd.Series:
    return pd.Series([0, 1, 0])


def test_logistic_model(sample_data: pd.DataFrame, sample_target: pd.Series) -> None:
    model = ModelFactory.get_model("logistic")
    model.train(sample_data, sample_target)
    predictions = model.predict(sample_data)
    assert isinstance(predictions, pd.Series)
    assert predictions.shape[0] == sample_data.shape[0]


def test_decision_tree_model(sample_data: pd.DataFrame, sample_target: pd.Series) -> None:
    model = ModelFactory.get_model("decisiontree")
    model.train(sample_data, sample_target)
    predictions = model.predict(sample_data)
    assert isinstance(predictions, pd.Series)
    assert predictions.shape[0] == sample_data.shape[0]
