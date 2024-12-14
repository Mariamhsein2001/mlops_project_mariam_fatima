import pandas as pd
import pytest

from air_pollution.model import ModelFactory


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Fixture to provide sample feature data for testing.

    Returns:
        pd.DataFrame: A DataFrame containing sample feature data.
    """
    return pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})


@pytest.fixture
def sample_target() -> pd.Series:
    """
    Fixture to provide sample target labels for testing.

    Returns:
        pd.Series: A Series containing sample target labels.
    """
    return pd.Series([0, 1, 0])


def test_logistic_model(sample_data: pd.DataFrame, sample_target: pd.Series) -> None:
    """
    Test the Logistic Regression model.

    Args:
        sample_data (pd.DataFrame): Sample feature data provided by the fixture.
        sample_target (pd.Series): Sample target labels provided by the fixture.

    Assertions:
        - Predictions are returned as a pandas Series.
        - The number of predictions matches the number of input samples.
    """
    model = ModelFactory.get_model("logistic")
    model.train(sample_data, sample_target)
    predictions = model.predict(sample_data)

    assert isinstance(predictions, pd.Series), "Predictions should be a pandas Series."
    assert (
        predictions.shape[0] == sample_data.shape[0]
    ), "Number of predictions should match the number of input samples."


def test_decision_tree_model(
    sample_data: pd.DataFrame, sample_target: pd.Series
) -> None:
    """
    Test the Decision Tree model.

    Args:
        sample_data (pd.DataFrame): Sample feature data provided by the fixture.
        sample_target (pd.Series): Sample target labels provided by the fixture.

    Assertions:
        - Predictions are returned as a pandas Series.
        - The number of predictions matches the number of input samples.
    """
    model = ModelFactory.get_model("decisiontree")
    model.train(sample_data, sample_target)
    predictions = model.predict(sample_data)

    assert isinstance(predictions, pd.Series), "Predictions should be a pandas Series."
    assert (
        predictions.shape[0] == sample_data.shape[0]
    ), "Number of predictions should match the number of input samples."
