import pandas as pd
import pytest

from air_pollution.data_pipeline.data_transformer import TransformerFactory


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Fixture to provide sample data for testing.

    Returns:
        pd.DataFrame: A DataFrame with sample numerical data for transformation.
    """
    return pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})


def test_standard_scaler_transform(sample_data: pd.DataFrame) -> None:
    """
    Test the Standard Scaler transformer to ensure it scales data correctly.

    Args:
        sample_data (pd.DataFrame): Sample data provided by the fixture.

    Assertions:
        - The transformed data is a pandas DataFrame.
        - The shape of the transformed data matches the original data's shape.
    """
    transformer = TransformerFactory.get_transformer("standard")
    transformed_data = transformer.transform(sample_data)

    assert isinstance(
        transformed_data, pd.DataFrame
    ), "Transformed data should be a pandas DataFrame."
    assert (
        transformed_data.shape == sample_data.shape
    ), "Shape of transformed data does not match original data."


def test_minmax_scaler_transform(sample_data: pd.DataFrame) -> None:
    """
    Test the Min-Max Scaler transformer to ensure it scales data correctly.

    Args:
        sample_data (pd.DataFrame): Sample data provided by the fixture.

    Assertions:
        - The transformed data is a pandas DataFrame.
        - The shape of the transformed data matches the original data's shape.
    """
    transformer = TransformerFactory.get_transformer("minmax")
    transformed_data = transformer.transform(sample_data)

    assert isinstance(
        transformed_data, pd.DataFrame
    ), "Transformed data should be a pandas DataFrame."
    assert (
        transformed_data.shape == sample_data.shape
    ), "Shape of transformed data does not match original data."
