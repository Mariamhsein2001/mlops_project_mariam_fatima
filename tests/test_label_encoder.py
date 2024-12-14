import pandas as pd
import pytest

from air_pollution.data_pipeline.label_encoder import LabelEncodingTransformer


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Fixture to provide sample data for testing.

    Returns:
        pd.DataFrame: A DataFrame with sample numerical data for transformation.
    """
    return pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})


def test_label_encoding_single_class(sample_data: pd.DataFrame) -> None:
    """
    Test the LabelEncodingTransformer on a single-class target column.

    Args:
        sample_data (pd.DataFrame): Sample data provided by the fixture.

    Assertions:
        - The transformed target column contains only a single unique value (0).
    """
    # Create single-class target column
    single_class_data = sample_data.copy()
    single_class_data["target"] = "Good"  # All values are the same

    # Initialize and apply the LabelEncodingTransformer
    label_encoder = LabelEncodingTransformer("target")
    transformed_data = label_encoder.fit_transform(single_class_data)

    # Validate that the target column contains only a single encoded class (0)
    assert set(transformed_data["target"].unique()) == {
        0
    }, "Single class should be encoded as 0."
