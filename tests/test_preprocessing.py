import pandas as pd
import pytest

from air_pollution.data_pipeline.preprocessing import Preprocessor


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Creates sample data for testing.

    Returns:
        pd.DataFrame: A DataFrame containing sample features and a target column.
    """
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "Air Quality": [
                "Good",
                "Moderate",
                "Good",
                "Poor",
                "Moderate",
                "Good",
                "Poor",
                "Good",
                "Moderate",
                "Poor",
            ],
        }
    )


@pytest.fixture
def sample_config():
    """
    Creates a sample configuration object for testing.

    Returns:
        DummyConfig: A simplified configuration object mimicking the actual config.
    """

    class DummyConfig:
        class splitting:
            test_size = 0.2

        class transformation:
            scaling_method = "standard"

    return DummyConfig()


def test_preprocessor(sample_config, sample_data):
    """
    Tests the Preprocessor class for correct functionality.

    Args:
        sample_config: A simplified configuration object for testing.
        sample_data (pd.DataFrame): Sample input data provided by the fixture.

    Assertions:
        - Checks that the output types are correct.
        - Ensures the data is split into training and test sets with expected shapes.
        - Verifies that the target column is correctly encoded.
        - Confirms that the feature data has been scaled.
    """
    target_column = "Air Quality"
    preprocessor = Preprocessor(sample_config, target_column)

    # Preprocess the data and split into training and test sets
    X_train, X_test, y_train, y_test = preprocessor.preprocess(sample_data)

    # Check that the output is in the expected format
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame."
    assert isinstance(X_test, pd.DataFrame), "X_test should be a DataFrame."
    assert isinstance(y_train, pd.Series), "y_train should be a Series."
    assert isinstance(y_test, pd.Series), "y_test should be a Series."

    # Check that the shapes match the expected splits
    expected_train_size = int(
        (1 - sample_config.splitting.test_size) * sample_data.shape[0]
    )
    expected_test_size = int(sample_config.splitting.test_size * sample_data.shape[0])

    assert X_train.shape[0] == expected_train_size, "X_train size mismatch."
    assert X_test.shape[0] == expected_test_size, "X_test size mismatch."
    assert y_train.shape[0] == expected_train_size, "y_train size mismatch."
    assert y_test.shape[0] == expected_test_size, "y_test size mismatch."

    # Check that the target column is correctly encoded
    assert y_train.nunique() == 3, "Target column encoding mismatch."

    # Check that the data has been scaled (values should not match original values)
    original_features = sample_data.drop(columns=[target_column]).values
    assert (
        not (X_train.values == original_features[: len(X_train)]).all().all()
    ), "X_train should be scaled."
    assert (
        not (X_test.values == original_features[len(X_train) :]).all().all()
    ), "X_test should be scaled."
