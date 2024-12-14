# tests/test_preprocessing.py
import pandas as pd
import pytest
from air_pollution.data_pipeline.preprocessing import Preprocessor

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Creates sample data for testing."""
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "Air Quality": ["Good", "Moderate", "Good", "Poor", "Moderate", 
                        "Good", "Poor", "Good", "Moderate", "Poor"]
    })

@pytest.fixture
def sample_config():
    """Creates a sample configuration for testing without using an external Config object."""
    class DummyConfig:
        class splitting:
            test_size = 0.2
            
        class transformation:
            scaling_method = "standard"
    return DummyConfig()

def test_preprocessor(sample_config, sample_data):
    """Tests the Preprocessor class."""
    target_column = "Air Quality"
    preprocessor = Preprocessor(sample_config, target_column)

    # Preprocess the data and split into training and test sets
    X_train, X_test, y_train, y_test = preprocessor.preprocess(sample_data, target_column)

    # Check that the output is in the expected format
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # Check that the shapes match the expected splits
    assert X_train.shape[0] == int((1 - sample_config.splitting.test_size) * sample_data.shape[0])
    assert X_test.shape[0] == int(sample_config.splitting.test_size * sample_data.shape[0])
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]

    # Check that the target column is correctly encoded
    assert y_train.nunique() == 3  

    # Check that the data has been scaled (values should not be the same as the original)
    original_features = sample_data.drop(columns=[target_column]).values
    assert not (X_train.values == original_features[:len(X_train)]).all().all()
    assert not (X_test.values == original_features[len(X_train):]).all().all()
