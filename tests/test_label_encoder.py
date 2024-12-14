
# tests/test_data_transform.py
import pandas as pd
import pytest

from air_pollution.data_pipeline.label_encoder import LabelEncodingTransformer


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

def test_label_encoding_single_class(sample_data: pd.DataFrame) -> None:
    """Tests the LabelEncodingTransformer on a single-class target column."""
    single_class_data = sample_data.copy()
    single_class_data["target"] = "Good"  # All values are the same

    label_encoder = LabelEncodingTransformer("target")
    transformed_data = label_encoder.fit_transform(single_class_data)

    assert set(transformed_data["target"].unique()) == {0}  # Single class should be encoded as 0
