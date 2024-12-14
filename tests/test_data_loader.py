from pathlib import Path

import pandas as pd
import pytest

from air_pollution.data_loader import DataLoaderFactory


@pytest.fixture
def sample_csv(tmp_path: Path) -> str:
    """
    Fixture to create a temporary CSV file for testing.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Returns:
        str: Path to the created sample CSV file.
    """
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("feature1,feature2,target\n1,4,0\n2,5,1\n3,6,0")
    return str(csv_file)


def test_csv_loader(sample_csv: str) -> None:
    """
    Test the CSV loader to ensure it loads data correctly.

    Args:
        sample_csv (str): Path to the sample CSV file created by the fixture.

    Assertions:
        - The loaded data is a pandas DataFrame.
        - The shape of the loaded DataFrame matches the expected dimensions.
    """
    loader = DataLoaderFactory.get_data_loader("csv")
    data = loader.load_data(sample_csv)

    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame."
    assert data.shape == (3, 3), "Data shape mismatch."
