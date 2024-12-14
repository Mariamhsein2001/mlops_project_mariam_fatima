# src/air_pollution/data_pipeline/label_encoder.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class LabelEncodingTransformer:
    """Encodes a specified column using LabelEncoder."""

    def __init__(self, column: str):
        """
        Initializes the LabelEncoder for a specific column.

        Args:
            column (str): The column to encode.
        """
        self.column = column
        self.encoder = LabelEncoder()

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fits the LabelEncoder and transforms the specified column."""
        data[self.column] = self.encoder.fit_transform(data[self.column])
        return data
