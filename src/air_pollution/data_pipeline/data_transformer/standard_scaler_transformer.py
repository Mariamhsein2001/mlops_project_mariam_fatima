# src/ml_data_pipeline/data_transform/standard_scaler_transformer.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base_transformer import DataTransformer


class StandardScalerTransformer(DataTransformer):
    """A transformer that scales data using Standard scaling (z-score normalization)."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms the input data using Standard scaling.

        Args:
            data (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data with standardized values.
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns)
