# src/ml_data_pipeline/data_transform/minmax_scaler_transformer.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .base_transformer import DataTransformer


class MinMaxScalerTransformer(DataTransformer):
    """A transformer that scales data using Min-Max scaling."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms the input data using Min-Max scaling.

        Args:
            data (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data with values scaled between 0 and 1.
        """
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns)
