import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

from .base_transformer import DataTransformer


class MinMaxScalerTransformer(DataTransformer):
    """A transformer that scales data using Min-Max scaling."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input data using Min-Max scaling.

        Min-Max scaling scales each feature to a given range (default: 0 to 1),
        preserving the relationships between values but normalizing them to a common scale.

        Args:
            data (pd.DataFrame): The input data to transform. Must contain only numeric columns.

        Returns:
            pd.DataFrame: The transformed data with values scaled between 0 and 1.

        Raises:
            ValueError: If the input data is not a DataFrame, is empty, or contains non-numeric columns.
        """
        logger.info("Starting Min-Max scaling transformation.")

        # Input validation
        if data.empty:
            logger.error("Input data is empty. Transformation cannot proceed.")
            raise ValueError("Input data cannot be empty.")

        if not all(data.dtypes.apply(pd.api.types.is_numeric_dtype)):
            logger.error("Input data contains non-numeric columns.")
            raise ValueError("All columns in the input data must be numeric.")

        logger.debug(f"Input data shape: {data.shape}")

        # Apply Min-Max scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        logger.info("Min-Max scaling transformation completed.")
        logger.debug(f"Transformed data shape: {scaled_data.shape}")

        return pd.DataFrame(scaled_data, columns=data.columns)
