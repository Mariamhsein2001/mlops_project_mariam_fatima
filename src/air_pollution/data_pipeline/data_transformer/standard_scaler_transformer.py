import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

from .base_transformer import DataTransformer


class StandardScalerTransformer(DataTransformer):
    """A transformer that scales data using Standard scaling (z-score normalization)."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input data using Standard scaling.

        Standard scaling (z-score normalization) centers the data by subtracting the mean
        and scales it to unit variance, ensuring that each feature has a mean of 0 and a standard deviation of 1.

        Args:
            data (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data with standardized values.

        Raises:
            ValueError: If the input data is not a DataFrame or is empty.
        """
        logger.info("Starting Standard scaling transformation.")

        # Input validation
        if data.empty:
            logger.error("Input data is empty. Transformation cannot proceed.")
            raise ValueError("Input data cannot be empty.")

        # Apply Standard scaling
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        logger.info("Standard scaling transformation completed.")
        return pd.DataFrame(scaled_data, columns=data.columns)
