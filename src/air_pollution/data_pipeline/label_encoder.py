import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder


class LabelEncodingTransformer:
    """Encodes a specified column using LabelEncoder."""

    def __init__(self, column: str) -> None:
        """
        Initializes the LabelEncoder for a specific column.

        Args:
            column (str): The column to encode.
        """
        self.column = column
        self.encoder = LabelEncoder()
        logger.info(f"Initialized LabelEncodingTransformer for column: {self.column}")

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the LabelEncoder and transforms the specified column.

        The values in the specified column are converted into integer labels
        ranging from 0 to n_classes-1.

        Args:
            data (pd.DataFrame): The input data containing the column to encode.

        Returns:
            pd.DataFrame: The transformed data with the specified column encoded.

        Raises:
            ValueError: If the input data is empty or the specified column does not exist.
        """
        logger.info("Starting label encoding transformation.")

        # Input validation
        if data.empty:
            logger.error("Input data is empty. Transformation cannot proceed.")
            raise ValueError("Input data cannot be empty.")
        if self.column not in data.columns:
            logger.error(f"Column '{self.column}' not found in the input data.")
            raise ValueError(
                f"Column '{self.column}' does not exist in the input data."
            )

        # Apply label encoding
        data[self.column] = self.encoder.fit_transform(data[self.column])
        logger.info(
            f"Label encoding transformation completed for column: {self.column}."
        )
        return data
