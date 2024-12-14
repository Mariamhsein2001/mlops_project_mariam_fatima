from typing import Any, Tuple

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from air_pollution.data_pipeline.data_transformer.factory import TransformerFactory
from air_pollution.data_pipeline.label_encoder import LabelEncodingTransformer


class Preprocessor:
    """Handles data preprocessing, including scaling, encoding, and splitting."""

    def __init__(self, config: Any, target_column: str) -> None:
        """
        Initializes the Preprocessor with configuration.

        Args:
            config (Any): Configuration object containing preprocessing details.
            target_column (str): The column to be used as the target variable.
        """
        self.config = config
        self.target_column = target_column
        logger.info(f"Initializing Preprocessor with target column: {target_column}")

        # Initialize the scaling transformer using the factory
        self.scaler = TransformerFactory.get_transformer(
            config.transformation.scaling_method
        )
        self.label_encoder = LabelEncodingTransformer(target_column)

    def preprocess(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Applies preprocessing steps to the data, including label encoding, scaling, and splitting.

        Args:
            data (pd.DataFrame): The input data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Preprocessed training and test data splits:
                X_train, X_test, y_train, y_test.

        Raises:
            ValueError: If the input data is empty or the target column is missing.
        """
        logger.info("Starting preprocessing pipeline.")

        # Input validation
        if data.empty:
            logger.error("Input data is empty. Preprocessing cannot proceed.")
            raise ValueError("Input data cannot be empty.")
        if self.target_column not in data.columns:
            logger.error(
                f"Target column '{self.target_column}' not found in the input data."
            )
            raise ValueError(
                f"Target column '{self.target_column}' does not exist in the input data."
            )

        # Apply label encoding to the target column
        logger.info("Applying label encoding to the target column.")
        data = self.label_encoder.fit_transform(data)

        # Split the data into training and test sets
        logger.info("Splitting the data into training and test sets.")
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.splitting.test_size
        )

        # Fit and transform the scaler on the training data
        logger.info("Scaling the training data.")
        X_train = self.scaler.transform(X_train)

        # Transform the test data using the fitted scaler
        logger.info("Scaling the test data.")
        X_test = self.scaler.transform(X_test)

        logger.info("Preprocessing pipeline completed successfully.")
        return X_train, X_test, y_train, y_test
