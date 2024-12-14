from typing import Any

import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression

from .base_model import Model


class LogisticModel(Model):
    """A logistic regression model for training and prediction."""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """
        Initializes the LogisticModel with the given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the LogisticRegression.
        """
        self.model = LogisticRegression(**kwargs)
        logger.info(f"Initialized LogisticModel with parameters: {kwargs}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the logistic regression model on the provided data.

        Args:
            X (pd.DataFrame): The input features for training.
            y (pd.Series): The target values for training.

        Raises:
            ValueError: If the input data is empty or if the target values are missing.
        """
        if X.empty or y.empty:
            logger.error(
                "Training data or target values are empty. Training cannot proceed."
            )
            raise ValueError("Training data and target values cannot be empty.")

        logger.info("Starting training of LogisticModel.")
        self.model.fit(X, y)
        logger.info("Training of LogisticModel completed successfully.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts the target values using the logistic regression model.

        Args:
            X (pd.DataFrame): The input features for prediction.

        Returns:
            pd.Series: The predicted target values.

        Raises:
            ValueError: If the input features are empty.
        """
        if X.empty:
            logger.error(
                "Input features for prediction are empty. Prediction cannot proceed."
            )
            raise ValueError("Input features for prediction cannot be empty.")

        logger.info("Starting prediction using LogisticModel.")
        predictions = self.model.predict(X)
        logger.info("Prediction using LogisticModel completed successfully.")

        return pd.Series(predictions, index=X.index, dtype="category")
