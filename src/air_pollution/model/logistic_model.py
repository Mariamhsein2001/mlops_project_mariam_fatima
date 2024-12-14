# src/air_pollution/model/logistic_model.py
from typing import Any
import pandas as pd
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

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the logistic regression model on the provided data.

        Args:
            X (pd.DataFrame): The input features for training.
            y (pd.Series): The target values for training.
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts the target values using the logistic regression model.

        Args:
            X (pd.DataFrame): The input features for prediction.

        Returns:
            pd.Series: The predicted target values.
        """
        predictions = self.model.predict(X)

        return pd.Series(predictions, index=X.index, dtype="category")
