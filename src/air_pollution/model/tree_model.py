# src/air_pollution/model/tree_model.py
from typing import Any

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .base_model import Model


class DecisionTreeModel(Model):
    """A decision tree model for training and prediction."""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """
        Initializes the DecisionTreeModel with the given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the DecisionTreeClassifier.
        """
        self.model = DecisionTreeClassifier(**kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the decision tree model on the provided data.

        Args:
            X (pd.DataFrame): The input features for training.
            y (pd.Series): The target values for training.
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts the target values using the decision tree model.

        Args:
            X (pd.DataFrame): The input features for prediction.

        Returns:
            pd.Series: The predicted target values.
        """
        predictions = self.model.predict(X)

        return pd.Series(predictions, index=X.index, dtype="category")