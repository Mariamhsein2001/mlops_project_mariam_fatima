# src/air_pollution/model/base_model.py
from abc import ABC, abstractmethod

import pandas as pd


class Model(ABC):
    """Abstract base class for machine learning models.

    This class defines the blueprint for machine learning models, which includes methods
    for training and predicting. Subclasses must implement the `train` and `predict`
    methods. The `train` method trains the model on the given data, while the `predict`
    method makes predictions using the trained model.
    """

    @abstractmethod
    def train(self, data: pd.DataFrame, labels: pd.Series) -> None:
        """Train the model using the provided data and labels.

        Args:
            data (pd.DataFrame): The input data for training the model.
            labels (pd.Series): The target labels for training.

        Returns:
            None: This method trains the model in-place.
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict using the trained model.

        Args:
            data (pd.DataFrame): A pandas DataFrame containing the data for prediction.

        Returns:
            pd.Series: A Series containing the model predictions.
        """
        pass
