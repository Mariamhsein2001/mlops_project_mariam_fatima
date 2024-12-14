# src/air_pollution/model/factory.py
from .base_model import Model
from .logistic_model import LogisticModel
from .tree_model import DecisionTreeModel


class ModelFactory:
    """Factory class to create model instances based on the model type."""

    @staticmethod
    def get_model(model_type: str) -> Model:
        """Returns an instance of a model based on the provided model type.

        Args:
            model_type (str): The type of model to create. Options are "logistic" or "decisiontree".

        Returns:
            Model: An instance of the requested model type.

        Raises:
            ValueError: If the provided model type is unsupported.
        """
        if model_type == "logistic":
            return LogisticModel()
        elif model_type == "decisiontree":
            return DecisionTreeModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")