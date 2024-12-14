# src/ml_data_pipeline/data_transform/factory.py
from typing import Any

from .base_transformer import DataTransformer
from .minmax_scaler_transformer import MinMaxScalerTransformer
from .standard_scaler_transformer import StandardScalerTransformer


class TransformerFactory:
    """Factory class to create transformer instances based on the scaling method."""

    @staticmethod
    def get_transformer(
        transformer_type: str, *args: Any, **kwargs: Any
    ) -> DataTransformer:
        """
        Returns an instance of a transformer based on the provided type.

        Args:
            transformer_type (str): The type of transformer to create.
            *args, **kwargs: Additional arguments for the transformer initialization.

        Returns:
            DataTransformer: An instance of the requested transformer.

        Raises:
            ValueError: If the transformer type is unsupported.
        """
        if transformer_type == "standard":
            return StandardScalerTransformer()
        elif transformer_type == "minmax":
            return MinMaxScalerTransformer()

        else:
            raise ValueError(f"Unsupported transformer type: {transformer_type}")
