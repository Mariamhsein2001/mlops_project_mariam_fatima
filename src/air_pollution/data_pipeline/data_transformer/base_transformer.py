# src/ml_data_pipeline/data_transform/base_transformer.py
from abc import ABC, abstractmethod

import pandas as pd


class DataTransformer(ABC):
    """Abstract base class for data transformers."""

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms the input data.

        Args:
            data (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        pass
