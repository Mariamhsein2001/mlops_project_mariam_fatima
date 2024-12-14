from .csv_loader import CSVLoader
from .json_loader import JSONLoader
from .base_loader import DataLoader
from typing import Union


class DataLoaderFactory:
    """Factory class for creating DataLoader instances."""

    @staticmethod
    def get_data_loader(file_type: str) -> Union[CSVLoader, JSONLoader]:
        """Returns the appropriate DataLoader based on the file type.

        Args:
            file_type (str): The type of file to load. Supported values are:
                "csv" and "json".

        Returns:
            Union[CSVLoader, JSONLoader]: An instance of CSVLoader or JSONLoader.

        Raises:
            ValueError: If the provided file type is not supported.
        """
        if file_type == "csv":
            return CSVLoader()
        elif file_type == "json":
            return JSONLoader()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
