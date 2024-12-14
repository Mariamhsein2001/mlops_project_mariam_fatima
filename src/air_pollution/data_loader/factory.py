from typing import Union

from loguru import logger

from .base_loader import DataLoader
from .csv_loader import CSVLoader
from .json_loader import JSONLoader


class DataLoaderFactory:
    """Factory class for creating DataLoader instances."""

    @staticmethod
    def get_data_loader(file_type: str) -> Union[DataLoader, CSVLoader, JSONLoader]:
        """
        Returns the appropriate DataLoader based on the file type.

        Args:
            file_type (str): The type of file to load. Supported values are:
                "csv" and "json".

        Returns:
            Union[DataLoader, CSVLoader, JSONLoader]:
                An instance of CSVLoader or JSONLoader.

        Raises:
            ValueError: If the provided file type is not supported.
        """
        try:
            if file_type == "csv":
                logger.info("Creating a CSVLoader instance.")
                return CSVLoader()
            elif file_type == "json":
                logger.info("Creating a JSONLoader instance.")
                return JSONLoader()
            else:
                logger.error(f"Unsupported file type provided: {file_type}")
                raise ValueError(f"Unsupported file type: {file_type}")
        except ValueError:
            logger.exception("Error occurred while creating DataLoader.")
            raise
