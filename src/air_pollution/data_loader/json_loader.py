import json

import pandas as pd
from loguru import logger

from .base_loader import DataLoader


class JSONLoader(DataLoader):
    """Concrete implementation of DataLoader for JSON files."""

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads data from a JSON file and normalizes it into a pandas DataFrame.

        Args:
            file_path (str): Path to the JSON file to be loaded.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the normalized JSON data.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file is not a valid JSON file.
        """
        try:
            logger.info(f"Attempting to load JSON file from path: {file_path}")
            with open(file_path, "r") as file:
                data = json.load(file)
            logger.info("JSON file successfully loaded and normalized.")
            return pd.json_normalize(data)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            logger.exception("FileNotFoundError encountered.")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in file: {file_path}")
            logger.exception("JSONDecodeError encountered.")
            raise
