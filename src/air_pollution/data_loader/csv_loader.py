# src/air_pollution/data_loader/csv_loader.py
import pandas as pd
from .base_loader import DataLoader
from loguru import logger

class CSVLoader(DataLoader):
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads CSV data from the specified file path.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: The data loaded into a DataFrame.
        """
        logger.info(f"Loading data from CSV file at {file_path}")
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise