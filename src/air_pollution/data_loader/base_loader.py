# src/air_pollution/data_loader/base_loader.py
from abc import ABC, abstractmethod
import pandas as pd


class DataLoader(ABC):
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        pass
