import pandas as pd
import json
from .base_loader import DataLoader


class JSONLoader(DataLoader):
    def load_data(self, file_path: str) -> pd.DataFrame:
        with open(file_path, "r") as file:
            data = json.load(file)
        return pd.json_normalize(data)
