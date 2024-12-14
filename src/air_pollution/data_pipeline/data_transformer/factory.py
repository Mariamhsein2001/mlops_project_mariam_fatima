# src/ml_data_pipeline/data_transform/factory.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder , StandardScaler
from .base_transformer import DataTransformer
from .standard_scaler_transformer import StandardScalerTransformer
from .minmax_scaler_transformer import MinMaxScalerTransformer


class TransformerFactory:
    """Factory class to create transformer instances based on the scaling method."""

    @staticmethod
    def get_transformer(transformer_type: str, *args, **kwargs) -> DataTransformer:
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

    @staticmethod
    def preprocess(data: pd.DataFrame, scaling_method: str = "standard", target_column: str = "Air Quality") -> pd.DataFrame:
        """
        Preprocesses the data by scaling numerical features and encoding the target column.

        Args:
            data (pd.DataFrame): The input data to preprocess.
            scaling_method (str): The scaling method to use ("standard" or "minmax").
            target_column (str): The column to encode as a target.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        # Select numerical columns for scaling
        numerical_columns = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
        # Ensure the numerical columns contain only numeric data
        numerical_data = data[numerical_columns].apply(pd.to_numeric, errors='coerce')
        numerical_data = numerical_data.dropna()

        # Create and apply the scaler
        # scaled_df = TransformerFactory.get_transformer(scaling_method, numerical_columns)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numerical_data)
        scaled_df = pd.DataFrame(scaled_data, columns=numerical_columns)
        print(scaled_df)
        # Create and apply the label encoder for the target column
        encoder = LabelEncoder()
        encoded_target = encoder.fit_transform(data[target_column])
        encoded_df = pd.DataFrame(encoded_target, columns=[target_column])

        # Combine scaled numerical data and encoded target column
        preprocessed_data = pd.concat([scaled_df, encoded_df], axis=1)
        return preprocessed_data
