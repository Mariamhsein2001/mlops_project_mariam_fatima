# src/air_pollution/data_pipeline/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from air_pollution.data_pipeline.data_transformer.factory import TransformerFactory
from air_pollution.data_pipeline.label_encoder import LabelEncodingTransformer

class Preprocessor:
    """Handles data preprocessing, including scaling, encoding, and splitting."""

    def __init__(self, config, target_column: str):
        """
        Initializes the Preprocessor with configuration.

        Args:
            config (Config): Configuration object containing preprocessing details.
            target_column (str): The column to be used as the target variable.
        """
        self.config = config
        # Initialize the scaling transformer using the factory
        self.scaler = TransformerFactory.get_transformer(config.transformation.scaling_method)
        self.label_encoder = LabelEncodingTransformer(target_column)

    def preprocess(self, data: pd.DataFrame, target_column: str) -> tuple:
        """
        Applies preprocessing steps to the data, including label encoding, scaling, and splitting.

        Args:
            data (pd.DataFrame): The input data.
            target_column (str): The column to be label encoded.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Apply label encoding to the target column
        data = self.label_encoder.fit_transform(data)

        # Split the data into training and test sets
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.splitting.test_size,
            random_state=42
        )

        # Fit and transform the scaler on the training data
        X_train = self.scaler.transform(X_train)

        # Transform the test data using the fitted scaler
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
