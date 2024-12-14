from __future__ import annotations

import pandas as pd
from loguru import logger
import joblib  # For loading the saved model

from air_pollution.config import ModelConfig, TransformationConfig
from air_pollution.data_pipeline.data_transformer.base_transformer import DataTransformer
from air_pollution.data_pipeline.data_transformer.factory import TransformerFactory
from air_pollution.model.base_model import Model


def load_pipeline(
    transformation_config: TransformationConfig
) -> "InferencePipeline":
    data_transformer = TransformerFactory.get_transformer(
        transformation_config.scaling_method
    )
    
    # Load the model from the saved path
    model_path = "trained_model.pkl"  # Ensure this path is defined in your config file
    model = joblib.load(model_path)
    
    return InferencePipeline(data_transformer, model)


class InferencePipeline:
    _data_transformer: DataTransformer
    _model: Model

    def __init__(self, data_transformer: DataTransformer, model: Model):
        self._data_transformer = data_transformer
        self._model = model

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Runs the inference data pipeline on the input data.

        Args:
            data (pd.DataFrame): The input data to process.

        Returns:
            pd.DataFrame: The processed data.
        """
        try:
            logger.info("Pipeline execution started.")

            logger.info("Applying Data transformation.")
            transformed_data = self._data_transformer.transform(data)
            logger.debug(f"Data: {transformed_data.head()}")
            logger.info("Data transformed successfully.")

            logger.info("Running Inference.")
            predictions = self._model.predict(transformed_data)
            logger.debug(f"Predictions: {predictions.head()}")
            logger.info("Model prediction completed successfully.")

            logger.info("Pipeline execution completed.")
            return predictions

        except Exception as e:
            logger.error(f"Failed in Pipeline Execution: {e}")
            raise
