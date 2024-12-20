from __future__ import annotations


import pandas as pd
from loguru import logger
import joblib  # For loading the saved model

from air_pollution.config import TransformationConfig
from air_pollution.data_pipeline.data_transformer.base_transformer import (
    DataTransformer,
)
from air_pollution.data_pipeline.data_transformer.factory import TransformerFactory
from air_pollution.model.base_model import Model
from fastapi import HTTPException


def load_pipeline(
    transformation_config: TransformationConfig, model_path: str
) -> "InferencePipeline":
    # Load the model from the saved path
    model_path = model_path

    # Handle potential errors in loading the model
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}.")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}. Please check the path.")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    # Initialize the data transformer (e.g., MinMaxScaler)
    # Assuming `TransformationConfig` defines a way to select the transformer
    transformer = TransformerFactory.get_transformer(
        transformation_config.scaling_method
    )

    return InferencePipeline(transformer, model)


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
            pd.DataFrame: The processed data (predictions).
        """
        try:
            logger.info("Pipeline execution started.")

            # Apply data transformation
            logger.info("Applying Data transformation.")
            transformed_data = self._data_transformer.transform(data)
            logger.debug(f"Transformed Data: {transformed_data.head()}")
            logger.info("Data transformed successfully.")

            # Run inference using the model
            logger.info("Running Inference.")
            predictions = self._model.predict(transformed_data)
            logger.debug(f"Predictions: {predictions.head()}")
            logger.info("Model prediction completed successfully.")

            logger.info("Pipeline execution completed.")
            return predictions

        except Exception as e:
            logger.error(f"Failed in Pipeline Execution: {e}")
            raise HTTPException(status_code=500, detail="Error during inference.")
