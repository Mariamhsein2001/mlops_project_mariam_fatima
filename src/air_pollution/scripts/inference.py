"""
Script for running inference on air quality prediction data.

This script provides functionality for:
- Parsing user input data in JSON or CSV-like formats.
- Loading a pre-trained machine learning model and its associated data transformation pipeline.
- Performing inference on input data and returning predictions as well as class labels.

Classes:
    InferencePipeline: A pipeline that handles data transformation and model predictions.

Functions:
    load_pipeline(transformation_config, model_path): Loads the pre-trained model and transformer pipeline.
    parse_args(): Parses command-line arguments.
    main(): Entry point for running inference through command-line input.
"""

import argparse
from typing import Dict, Any
import pandas as pd
from loguru import logger
import joblib  # For loading the saved model
from fastapi import HTTPException
from air_pollution.config import TransformationConfig
from air_pollution.data_pipeline.data_transformer.base_transformer import (
    DataTransformer,
)
from air_pollution.data_pipeline.data_transformer.factory import TransformerFactory
from air_pollution.model.base_model import Model
import json


logger.add("logs/inference.log", rotation="500 MB")
# Define the class labels
CLASS_LABELS = {
    0: "Good",
    1: "Moderate",
    2: "Hazardous",
    3: "Poor",
}


def load_pipeline(
    transformation_config: TransformationConfig, model_path: str
) -> "InferencePipeline":
    """
    Loads the model and transformer pipeline based on the provided configuration.

    Args:
        transformation_config (TransformationConfig): The configuration for the data transformer.
        model_path (str): The file path to the trained model.

    Returns:
        InferencePipeline: An instance of the InferencePipeline with a transformer and model loaded.

    Raises:
        FileNotFoundError: If the model file is not found at the given path.
        Exception: For other errors encountered during model loading.
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}.")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}. Please check the path.")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    transformer = TransformerFactory.get_transformer(
        transformation_config.scaling_method
    )

    return InferencePipeline(transformer, model)


class InferencePipeline:
    """
    A class that represents the inference pipeline, including data transformation and model prediction.

    Attributes:
        _data_transformer (DataTransformer): The data transformer for preprocessing input data.
        _model (Model): The trained model used for making predictions.

    Methods:
        run(data: pd.DataFrame) -> pd.Series:
            Runs the inference pipeline on the input data and returns predictions.
    """

    _data_transformer: DataTransformer
    _model: Model

    def __init__(self, data_transformer: DataTransformer, model: Model):
        """
        Initializes the inference pipeline with the given transformer and model.

        Args:
            data_transformer (DataTransformer): The data transformer.
            model (Model): The trained model.
        """
        self._data_transformer = data_transformer
        self._model = model

    def run(self, data: pd.DataFrame) -> pd.Series:
        """
        Runs the inference pipeline on the input data and returns the predictions.

        Args:
            data (pd.DataFrame): The input data for inference.

        Returns:
            pd.Series: The predictions made by the model.

        Raises:
            HTTPException: If any error occurs during pipeline execution.
        """
        try:
            logger.info("Pipeline execution started.")
            logger.info("Applying Data transformation.")
            transformed_data = self._data_transformer.transform(data)
            logger.debug(f"Transformed Data: {transformed_data.head()}")
            logger.info("Data transformed successfully.")

            logger.info("Running Inference.")
            predictions = self._model.predict(transformed_data)
            logger.debug(f"Predictions: {predictions}")
            logger.info("Model prediction completed successfully.")
            logger.info("Pipeline execution completed.")
            return predictions  # Return raw predictions for further processing

        except Exception as e:
            logger.error(f"Failed in Pipeline Execution: {e}")
            raise HTTPException(status_code=500, detail="Error during inference.")


def process_input(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts the input dictionary to a pandas DataFrame in the required feature format.

    Args:
        input_dict (Dict[str, Any]): The input data in dictionary format.

    Returns:
        pd.DataFrame: The input data converted into a DataFrame.

    Raises:
        HTTPException: If the input dictionary is missing any required features or is invalid.
    """
    try:
        feature_names = [
            "Temperature",
            "Humidity",
            "PM2.5",
            "PM10",
            "NO2",
            "SO2",
            "CO",
            "Proximity_to_Industrial_Areas",
            "Population_Density",
        ]

        # Check if all required features are present
        missing_features = [f for f in feature_names if f not in input_dict]
        if missing_features:
            raise ValueError(f"Missing features in input: {missing_features}")

        # Extract features in the correct order
        feature_values = [input_dict[f] for f in feature_names]
        data = pd.DataFrame([feature_values], columns=feature_names)
        logger.info(f"Input dictionary converted to DataFrame: {data.head()}")
        return data

    except Exception as e:
        logger.error(f"Error processing input: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input format: {e}")


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for input data.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run inference on input data.")
    parser.add_argument(
        "--data",
        type=str,
        help='Inference data in JSON string format (e.g., \'{"Temperature": 28.3, "Humidity": 75.6, ...}\').',
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a JSON file containing inference data.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter inference data interactively.",
    )
    return parser.parse_args()


def get_input_data(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Retrieves input data based on the provided method (command-line argument, file, or interactive).

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        Dict[str, Any]: The input data as a dictionary.

    Raises:
        ValueError: If no valid input method is provided or if the input data is not a dictionary.
        json.JSONDecodeError: If the JSON data is invalid.
    """
    try:
        if args.data:
            # Parse JSON string
            logger.info("Using data from --data argument.")
            input_data = json.loads(args.data)
            if not isinstance(input_data, dict):
                raise ValueError("Input data must be a dictionary.")
            return input_data

        elif args.file:
            # Read data from JSON file
            logger.info(f"Reading data from file: {args.file}")
            with open(args.file, "r") as f:
                input_data = json.load(f)
                if not isinstance(input_data, dict):
                    raise ValueError("Input data from file must be a dictionary.")
                return input_data

        elif args.interactive:
            # Prompt the user for input interactively
            logger.info("Entering interactive mode. Please provide the required data.")
            input_dict = {}
            for feature in [
                "Temperature",
                "Humidity",
                "PM2.5",
                "PM10",
                "NO2",
                "SO2",
                "CO",
                "Proximity_to_Industrial_Areas",
                "Population_Density",
            ]:
                value = input(f"Enter value for {feature}: ")
                input_dict[feature] = float(value)  # Convert to float for processing
            return input_dict

        else:
            raise ValueError(
                "No input method specified. Use --data, --file, or --interactive."
            )

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        raise
    except Exception as e:
        logger.error(f"Error retrieving input data: {e}")
        raise


def main() -> None:
    """
    The main function to execute the inference pipeline.

    Retrieves input data, processes it, runs inference using a trained model, and outputs the predictions.
    """
    args = parse_args()

    try:
        # Retrieve input data based on the chosen method
        input_dict = get_input_data(args)

        # Validate the input data
        if not isinstance(input_dict, dict):
            raise ValueError("Input data must be a dictionary.")

        # Convert input to DataFrame
        data = process_input(input_dict)

        # Setup transformation configuration
        transformation_config = TransformationConfig(
            scaling_method="minmax", normalize=False
        )
        model_path = "trained_model/trained_model.pkl"

        # Load the pipeline
        pipeline = load_pipeline(transformation_config, model_path)

        # Run inference and get raw predictions
        predictions = pipeline.run(data)
        logger.info(f"Inference completed. Predictions: {predictions}")

        # Map predictions to class labels
        class_labels = {
            "predictions": [
                CLASS_LABELS.get(int(pred), "Unknown") for pred in predictions
            ]
        }
        logger.info(f"Predicted class labels: {class_labels}")

        # Print predictions
        print(json.dumps(class_labels))  # Convert output to JSON for readability

    except Exception as e:
        logger.error(f"Error during inference: {e}")


if __name__ == "__main__":
    main()
