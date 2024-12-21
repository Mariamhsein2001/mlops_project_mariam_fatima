"""
Script for training the air quality prediction model.

This script executes the complete training pipeline, which includes:
- Loading configuration files.
- Ingesting and preprocessing data.
- Training a machine learning model.
- Evaluating the model on test data.
- Logging metrics and artifacts with MLflow.
- Registering the trained model in the MLflow Model Registry.

Functions:
    run_training_pipeline(config_path): Executes the end-to-end training pipeline.
    main(): Entry point for triggering the training pipeline via command-line arguments.
"""

import argparse
import os
import joblib
from loguru import logger
from air_pollution.config import load_config
from air_pollution.data_loader.factory import DataLoaderFactory
from air_pollution.data_pipeline.preprocessing import Preprocessor
from air_pollution.model.factory import ModelFactory
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict

# Set up logging
logger.add("logs/training.log", rotation="500 MB")

# Argument parser for command-line execution
parser = argparse.ArgumentParser(
    description="Run the Air Quality Pollution data pipeline training with specified configuration."
)
parser.add_argument(
    "--config", type=str, required=True, help="Path to the configuration YAML file."
)


def run_training_pipeline(config_path: str) -> Dict[str, float]:
    """
    Execute the ML data pipeline for training and evaluation.

    This function performs the following:
    - Loads configurations from a YAML file.
    - Loads and preprocesses data.
    - Trains a machine learning model.
    - Evaluates the model on test data.
    - Logs metrics and artifacts with MLflow.
    - Registers the trained model in the MLflow Model Registry.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        Dict[str, float]: A dictionary containing evaluation metrics such as accuracy and F1 score.

    Raises:
        Exception: If any error occurs during pipeline execution.
    """
    try:
        # Step 1: Load configuration
        logger.info(f"Loading configuration from {config_path}.")
        config = load_config(config_path)
        logger.info("Configuration successfully loaded.")

        # Step 2: Initialize MLflow
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
        mlflow.autolog()

        with mlflow.start_run():
            # Step 3: Load data
            logger.info(f"Loading data using {config.data_loader.file_type} loader.")
            data_loader = DataLoaderFactory.get_data_loader(config.data_loader.file_type)
            data = data_loader.load_data(config.data_loader.file_path)
            logger.info(f"Data successfully loaded. Shape: {data.shape}")

            # Log model parameters
            mlflow.log_param("model_type", config.model.type)
            mlflow.log_param("model_parameters", config.model.params)

            # Step 4: Preprocess the data
            target_column = "Air Quality"
            logger.info(f"Initializing preprocessor with target column: {target_column}")
            preprocessor = Preprocessor(config, target_column)
            X_train, X_test, y_train, y_test = preprocessor.preprocess(data)
            logger.info("Data preprocessing completed.")
            logger.debug(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

            # Step 5: Train the model
            logger.info(f"Initializing and training model of type: {config.model.type}")
            model = ModelFactory.get_model(config.model.type, config.model.params)
            model.train(X_train, y_train)
            logger.info("Model training completed.")

            # Step 6: Predict and evaluate
            logger.info("Generating predictions on the test set.")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            logger.info(f"Model evaluation completed. Accuracy: {accuracy}, F1-Score: {f1}")

            # Step 7: Save the trained model
            model_directory = "trained_model"
            os.makedirs(model_directory, exist_ok=True)
            model_path = os.path.join(model_directory, "trained_model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")

            # Step 8: Register model with MLflow
            active_run = mlflow.active_run()
            run_id = active_run.info.run_id if active_run else mlflow.start_run().info.run_id
            model_uri = f"runs:/{run_id}/model"
            model_name = "AirQualityPredictionModel"

            registered_model_version = mlflow.register_model(model_uri, model_name)
            client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)
            version_str = str(registered_model_version.version)
            client.set_registered_model_alias(model_name, "champion", version_str)
            logger.info(f"Model registered with alias 'champion': version {version_str}")

            return {"accuracy": accuracy, "f1_score": f1}

    except Exception:
        logger.exception("An error occurred during pipeline execution.")
        raise


def main() -> None:
    """
    Entry point for running the training pipeline.

    This function parses command-line arguments, loads the specified configuration,
    and triggers the training pipeline.
    """
    logger.info("Parsing command line arguments.")
    args = parser.parse_args()
    logger.debug(f"Command line arguments: {args}.")
    result = run_training_pipeline(args.config)
    print(result)


if __name__ == "__main__":
    main()
