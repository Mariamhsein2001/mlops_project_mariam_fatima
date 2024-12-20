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

logger.add("logs/training.log", rotation="500 MB")
parser = argparse.ArgumentParser(
    description="Run the Air Quality Pollution data pipeline training with specified configuration."
)
parser.add_argument(
    "--config", type=str, required=True, help="Path to the configuration YAML file."
)


def run_training_pipeline(config_path: str) -> Dict[str, float]:
    """
    Execute the ML data pipeline for training and evaluation.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: A dictionary containing the confusion matrix and classification report.

    Raises:
        Exception: If any error occurs during the pipeline execution.
    """
    try:
        # Step 1: Load configuration
        logger.info(f"Loading configuration from {config_path}.")
        config = load_config(config_path)
        print(config.mlflow)
        logger.info("Configuration successfully loaded.")

        # Initialize MLflow
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
        mlflow.autolog()

        with mlflow.start_run():
            # Step 2: Load data
            logger.info(f"Loading data using {config.data_loader.file_type} loader.")
            data_loader = DataLoaderFactory.get_data_loader(
                config.data_loader.file_type
            )
            data = data_loader.load_data(config.data_loader.file_path)
            logger.info(f"Data successfully loaded. Shape: {data.shape}")

            # Log model parameters
            mlflow.log_param("model_type", config.model.type)
            mlflow.log_param("model_parameters", config.model.params)
            # Step 3: Preprocess the data
            target_column = "Air Quality"
            logger.info(
                f"Initializing preprocessor with target column: {target_column}"
            )
            preprocessor = Preprocessor(config, target_column)
            X_train, X_test, y_train, y_test = preprocessor.preprocess(data)
            logger.info("Data preprocessing completed.")
            logger.debug(
                f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}"
            )

            # Step 4: Train the model
            logger.info(f"Initializing and training model of type: {config.model.type}")
            model = ModelFactory.get_model(config.model.type, config.model.params)
            model.train(X_train, y_train)
            logger.info("Model training completed.")

            # Step 5: Predict and evaluate
            logger.info("Generating predictions on the test set.")
            y_pred = model.predict(X_test)
            logger.debug(f"Predictions: {y_pred}")

            # Step 6: Evaluate the model
            logger.info("Model Evaluation")
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            f1 = f1_score(y_test, y_pred, average="weighted")
            mlflow.log_metric("f1_score", f1)
            logger.debug(f"Accuracy :  {accuracy} , F1-Score : {f1}")

            # Log model artifact
            input_example = X_test[
                0:1
            ]  # Take the first sample from the test set as an example
            mlflow.sklearn.log_model(
                model, artifact_path="model", input_example=input_example
            )

            # Step 7: Save the trained model to a directory
            model_directory = "trained_model"
            os.makedirs(
                model_directory, exist_ok=True
            )  # Create directory if it doesn't exist
            model_path = os.path.join(model_directory, "trained_model.pkl")
            joblib.dump(model, model_path)  # Save the model using joblib
            logger.info(f"Model saved to {model_path}")

            # Step 8: Register model to MLflow Model Registry
            # Check if there's an active run
            active_run = mlflow.active_run()

            if active_run:
                # If an active run exists, access the run ID
                run_id = active_run.info.run_id
            else:
                # Handle case where there's no active run
                with mlflow.start_run() as run:
                    run_id = run.info.run_id

            # Now you can safely work with run_id
            model_uri = f"runs:/{run_id}/model"
            model_name = "AirQualityPredictionModel"

            # Register the model to the Model Registry
            registered_model_version = mlflow.register_model(model_uri, model_name)
            logger.info(
                f"Model registered: {model_name} (version: {registered_model_version.version})"
            )

            client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)

            # Set alias 'champion' for the registered model version
            version_str = str(
                registered_model_version.version
            )  # Convert version to string
            client.set_registered_model_alias(model_name, "champion", version_str)
            logger.info(
                f"Alias 'champion' set for version {version_str} of model '{model_name}'."
            )

            # Get the model version by alias
            model_version = client.get_model_version_by_alias(model_name, "champion")
            logger.info(
                f"Retrieved model version using alias 'champion': {model_version.version}"
            )

            return {"accuracy": accuracy, "f1_score": f1}
    except Exception:
        logger.exception("An error occurred during pipeline execution.")
        raise


def main() -> None:
    logger.info("Parsing command line arguments.")
    args = parser.parse_args()
    logger.debug(f"Command line arguments: {args}.")
    # Execute the training pipeline
    result = run_training_pipeline(args.config)
    print(result)


if __name__ == "__main__":
    main()
