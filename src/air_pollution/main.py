import argparse

from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

from air_pollution.config import load_config
from air_pollution.data_loader.factory import DataLoaderFactory
from air_pollution.data_pipeline.preprocessing import Preprocessor
from air_pollution.model.factory import ModelFactory

# Configure the argument parser
parser = argparse.ArgumentParser(
    description="Run the ML data pipeline with the specified configuration."
)
parser.add_argument(
    "--config", type=str, required=True, help="Path to the configuration YAML file."
)


def main() -> None:
    """
    Main function to execute the ML data pipeline.

    Steps:
        1. Load the configuration.
        2. Load and preprocess the data.
        3. Split the data into training and test sets.
        4. Train the specified model.
        5. Evaluate the model and display metrics.

    Raises:
        Exception: If any error occurs during pipeline execution, it will be logged.
    """
    try:
        # Step 1: Load configuration
        args = parser.parse_args()
        logger.info(f"Loading configuration from {args.config}.")
        config = load_config(args.config)
        logger.info("Configuration successfully loaded.")

        # Step 2: Load data
        logger.info(f"Loading data using {config.data_loader.file_type} loader.")
        data_loader = DataLoaderFactory.get_data_loader(config.data_loader.file_type)
        data = data_loader.load_data(config.data_loader.file_path)
        logger.info(f"Data successfully loaded. Shape: {data.shape}")

        # Step 3: Preprocess the data
        target_column = "Air Quality"
        logger.info(f"Initializing preprocessor with target column: {target_column}")
        preprocessor = Preprocessor(config, target_column)
        X_train, X_test, y_train, y_test = preprocessor.preprocess(data)
        logger.info("Data preprocessing completed.")
        logger.debug(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        # Step 4: Train the model
        logger.info(f"Initializing and training model of type: {config.model.type}")
        model = ModelFactory.get_model(config.model.type)
        model.train(X_train, y_train)
        logger.info("Model training completed.")

        # Step 5: Predict and evaluate
        logger.info("Generating predictions on the test set.")
        predictions = model.predict(X_test)
        logger.debug(f"Predictions: {predictions}")

        # Step 6: Evaluate the model
        logger.info("Generating confusion matrix.")
        cm = confusion_matrix(y_test, predictions)
        logger.info(f"Confusion Matrix:\n{cm}")

        logger.info("Generating classification report.")
        report = classification_report(y_test, predictions, output_dict=False)
        logger.info(f"Classification Report:\n{report}")

    except Exception:
        logger.exception("An error occurred during pipeline execution.")
        raise


if __name__ == "__main__":
    main()
