from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from air_pollution.config import load_config
from air_pollution.data_loader.factory import DataLoaderFactory
from air_pollution.data_pipeline.preprocessing import Preprocessor
from air_pollution.model.factory import ModelFactory
import mlflow
import mlflow.sklearn
import joblib
from sklearn.metrics import accuracy_score , f1_score

def run_training_pipeline(config_path: str):
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
            data_loader = DataLoaderFactory.get_data_loader(config.data_loader.file_type)
            data = data_loader.load_data(config.data_loader.file_path)
            logger.info(f"Data successfully loaded. Shape: {data.shape}")

            # Log model parameters
            mlflow.log_param("model_type", config.model.type)
            mlflow.log_param("model_parameters", config.model.params)
            # Step 3: Preprocess the data
            target_column = "Air Quality"
            logger.info(f"Initializing preprocessor with target column: {target_column}")
            preprocessor = Preprocessor(config, target_column)
            X_train, X_test, y_train, y_test = preprocessor.preprocess(data)
            logger.info("Data preprocessing completed.")
            logger.debug(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

            # Step 4: Train the model
            logger.info(f"Initializing and training model of type: {config.model.type}")
            model = ModelFactory.get_model(config.model.type , config.model.params)
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
            f1 = f1_score(y_test , y_pred ,  average='weighted')
            mlflow.log_metric("f1_score", f1)
            logger.debug(f"Accuracy :  {accuracy} , F1-Score : {f1}")
            # logger.info("Generating confusion matrix.")
            # cm = confusion_matrix(y_test, predictions).tolist()
            # mlflow.log_metric("confusion_matrix", cm)

            # logger.info("Generating classification report.")
            # report = classification_report(y_test, predictions, output_dict=True)

            # # Log evaluation metrics
            # for label, metrics in report.items():
            #     if isinstance(metrics, dict):
            #         for metric, value in metrics.items():
            #             mlflow.log_metric(f"{label}_{metric}", value)
            #     else:
            #         mlflow.log_metric(label, metrics)

            # # Step 7: Save the trained model to disk
            # model_path = r"C:\Users\user\Desktop\SE\air_pollution\trained_model.pkl"  # Ensure `save_path` is part of the configuration file
            # logger.info(f"Saving trained model to {model_path}.")
            # joblib.dump(model, model_path)
            # logger.info(f"Model successfully saved to {model_path}.")

            # Log model artifact
            input_example = X_test[0:1]  # Take the first sample from the test set as an example
            mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)


            return {
                "accuracy": accuracy,
                "f1_score": f1
            }
    except Exception:
        logger.exception("An error occurred during pipeline execution.")
        raise
