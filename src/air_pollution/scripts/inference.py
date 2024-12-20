import argparse
import pandas as pd
from loguru import logger
import joblib  # For loading the saved model
import json  # For parsing the JSON-like input

from air_pollution.config import TransformationConfig
from air_pollution.data_pipeline.data_transformer.base_transformer import DataTransformer
from air_pollution.data_pipeline.data_transformer.factory import TransformerFactory
from air_pollution.model.base_model import Model
from fastapi import HTTPException

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
    """Load the model and transformer pipeline."""
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
    _data_transformer: DataTransformer
    _model: Model

    def __init__(self, data_transformer: DataTransformer, model: Model):
        self._data_transformer = data_transformer
        self._model = model

    def run(self, data: pd.DataFrame) -> str:
        """Runs the inference data pipeline on the input data and returns the class label."""
        try:
            logger.info("Pipeline execution started.")
            logger.info("Applying Data transformation.")
            transformed_data = self._data_transformer.transform(data)
            logger.debug(f"Transformed Data: {transformed_data.head()}")
            logger.info("Data transformed successfully.")

            logger.info("Running Inference.")
            predictions = self._model.predict(transformed_data)
            logger.debug(f"Predictions: {predictions.head()}")
            logger.info("Model prediction completed successfully.")
            logger.info("Pipeline execution completed.")

            # Convert the numerical prediction to a class label
            predicted_class = predictions.iloc[0]  # Get the first prediction (assuming one row of input)
            class_label = CLASS_LABELS.get(predicted_class, "Unknown")  # Map to class label
            logger.info(f"Predicted class label: {class_label}")
            return class_label

        except Exception as e:
            logger.error(f"Failed in Pipeline Execution: {e}")
            raise HTTPException(status_code=500, detail="Error during inference.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on input data.")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Inference data in JSON object format (e.g., '{\"Temperature\": 28.3, \"Humidity\": 75.6, ...}') or a comma-separated list.",
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Attempt to parse the input data as JSON object
    try:
        try:
            data_dict = json.loads(args.data)  # Parse as JSON string
            logger.info(f"User input as JSON object: {data_dict}")
        except json.JSONDecodeError:
            # If JSON fails, try treating the input as a comma-separated string
            feature_values = args.data.split(',')
            feature_names = [
                "Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2", "CO",
                "Proximity_to_Industrial_Areas", "Population_Density"
            ]
            if len(feature_values) != len(feature_names):
                raise ValueError("Input does not match expected number of features.")

            data_dict = dict(zip(feature_names, map(float, feature_values)))
            logger.info(f"User input as comma-separated values: {data_dict}")

    except Exception as e:
        logger.error(f"Invalid input format: {e}")
        return

    # Convert the dictionary to a pandas DataFrame
    try:
        data = pd.DataFrame([data_dict])  # Convert single row dictionary to DataFrame
        logger.info(f"Data converted to DataFrame: {data.head()}")
    except Exception as e:
        logger.error(f"Error converting dictionary to DataFrame: {e}")
        return

    # Setup transformation configuration
    transformation_config = TransformationConfig(scaling_method="minmax",normalize=False)
    model_path = "trained_model/trained_model.pkl"
    
    # Load the pipeline
    pipeline = load_pipeline(transformation_config, model_path)

    try:
        # Run inference and display predictions
        class_label = pipeline.run(data)  # Get the class label instead of the DataFrame
        logger.info(f"Inference completed. Predicted class label: {class_label}")
    except HTTPException as e:
        logger.error(f"Inference failed: {e.detail}")


if __name__ == "__main__":
    main()
