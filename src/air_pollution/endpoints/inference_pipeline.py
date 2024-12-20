from typing import Any, Dict, List
import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel
from prometheus_client import Counter, Summary
from air_pollution.config import TransformationConfig
from air_pollution.scripts.inference import load_pipeline

# Prometheus Metrics
REQUEST_COUNT = Counter(
    "predict_requests_total", "Total number of requests to the predict endpoint"
)
REQUEST_LATENCY = Summary(
    "predict_request_latency_seconds", "Latency of predict requests in seconds"
)
REQUEST_ERRORS = Counter(
    "predict_request_errors_total", "Total number of errors in predict requests"
)


# Define the class names (Air Quality categories)
CLASS_LABELS = {
    0: "Good",
    1: "Moderate",
    2: "Hazardous",
    3: "Poor",
}


# Input and Output schemas
class PredictInput(BaseModel):
    data: List[
        Dict[str, Any]
    ]  # List of dictionaries, each representing a row with feature values


class PredictOutput(BaseModel):
    predictions: List[
        str
    ]  # List of predicted class labels (e.g., Good, Moderate, etc.)


# Create a router instance
router = APIRouter()

# Instantiate the Pipeline with Default Configuration
TRANSFORMATION_CONFIG = TransformationConfig(scaling_method="minmax", normalize=False)
model_path = "trained_model.pkl"
pipeline_endpoint = load_pipeline(TRANSFORMATION_CONFIG, model_path)


@router.post("/predict", response_model=PredictOutput)
async def predict_endpoint(input_data: PredictInput) -> PredictOutput:
    """
    Converts input JSON to DataFrame, runs the pipeline, and converts output to predicted class labels.
    """
    REQUEST_COUNT.inc()  # Increment request count
    with REQUEST_LATENCY.time():
        try:
            # Convert input JSON to pandas DataFrame
            input_df = pd.DataFrame(input_data.data)
            logger.info("Input data converted to DataFrame.")

            # Run pipeline predict method
            predictions_df = pipeline_endpoint.run(input_df)
            print(predictions_df)
            # Assuming the model returns predicted class indices, map them to class labels
            # Convert Categorical data to a pandas Series, then extract values
            predictions_classes = [
                CLASS_LABELS.get(pred, "Unknown")
                for pred in predictions_df.astype(int).values
            ]

            logger.info("Prediction completed successfully.")
            return PredictOutput(predictions=predictions_classes)

        except Exception as e:
            REQUEST_ERRORS.inc()  # Increment error count
            logger.error(f"Error in predict endpoint: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed.")
