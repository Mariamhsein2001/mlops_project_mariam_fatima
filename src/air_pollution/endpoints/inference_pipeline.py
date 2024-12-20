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


class PredictInput(BaseModel):
    """
    Input schema for the prediction endpoint.


    Attributes:
        data (List[Dict[str, Any]]): A list of dictionaries, each representing a row with feature values.
    """
    data: List[Dict[str, Any]]


class PredictOutput(BaseModel):
    """
    Output schema for the prediction endpoint.


    Attributes:
        predictions (List[str]): A list of predicted class labels (e.g., Good, Moderate, etc.).
    """
    predictions: List[str]


# Create a router instance
router = APIRouter()


# Instantiate the Pipeline with Default Configuration
TRANSFORMATION_CONFIG = TransformationConfig(scaling_method="minmax", normalize=False)
model_path = "trained_model/trained_model.pkl"
pipeline_endpoint = load_pipeline(TRANSFORMATION_CONFIG, model_path)


@router.post("/predict", response_model=PredictOutput)
async def predict_endpoint(input_data: PredictInput) -> PredictOutput:
    """
    Handle predictions for the input data.


    This endpoint receives input data in JSON format, processes it, and returns
    predictions based on a pre-trained pipeline.


    Args:
        input_data (PredictInput): Input data schema containing a list of rows with feature values.


    Returns:
        PredictOutput: Output schema containing a list of predicted class labels.


    Raises:
        HTTPException: If the prediction process fails, raises a 500 Internal Server Error.
    """
    REQUEST_COUNT.inc()  # Increment request count
    with REQUEST_LATENCY.time():
        try:
            # Convert input JSON to pandas DataFrame
            input_df = pd.DataFrame(input_data.data)
            logger.info("Input data converted to DataFrame.")


            # Run pipeline predict method
            predictions_df = pipeline_endpoint.run(input_df)
            logger.info("Pipeline prediction executed.")


            # Map predicted indices to class labels
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



