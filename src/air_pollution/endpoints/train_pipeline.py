"""
Script for handling training requests in the air quality prediction system.

This script defines:
- An API endpoint for initiating the model training process.
- Metrics collection for monitoring training request counts, latency, errors, and performance metrics.
- Input and output schemas for validating training configurations and returning training results.

Classes:
    TrainRequest: Schema for incoming training requests.
    TrainResponse: Schema for outgoing training results.

Endpoints:
    POST /train: Accepts a configuration file path and triggers the training pipeline.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Summary
from air_pollution.scripts.train import run_training_pipeline
from loguru import logger

# Initialize router
router = APIRouter()

# Prometheus Metrics
TRAINING_REQUEST_COUNT = Counter(
    "training_requests_total", "Total number of requests to the train endpoint"
)
TRAINING_LATENCY = Summary(
    "training_request_latency_seconds", "Latency of training requests in seconds"
)
TRAINING_ERRORS = Counter(
    "training_request_errors_total", "Total number of errors in training requests"
)
TRAINING_ACCURACY = Counter(
    "training_accuracy", "Model accuracy from the training process"
)
TRAINING_F1_SCORE = Counter(
    "training_f1_score", "Model F1 score from the training process"
)


class TrainRequest(BaseModel):
    """
    Input schema for the training endpoint.

    Attributes:
        config_path (str): The file path to the configuration file used for the training process.
    """

    config_path: str


class TrainResponse(BaseModel):
    """
    Output schema for the training endpoint.

    Attributes:
        accuracy (float): The accuracy of the trained model.
        f1_score (float): The F1 score of the trained model.
    """

    accuracy: float
    f1_score: float


@router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest) -> TrainResponse:
    """
    Train the ML model based on the provided configuration file.

    This endpoint receives the path to the configuration file, initiates the
    training pipeline, and returns the resulting model performance metrics.

    Args:
        request (TrainRequest): The request body containing the path to the configuration file.

    Returns:
        TrainResponse: A response containing the model's accuracy and F1 score.

    Raises:
        HTTPException: If an error occurs during the training process.
    """
    TRAINING_REQUEST_COUNT.inc()  # Increment request count

    with TRAINING_LATENCY.time():  # Measure latency
        try:
            # Call the training pipeline
            result = run_training_pipeline(request.config_path)

            # Log training KPIs to Prometheus
            TRAINING_ACCURACY.inc(result["accuracy"])
            TRAINING_F1_SCORE.inc(result["f1_score"])

            # Return the training result
            return TrainResponse(**result)

        except Exception as e:
            TRAINING_ERRORS.inc()  # Increment error count
            logger.error(f"Error during training: {e}")
            raise HTTPException(status_code=500, detail=str(e))
