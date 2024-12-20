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


# Request schema
class TrainRequest(BaseModel):
    config_path: str


# Response schema
class TrainResponse(BaseModel):
    accuracy: float
    f1_score: float


@router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest) -> TrainResponse:
    """
    Train the ML model based on the provided configuration file.

    Args:
        request (TrainRequest): The request body containing the path to the configuration file.

    Returns:
        TrainResponse: The accuracy and f1-score weighted as the result of the training process.
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
