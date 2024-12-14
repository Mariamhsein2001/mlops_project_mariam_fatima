from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from air_pollution.scripts.train import run_training_pipeline

# Initialize router
router = APIRouter()

# Request schema
class TrainRequest(BaseModel):
    config_path: str

# Response schema
class TrainResponse(BaseModel):
    accuracy: float
    f1_score: float


@router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Train the ML model based on the provided configuration file.

    Args:
        request (TrainRequest): The request body containing the path to the configuration file.

    Returns:
        TrainResponse: The accuracy and f1-score weighted as the result of the training process.
    """
    try:
        result = run_training_pipeline(request.config_path)
        return TrainResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
