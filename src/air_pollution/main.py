from fastapi import FastAPI
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from air_pollution.endpoints.health import router as health_router
from air_pollution.endpoints.train_pipeline import router as pipeline_router
from air_pollution.endpoints.inference_pipeline import router as inference_router

# Initialize the FastAPI app
app = FastAPI(title="Air Pollution Project API", version="1.0")

# Include API routes
app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(pipeline_router, prefix="/api", tags=["Train"])
app.include_router(inference_router, prefix="/api", tags=["Inference"])

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

logger.add("logs/ml_service.log", rotation="10 MB")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
