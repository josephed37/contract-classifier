"""
Main Application Entry Point.

This script initializes and configures the FastAPI application.

Responsibilities:
- Creating the main FastAPI app instance.
- Setting up the lifespan event handler to load the ML model on startup.
- Including the API routers that define the application's endpoints.
- Defining a root endpoint for health checks.
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path
import logging

from src.api.endpoints import router as api_router
from src.api.schemas import HealthCheckResponse
from src.services.ml_services import ml_service

# Configure logging
logging.basicConfig(level=logging.INFO)

MODEL_PATH = Path(__file__).parent.parent / "models" / "final_legalbert_model"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events. The model is loaded on startup.
    """
    ml_service.load(MODEL_PATH)
    yield
    # Cleanup logic can go here if needed

app = FastAPI(
    title="Contract Classification API",
    description="A modular API to classify legal documents using LegalBERT.",
    version="1.0.0",
    lifespan=lifespan
)

# Include the router from our endpoints file
app.include_router(api_router)

@app.get("/", response_model=HealthCheckResponse)
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"status": "Contract Classification API is running."}
