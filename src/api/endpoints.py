"""
API Endpoints/Routes for the Application.

This module defines the API routes for the contract classification service.
It uses an APIRouter to keep the endpoint logic organized and separate
from the main application setup.

The `/classify` endpoint is the core of the service, handling the
prediction requests by calling the machine learning service.
"""
from fastapi import APIRouter, HTTPException
from src.api.schemas import ContractRequest, ClassificationResponse
from src.services.ml_services import ml_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/classify", response_model=ClassificationResponse)
def classify_contract(request: ContractRequest):
    """
    Accepts raw contract text and returns the predicted category.
    """
    try:
        category, confidence = ml_service.predict(request.text)
        logger.info(f"Classified text. Category: {category}, Confidence: {confidence:.4f}")
        return ClassificationResponse(
            predicted_category=category,
            confidence_score=confidence
        )
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")
