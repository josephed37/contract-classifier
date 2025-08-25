"""
API Endpoints/Routes for the Application.

This module defines the API routes for the contract classification service.
It uses an APIRouter to keep the endpoint logic organized and separate
from the main application setup.

- `/classify`: Returns the top prediction for a given text.
- `/classify-explain-batch`: Accepts a list of texts for efficient explainability.
"""
from fastapi import APIRouter, HTTPException
from src.api.schemas import (
    ContractRequest, ClassificationResponse, 
    BatchContractRequest, BatchExplainabilityResponse
)
from src.services.ml_services import ml_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/classify", response_model=ClassificationResponse)
def classify_contract(request: ContractRequest):
    """Accepts raw contract text and returns the top predicted category."""
    try:
        category, confidence = ml_service.predict(request.text)
        return ClassificationResponse(predicted_category=category, confidence_score=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify-explain-batch", response_model=BatchExplainabilityResponse)
def classify_contract_explain_batch(request: BatchContractRequest):
    """
    Accepts a batch of texts and returns the full probability distribution
    for each, optimized for XAI tools like LIME.
    """
    try:
        all_probabilities = ml_service.predict_explain_batch(request.texts)
        return BatchExplainabilityResponse(all_probabilities=all_probabilities)
    except Exception as e:
        logger.error(f"Batch explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
