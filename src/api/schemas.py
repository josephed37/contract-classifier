"""
Pydantic Schemas for API Data Validation.

This module defines the data structures for API requests and responses
using Pydantic's BaseModel.

This ensures that:
- Incoming data for prediction requests is in the correct format.
- Outgoing responses from the API are consistent and well-defined.

Using schemas is a best practice for building robust and reliable APIs,
as it provides automatic data validation and clear documentation.
"""
from pydantic import BaseModel, Field
from typing import Dict, List

class ContractRequest(BaseModel):
    """Defines the structure for a single classification request."""
    # Re-introducing min_length for the single prediction endpoint for robustness.
    text: str = Field(..., min_length=50, description="The raw text of the contract to classify.")

class ClassificationResponse(BaseModel):
    """Defines the structure for a single classification response."""
    predicted_category: str
    confidence_score: float = Field(..., ge=0, le=1)

class ExplainabilityResponse(BaseModel):
    """Defines the structure for the explainability endpoint response."""
    probabilities: Dict[str, float]

class HealthCheckResponse(BaseModel):
    """Defines the structure for the health check response."""
    status: str

class BatchContractRequest(BaseModel):
    """
    Defines the structure for a batch request. Note the absence of a
    min_length constraint on individual texts to support LIME.
    """
    texts: List[str] = Field(..., min_items=1, max_items=5000)

class BatchClassificationResponse(BaseModel):
    """Defines the structure for a batch classification response."""
    predictions: List[ClassificationResponse]

class BatchExplainabilityResponse(BaseModel):
    """Defines the structure for a batch explainability response."""
    all_probabilities: List[Dict[str, float]]
