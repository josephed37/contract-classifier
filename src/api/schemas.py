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

class ContractRequest(BaseModel):
    """Defines the structure of the incoming request body."""
    text: str = Field(..., min_length=50, description="The raw text of the contract to classify.")

class ClassificationResponse(BaseModel):
    """Defines the structure of the API's response."""
    predicted_category: str
    confidence_score: float = Field(..., ge=0, le=1, description="The model's confidence in the prediction (0.0 to 1.0).")

class HealthCheckResponse(BaseModel):
    """Defines the structure for the health check response."""
    status: str
