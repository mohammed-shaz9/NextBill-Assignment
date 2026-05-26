"""
Pydantic schemas for API request / response validation.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Single invoice text for classification."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="Invoice description text to classify.",
        json_schema_extra={"examples": ["Blue Dart courier charges for warehouse delivery"]},
    )


class PredictionResponse(BaseModel):
    """Classification result with confidence score."""

    category: str = Field(
        ..., description="Predicted expense category."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score (0.0 – 1.0).",
    )
    processing_time_ms: float = Field(
        ..., description="Server-side processing time in milliseconds."
    )
    processed_tokens: list[str] = Field(
        default_factory=list, description="Tokens extracted after text preprocessing."
    )


class BatchPredictionRequest(BaseModel):
    """Multiple invoice texts for batch classification."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of invoice texts (max 100 per request).",
    )


class BatchPredictionResponse(BaseModel):
    """Batch classification results."""

    predictions: list[PredictionResponse]
    total_processing_time_ms: float = Field(
        ..., description="Total processing time for the entire batch."
    )


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    model_loaded: bool
    model_version: str
    categories: list[str]
