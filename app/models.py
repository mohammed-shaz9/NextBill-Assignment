"""
Pydantic schemas for API request / response validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Single invoice text for classification."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="The raw, unstructured invoice description or line-item text to be vectorized and classified.",
        json_schema_extra={
            "examples": ["AWS EC2 On-Demand Instances in US-East-1 - Monthly usage charge for auto-scaling group"]
        },
    )


class PredictionResponse(BaseModel):
    """Classification result with confidence score."""

    category: str = Field(
        ..., 
        description="The predicted canonical expense category taxonomy node.",
        json_schema_extra={"examples": ["Cloud/Software"]}
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated probability density output from the logistic sigmoid function (0.0 to 1.0).",
        json_schema_extra={"examples": [0.9984]}
    )
    processing_time_ms: float = Field(
        ..., 
        description="Server-side pipeline execution latency in milliseconds (excluding network transfer overhead).",
        json_schema_extra={"examples": [1.42]}
    )
    processed_tokens: List[str] = Field(
        default_factory=list, 
        description="The clean tokens remaining after stop-word removal and WordNet lemmatization, passed into the TF-IDF vectorizer.",
        json_schema_extra={"examples": [["aws", "ec2", "demand", "instance", "us", "east", "monthly", "usage", "charge"]]}
    )


class BatchPredictionRequest(BaseModel):
    """Multiple invoice texts for highly-concurrent batch classification."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="An array of raw invoice string payloads. Maximum batch limit is 100 per request for fair-use rate limiting constraints.",
        json_schema_extra={
            "examples": [
                [
                    "AWS EC2 On-Demand Instances in US-East-1",
                    "Blue Dart courier charges for warehouse delivery",
                    "Indigo Flight BLR-BOM Business Class"
                ]
            ]
        }
    )


class BatchPredictionResponse(BaseModel):
    """Aggregated batch classification results payload."""

    predictions: List[PredictionResponse] = Field(
        ...,
        description="Ordered list of classification responses corresponding 1:1 with the input array."
    )
    total_processing_time_ms: float = Field(
        ..., 
        description="Total time required to resolve all thread-pool asynchronous futures for the batch (ms).",
        json_schema_extra={"examples": [5.68]}
    )


class HealthResponse(BaseModel):
    """System health and diagnostic telemetry."""

    status: str = Field(..., description="Overall system availability state.", json_schema_extra={"examples": ["healthy"]})
    model_loaded: bool = Field(..., description="Indicates if the in-memory Scikit-Learn pipeline is loaded and warm.", json_schema_extra={"examples": [True]})
    model_version: str = Field(..., description="Git commit hash or MD5 checksum of the active model binary.", json_schema_extra={"examples": ["v_74b92d_2026"]})
    categories: List[str] = Field(..., description="Supported classification taxonomy nodes.", json_schema_extra={"examples": [["Cloud/Software", "Travel", "Logistics", "Office Supplies"]]})


class MetricsResponse(BaseModel):
    """System inference metrics and diagnostic payload."""

    active_connections: int = Field(..., description="Current number of concurrent active TCP connections.")
    inferences_processed: int = Field(..., description="Total cumulative inferences completed since server boot.")
    avg_latency_ms: float = Field(..., description="Moving average of model prediction latency (ms).")
    uptime_seconds: int = Field(..., description="Process uptime in seconds.")
    memory_usage_mb: float = Field(..., description="RSS memory footprint of the Uvicorn worker.")


class FeedbackRequest(BaseModel):
    """Ground-truth feedback payload for the Reinforcement Learning loop."""

    original_text: str = Field(..., description="The raw invoice text previously sent for inference.")
    predicted_category: str = Field(..., description="The category the model predicted.")
    actual_category: str = Field(..., description="The true ground-truth category provided by human correction.")
    
class FeedbackResponse(BaseModel):
    """Confirmation of feedback ingestion."""
    
    status: str = Field(..., description="Ingestion status.", json_schema_extra={"examples": ["queued_for_rlhf"]})
    job_id: str = Field(..., description="Unique UUID tracking the ingestion job.")
