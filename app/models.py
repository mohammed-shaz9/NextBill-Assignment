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
        description="The raw, unstructured invoice description or line-item text to be vectorized and classified.",
        json_schema_extra={
            "examples": [
                "AWS EC2 On-Demand Instances in US-East-1 - Monthly usage charge for auto-scaling group"
            ]
        },
    )


class Alternative(BaseModel):
    category: str
    confidence: float


class PredictionResponse(BaseModel):
    """Classification result with confidence score."""

    category: str = Field(
        ...,
        description="The predicted canonical expense category taxonomy node.",
        json_schema_extra={"examples": ["Cloud/Software"]},
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model class probability (0.0 to 1.0). Below threshold -> Unknown.",
        json_schema_extra={"examples": [0.9984]},
    )
    processing_time_ms: float = Field(
        ...,
        description="Server-side pipeline execution latency in milliseconds (excluding network transfer overhead).",
        json_schema_extra={"examples": [1.42]},
    )
    processed_tokens: list[str] = Field(
        default_factory=list,
        description="Clean tokens remaining after stop-word removal, passed into the TF-IDF vectorizers.",
        json_schema_extra={
            "examples": [["aws", "ec2", "demand", "instance", "us", "east", "usage"]]
        },
    )
    alternatives: list[Alternative] = Field(
        default_factory=list, description="Top-3 categories with probabilities."
    )


class BatchPredictionRequest(BaseModel):
    """Multiple invoice texts for highly-concurrent batch classification."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="An array of raw invoice string payloads. Maximum batch limit is 100 per request for fair-use rate limiting constraints.",
        json_schema_extra={
            "examples": [
                [
                    "AWS EC2 On-Demand Instances in US-East-1",
                    "Blue Dart courier charges for warehouse delivery",
                    "Indigo Flight BLR-BOM Business Class",
                ]
            ]
        },
    )


class BatchPredictionResponse(BaseModel):
    """Aggregated batch classification results payload."""

    predictions: list[PredictionResponse] = Field(
        ...,
        description="Ordered list of classification responses corresponding 1:1 with the input array.",
    )
    total_processing_time_ms: float = Field(
        ...,
        description="Total time required to resolve all thread-pool asynchronous futures for the batch (ms).",
        json_schema_extra={"examples": [5.68]},
    )


class HealthResponse(BaseModel):
    """System health and diagnostic telemetry."""

    status: str = Field(
        ...,
        description="Overall system availability state.",
        json_schema_extra={"examples": ["healthy"]},
    )
    model_loaded: bool = Field(
        ...,
        description="Indicates if the in-memory Scikit-Learn pipeline is loaded and warm.",
        json_schema_extra={"examples": [True]},
    )
    model_version: str = Field(
        ...,
        description="Git commit hash or MD5 checksum of the active model binary.",
        json_schema_extra={"examples": ["v_74b92d_2026"]},
    )
    categories: list[str] = Field(
        ...,
        description="Supported classification taxonomy nodes.",
        json_schema_extra={
            "examples": [["Cloud/Software", "Travel", "Logistics", "Office Supplies"]]
        },
    )


class MetricsResponse(BaseModel):
    """Real runtime counters + offline evaluation of the loaded model."""

    inferences_processed: int
    avg_latency_ms: float
    unknown_rate: float
    uptime_seconds: int
    model_evaluation: dict


class FeedbackRequest(BaseModel):
    """Human correction; appended to data/feedback.csv and used on next training run."""

    original_text: str = Field(..., min_length=1, max_length=2000)
    predicted_category: str
    actual_category: str


class FeedbackResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"examples": ["stored"]})
    feedback_rows: int
