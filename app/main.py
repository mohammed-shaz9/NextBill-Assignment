"""
NextBill Invoice Expense Classifier — FastAPI Application

Features:
  - Header-based API Key Authentication
  - In-Memory Rate Limiting
  - Dependency Injection (for Model Classifier)
  - Custom Global Exception Handling
"""

import csv
import json
import logging
import secrets
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.classifier import InvoiceClassifier
from app.config import settings
from app.models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
)

# Setup Logger
logger = logging.getLogger("api")
BOOT_TIME = time.time()


# ---------- Dependency Injection & Caching ----------
@lru_cache
def get_classifier() -> InvoiceClassifier:
    """
    Dependency provider that caches the InvoiceClassifier instance.
    Eagerly validates model existence or throws 503.
    """
    try:
        return InvoiceClassifier()
    except FileNotFoundError as exc:
        logger.error("Classifier initialization failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not available. Please run model training first.",
        )


# ---------- Security & Authentication ----------
api_key_header = APIKeyHeader(name=settings.API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> None:
    """Enforced only when API_KEY is configured; otherwise open (assignment spec)."""
    if settings.API_KEY and not (api_key and secrets.compare_digest(api_key, settings.API_KEY)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# ---------- Runtime metrics (real counters) ----------
_stats = {"n": 0, "total_ms": 0.0, "unknown": 0}
_stats_lock = threading.Lock()


def _record(results: list[dict]) -> None:
    with _stats_lock:
        _stats["n"] += len(results)
        _stats["total_ms"] += sum(r["processing_time_ms"] for r in results)
        _stats["unknown"] += sum(r["category"] == "Unknown" for r in results)


# ---------- In-Memory Rate Limiting ----------
class ThreadSafeRateLimiter:
    """
    A thread-safe, memory-safe sliding-window rate limiter per client IP.
    Regularly prunes expired database entries to prevent memory leaks.
    """

    def __init__(self, limit: int = 60, window_seconds: float = 60.0):
        self.limit = limit
        self.window_seconds = window_seconds
        self.db: dict[str, list[float]] = {}
        self.lock = threading.Lock()

        # FAANG-grade: Background pruning thread to avoid hot-path global lock contention
        self._prune_thread = threading.Thread(target=self._prune_loop, daemon=True)
        self._prune_thread.start()

    def _prune_loop(self):
        """Runs in background to prevent memory leaks without blocking request threads."""
        while True:
            time.sleep(300)  # Prune every 5 minutes
            now = time.time()
            with self.lock:
                inactive_ips = []
                for ip, timestamps in self.db.items():
                    active = [t for t in timestamps if now - t < self.window_seconds]
                    if not active:
                        inactive_ips.append(ip)
                    else:
                        self.db[ip] = active
                for ip in inactive_ips:
                    del self.db[ip]

    def is_rate_limited(self, client_ip: str) -> bool:
        now = time.time()
        with self.lock:
            # Removed O(N) global prune from hot path
            timestamps = self.db.get(client_ip, [])
            timestamps = [t for t in timestamps if now - t < self.window_seconds]

            if len(timestamps) >= self.limit:
                return True

            timestamps.append(now)
            self.db[client_ip] = timestamps
            return False


# Initialize global rate limiter instance
_rate_limiter_instance = ThreadSafeRateLimiter(limit=60, window_seconds=60.0)


async def rate_limiter(request: Request) -> None:
    """
    Ensures client IP doesn't exceed rate-limiting limits.
    """
    client_ip = request.client.host if request.client else "unknown"
    if _rate_limiter_instance.is_rate_limited(client_ip):
        logger.warning("Rate limit exceeded for IP: %s", client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 60 requests per minute.",
        )


# ---------- Lifespan Management ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Eagerly loads and warm-ups the classifier model during server startup,
    and runs production security validation constraints.
    """
    logger.info("Initializing API application...")

    if settings.ENVIRONMENT == "production" and not settings.API_KEY:
        raise RuntimeError("API_KEY must be set when ENVIRONMENT=production.")

    try:
        # Pre-warm the classifier
        clf = get_classifier()
        logger.info("[OK] Model pre-loaded successfully (version=%s)", clf.version)
    except HTTPException as exc:
        logger.warning("[WARN] Server starting in degraded mode: %s", exc)
    yield
    logger.info("Shutting down API application...")


tags_metadata = [
    {
        "name": "Inference Engine",
        "description": "Core machine learning pipelines for synchronous and asynchronous bulk invoice categorization. All models run in memory using Starlette worker threadpools to prevent event loop blocking.",
    },
    {
        "name": "Classification Taxonomy",
        "description": "Operations to retrieve the active graph of canonical expense categories configured in the taxonomy.",
    },
    {
        "name": "System & Diagnostics",
        "description": "Internal telemetry, health checks, and Prometheus-style metric aggregations for infrastructure monitoring.",
    },
    {
        "name": "MLOps",
        "description": "Human-correction feedback stored for the next retraining run.",
    },
]

# ---------- FastAPI Application Setup ----------
app = FastAPI(
    title=settings.API_TITLE,
    description="""
**NextBill Autonomous Finance REST API.** 
Provides extremely low-latency, scalable NLP invoice parsing capabilities using Scikit-Learn pipelines.

### Enterprise Features
* **Thread-Safe Architecture:** Handles O(N) inference tasks in background worker pools.
* **Feedback loop:** human corrections are stored and folded into the next training run.
* **Security & Rate Limiting:** Global IP-based sliding window limiter prevents DDoS saturation.
""",
    version=settings.API_VERSION,
    openapi_tags=tags_metadata,
    contact={
        "name": "NextBill AI Core Team",
        "url": "https://nextbill-assignment.onrender.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    lifespan=lifespan,
    dependencies=[Depends(rate_limiter)],
)

# CORS configuration (Restricted to configured origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Custom Exception Handler ----------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Structured JSON error responses for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
            },
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all structured JSON error handler for unhandled exceptions."""
    logger.exception("Unhandled server error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": 500,
                "message": "Internal server error.",
            },
        },
    )


# ---------- Endpoints ----------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """Welcome endpoint serving the Sandbox UI Developer Console."""
    index_path = settings.BASE_DIR / "app" / "index.html"
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Welcome to the NextBill API Sandbox</h1><p>Developer Console index.html was not found.</p>"
        )


@app.get("/image.jpeg", include_in_schema=False)
async def get_logo():
    """Serve the logo image file."""
    logo_path = settings.BASE_DIR / "image.jpeg"
    if logo_path.exists():
        return FileResponse(logo_path)
    raise HTTPException(status_code=404, detail="Logo file not found")


@app.get("/health", response_model=HealthResponse, tags=["System & Diagnostics"])
async def health_check():
    """Check API health status and model availability."""
    try:
        clf = get_classifier()
        model_loaded = True
        model_version = clf.version
    except HTTPException:
        model_loaded = False
        model_version = "N/A"

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_version,
        categories=settings.CATEGORIES,
    )


@app.get("/data/download", tags=["System & Diagnostics"])
async def download_data():
    """Download the dynamically generated synthetic training dataset."""
    data_path = settings.BASE_DIR / "data" / "training_data.csv"
    if data_path.exists():
        return FileResponse(data_path, filename="nextbill_training_data.csv", media_type="text/csv")
    raise HTTPException(
        status_code=404,
        detail="Training data not found. Ensure the model has been trained.",
    )


@app.get("/categories", tags=["Classification Taxonomy"])
async def list_categories():
    """List all supported expense classification categories."""
    return {"categories": settings.CATEGORIES}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Inference Engine"],
)
async def predict(
    request: PredictionRequest, classifier: InvoiceClassifier = Depends(get_classifier)
):
    """Classify one invoice text -> category + confidence + top-3 alternatives."""
    result = (await run_in_threadpool(classifier.predict_many, [request.text]))[0]
    _record([result])
    return result


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Inference Engine"],
)
async def predict_batch(
    request: BatchPredictionRequest,
    classifier: InvoiceClassifier = Depends(get_classifier),
):
    """Classify up to 100 texts in one vectorized model call."""
    start = time.perf_counter()
    results = await run_in_threadpool(classifier.predict_many, request.texts)
    _record(results)
    return BatchPredictionResponse(
        predictions=results,
        total_processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["System & Diagnostics"],
    dependencies=[Depends(verify_api_key)],
)
async def get_metrics():
    """Live counters since boot + held-out evaluation written by the training script."""
    try:
        evaluation = json.loads(settings.METRICS_PATH.read_text())
    except FileNotFoundError:
        evaluation = {}
    with _stats_lock:
        n = _stats["n"]
        return MetricsResponse(
            inferences_processed=n,
            avg_latency_ms=round(_stats["total_ms"] / n, 3) if n else 0.0,
            unknown_rate=round(_stats["unknown"] / n, 4) if n else 0.0,
            uptime_seconds=int(time.time() - BOOT_TIME),
            model_evaluation=evaluation,
        )


_feedback_lock = threading.Lock()


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["MLOps"],
    dependencies=[Depends(verify_api_key)],
)
def submit_feedback(request: FeedbackRequest):
    """Store a human correction. Next `python -m training.train_model` run includes it."""
    if request.actual_category not in settings.CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"actual_category must be one of {settings.CATEGORIES}",
        )
    path = settings.BASE_DIR / "data" / "feedback.csv"
    # ponytail: CSV + process lock; move to a DB table when running multiple workers.
    with _feedback_lock:
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["text", "category", "predicted"])
            w.writerow(
                [
                    request.original_text,
                    request.actual_category,
                    request.predicted_category,
                ]
            )
        with open(path, encoding="utf-8") as f:
            rows = sum(1 for _ in f) - 1
    return FeedbackResponse(status="stored", feedback_rows=rows)
