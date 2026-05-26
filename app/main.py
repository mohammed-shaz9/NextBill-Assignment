"""
NextBill Invoice Expense Classifier — FastAPI Application

Features:
  - Header-based API Key Authentication
  - In-Memory Rate Limiting
  - Dependency Injection (for Model Classifier)
  - Custom Global Exception Handling
"""

import time
import logging
import sys
import threading
import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Depends, Security, Request, status
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse

from app.config import settings
from app.models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from app.classifier import InvoiceClassifier

# Setup Logger
logger = logging.getLogger("api")

# ---------- Dependency Injection & Caching ----------
@lru_cache()
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
api_key_header = APIKeyHeader(name=settings.API_KEY_NAME, auto_error=True)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Validates the client's API Key against configured secret.
    """
    if api_key != settings.API_KEY:
        logger.warning("Unauthorized access attempt with invalid API Key")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials: Invalid API Key.",
        )
    return api_key


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
    
    # Security Guard: Stop server if default dev keys are deployed in production env
    if settings.ENVIRONMENT == "production" and settings.API_KEY == "nextbill_dev_secret_key_2026":
        logger.critical("SECURITY BREACH: Default developer API Key cannot be used in a production environment!")
        raise RuntimeError("Security configuration error: Default dev API Key detected in production mode.")

    try:
        # Pre-warm the classifier
        clf = get_classifier()
        logger.info("[OK] Model pre-loaded successfully (version=%s)", clf.version)
    except Exception as exc:
        logger.warning("[WARN] Server starting in degraded mode: %s", exc)
    yield
    logger.info("Shutting down API application...")


# ---------- FastAPI Application Setup ----------
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
    dependencies=[Depends(rate_limiter)]
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
            }
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
                "message": f"Internal server error: {str(exc)}",
            }
        },
    )


# ---------- Endpoints ----------

@app.get("/", response_class=HTMLResponse, tags=["General"])
async def root():
    """Welcome endpoint serving the Sandbox UI Developer Console."""
    index_path = settings.BASE_DIR / "app" / "index.html"
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Welcome to the NextBill API Sandbox</h1><p>Developer Console index.html was not found.</p>"
        )


@app.get("/image.jpeg", tags=["General"])
async def get_logo():
    """Serve the logo image file."""
    logo_path = settings.BASE_DIR / "image.jpeg"
    if logo_path.exists():
        return FileResponse(logo_path)
    raise HTTPException(status_code=404, detail="Logo file not found")


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health status and model availability."""
    try:
        clf = get_classifier()
        model_loaded = True
        model_version = clf.version
    except Exception:
        model_loaded = False
        model_version = "N/A"

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_version,
        categories=settings.CATEGORIES,
    )


@app.get("/categories", tags=["General"])
async def list_categories():
    """List all supported expense classification categories."""
    return {"categories": settings.CATEGORIES}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Classification"]
)
async def predict(
    request: PredictionRequest,
    classifier: InvoiceClassifier = Depends(get_classifier)
):
    """
    Classify a single invoice text. Requires 'x-api-key' header.
    
    Example:
        x-api-key: nextbill_dev_secret_key_2026
    """
    # Offload blocking ML pipeline inference to the Starlette worker threadpool
    category, confidence, processing_time_ms, processed_tokens = await run_in_threadpool(
        classifier.predict, request.text
    )
    return PredictionResponse(
        category=category,
        confidence=confidence,
        processing_time_ms=processing_time_ms,
        processed_tokens=processed_tokens,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Classification"]
)
async def predict_batch(
    request: BatchPredictionRequest,
    classifier: InvoiceClassifier = Depends(get_classifier)
):
    """
    Classify multiple invoice texts (max 100). Requires 'x-api-key' header.
    """
    start = time.perf_counter()

    async def process_single(text: str) -> PredictionResponse:
        if not text.strip():
            return PredictionResponse(
                category="Unknown", confidence=0.0, processing_time_ms=0.0, processed_tokens=[]
            )
        try:
            category, confidence, proc_ms, tokens = await run_in_threadpool(
                classifier.predict, text
            )
            return PredictionResponse(
                category=category,
                confidence=confidence,
                processing_time_ms=proc_ms,
                processed_tokens=tokens,
            )
        except Exception as exc:
            logger.error("Error predicting token: %s", exc)
            return PredictionResponse(
                category="Error", confidence=0.0, processing_time_ms=0.0, processed_tokens=[]
            )

    # Parallelize threadpool execution using asyncio.gather
    predictions = await asyncio.gather(*(process_single(text) for text in request.texts))

    total_ms = round((time.perf_counter() - start) * 1000, 2)
    return BatchPredictionResponse(
        predictions=predictions,
        total_processing_time_ms=total_ms,
    )
