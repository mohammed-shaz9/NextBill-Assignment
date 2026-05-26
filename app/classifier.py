"""
Invoice Classifier — model loading, prediction, and confidence scoring using Scikit-Learn Pipeline.
"""

import os
import sys
import time
import joblib
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import settings  # noqa: E402
from preprocessing.text_processor import clean_text  # noqa: E402


class InvoiceClassifier:
    """Loads a trained sklearn pipeline and exposes a `predict()` method."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        path = Path(model_path or settings.MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Pipeline model file not found at {path}. "
                "Please run `python -m training.train_model` first."
            )

        # Ensure preprocessing.text_processor is importable before loading pickle
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        self.pipeline = joblib.load(path)
        self.version = settings.MODEL_VERSION
        self.categories = settings.CATEGORIES
        self._classes = list(self.pipeline.classes_)

    def predict(self, text: str) -> tuple[str, float, float, list[str]]:
        """
        Classify a single invoice text.

        Optimization: Runs vectorization and Logistic Regression exactly once by using
        predict_proba and computing the argmax, cutting latency in half.

        Returns:
            (category, confidence, processing_time_ms, processed_tokens)
        """
        start = time.perf_counter()

        # The pipeline preprocessor clean_text runs on the input list,
        # vectorizes it, and returns the probabilities for all classes.
        probabilities = self.pipeline.predict_proba([text])[0]

        class_index = probabilities.argmax()
        confidence = float(probabilities[class_index])

        # OOD (Out-of-Distribution) mitigation: fall back to Unknown if below threshold
        if confidence < settings.CONFIDENCE_THRESHOLD:
            category = "Unknown"
        else:
            category = self._classes[class_index]

        # Get processed tokens for frontend inspection
        cleaned = clean_text(text)
        processed_tokens = [tok for tok in cleaned.split() if tok]

        elapsed_ms = (time.perf_counter() - start) * 1000
        return category, round(confidence, 4), round(elapsed_ms, 2), processed_tokens
