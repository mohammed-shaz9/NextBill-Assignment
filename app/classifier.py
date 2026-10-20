"""
Invoice Classifier — loads the trained sklearn pipeline, predicts with confidence.
"""

import os
import sys
import time
from pathlib import Path

import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import settings
from preprocessing.text_processor import clean_text


class InvoiceClassifier:
    def __init__(self, model_path: Path | str | None = None) -> None:
        path = Path(model_path or settings.MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Pipeline model file not found at {path}. Run `python -m training.train_model` first."
            )
        self.pipeline = joblib.load(path)
        self.version = settings.MODEL_VERSION
        self.categories = settings.CATEGORIES
        self._classes = list(self.pipeline.classes_)

    def predict_many(self, texts: list[str]) -> list[dict]:
        """Vectorized: one predict_proba call for the whole batch."""
        start = time.perf_counter()
        probas = self.pipeline.predict_proba(texts)
        per_item_ms = round((time.perf_counter() - start) * 1000 / max(len(texts), 1), 2)
        out = []
        for text, p in zip(texts, probas):
            order = p.argsort()[::-1]
            confidence = float(p[order[0]])
            tokens = clean_text(text).split()
            # No known tokens or low confidence -> don't guess; route to human review.
            unknown = not tokens or confidence < settings.CONFIDENCE_THRESHOLD
            out.append(
                {
                    "category": "Unknown" if unknown else self._classes[order[0]],
                    "confidence": round(confidence, 4),
                    "processing_time_ms": per_item_ms,
                    "processed_tokens": tokens,
                    "alternatives": [
                        {
                            "category": self._classes[i],
                            "confidence": round(float(p[i]), 4),
                        }
                        for i in order[:3]
                    ],
                }
            )
        return out

    def predict(self, text: str) -> tuple[str, float, float, list[str]]:
        r = self.predict_many([text])[0]
        return (
            r["category"],
            r["confidence"],
            r["processing_time_ms"],
            r["processed_tokens"],
        )
