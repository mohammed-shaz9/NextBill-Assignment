"""
NextBill Invoice Expense Classifier — Python Integration SDK
Zero-dependency implementation using Python's built-in urllib.
"""

import json
import urllib.request
import urllib.error
from typing import List, Dict, Any


class NextBillClassifierClient:
    """
    Zero-dependency Python Client for integrating the NextBill Invoice Expense Classifier.
    """

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "", timeout: float = 5.0):
        """
        Initialize the client.

        Args:
            base_url: The URL where the FastAPI classifier is hosted.
            api_key: The API Key (x-api-key header) configured for authorization.
            timeout: Network request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Extract structured API error messages
            try:
                body = json.loads(exc.read().decode("utf-8"))
                error_msg = body.get("error", {}).get("message", exc.reason)
            except Exception:
                error_msg = exc.reason
            raise RuntimeError(f"NextBill API Error (HTTP {exc.code}): {error_msg}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to connect to NextBill API: {exc}") from exc

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify a single invoice description.

        Args:
            text: Raw invoice text description.

        Returns:
            Dict containing 'category', 'confidence', and 'processing_time_ms'.
        """
        if not text or not text.strip():
            raise ValueError("Invoice text cannot be empty.")
        return self._request("/predict", {"text": text})

    def classify_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        Classify a batch of invoice descriptions (max 100).

        Args:
            texts: List of invoice description strings.

        Returns:
            Dict containing 'predictions' list and 'total_processing_time_ms'.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty.")
        if len(texts) > 100:
            raise ValueError("Maximum batch size is 100.")
        return self._request("/predict/batch", {"texts": texts})
