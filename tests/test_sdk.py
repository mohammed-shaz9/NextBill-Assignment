"""Unit tests for the NextBillClassifierClient SDK."""

from unittest.mock import patch, MagicMock
import pytest
from sdk.client import NextBillClassifierClient


class TestNextBillClassifierClient:
    """Tests for the zero-dependency Python Client SDK."""

    @patch("urllib.request.urlopen")
    def test_classify_success(self, mock_urlopen):
        """Should return classification dict on successful API call."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = (
            b'{"category": "Logistics", "confidence": 0.985, "processing_time_ms": 1.2}'
        )
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = NextBillClassifierClient(base_url="http://localhost:8000", api_key="test_key")
        result = client.classify("courier charges for warehouse delivery")

        assert result["category"] == "Logistics"
        assert result["confidence"] == 0.985
        assert result["processing_time_ms"] == 1.2

    @patch("urllib.request.urlopen")
    def test_classify_batch_success(self, mock_urlopen):
        """Should return batch prediction results on successful API call."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = (
            b'{"predictions": [{"category": "Logistics", "confidence": 0.985, "processing_time_ms": 1.2}], '
            b'"total_processing_time_ms": 1.2}'
        )
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = NextBillClassifierClient(base_url="http://localhost:8000", api_key="test_key")
        result = client.classify_batch(["courier charges for warehouse delivery"])

        assert len(result["predictions"]) == 1
        assert result["predictions"][0]["category"] == "Logistics"
        assert result["total_processing_time_ms"] == 1.2

    def test_empty_text_raises_value_error(self):
        """Should raise ValueError when text is empty."""
        client = NextBillClassifierClient()
        with pytest.raises(ValueError, match="Invoice text cannot be empty"):
            client.classify("")

    def test_empty_batch_raises_value_error(self):
        """Should raise ValueError when batch list is empty."""
        client = NextBillClassifierClient()
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            client.classify_batch([])

    def test_large_batch_raises_value_error(self):
        """Should raise ValueError when batch size exceeds 100."""
        client = NextBillClassifierClient()
        large_batch = ["invoice text"] * 101
        with pytest.raises(ValueError, match="Maximum batch size is 100"):
            client.classify_batch(large_batch)
