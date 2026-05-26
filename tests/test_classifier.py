"""Unit tests for the InvoiceClassifier."""

import pytest
from app.classifier import InvoiceClassifier
from app.config import settings


@pytest.fixture(scope="module")
def classifier():
    """Load the classifier once for all tests in this module."""
    return InvoiceClassifier()


class TestInvoiceClassifier:
    """Tests for the InvoiceClassifier class."""

    def test_model_loads(self, classifier):
        """Model pipeline should load without errors."""
        assert classifier.pipeline is not None

    def test_predict_returns_tuple(self, classifier):
        """predict() should return (category, confidence, time_ms, processed_tokens)."""
        result = classifier.predict("Blue Dart courier delivery charges")
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_predict_category_is_valid(self, classifier):
        """Predicted category must be one of the supported categories."""
        category, _, _, _ = classifier.predict("AWS cloud hosting bill")
        assert category in settings.CATEGORIES or category == "Unknown"

    def test_confidence_range(self, classifier):
        """Confidence score must be between 0 and 1."""
        _, confidence, _, _ = classifier.predict("Monthly electricity bill")
        assert 0.0 <= confidence <= 1.0

    def test_processing_time_positive(self, classifier):
        """Processing time should be a positive number."""
        _, _, proc_time, _ = classifier.predict("Office stationery supplies")
        assert proc_time >= 0

    def test_logistics_prediction(self, classifier):
        """Common logistics text should predict Logistics."""
        category, _, _, _ = classifier.predict("FedEx shipping charges for parcel")
        assert category == "Logistics"

    def test_cloud_software_prediction(self, classifier):
        """Cloud service text should predict Cloud/Software."""
        category, _, _, _ = classifier.predict("AWS monthly cloud hosting bill")
        assert category == "Cloud/Software"

    def test_utilities_prediction(self, classifier):
        """Utility bill text should predict Utilities."""
        category, _, _, _ = classifier.predict("Monthly electricity bill for office")
        assert category == "Utilities"

    def test_travel_prediction(self, classifier):
        """Travel text should predict Travel."""
        category, _, _, _ = classifier.predict("Flight tickets for business trip")
        assert category == "Travel"

    def test_office_supplies_prediction(self, classifier):
        """Office supplies text should predict Office Supplies."""
        category, _, _, _ = classifier.predict("Printer ink cartridges and A4 paper")
        assert category == "Office Supplies"

    def test_inventory_prediction(self, classifier):
        """Inventory text should predict Inventory."""
        category, _, _, _ = classifier.predict("Purchase of raw materials from supplier")
        assert category == "Inventory"

    def test_model_version(self, classifier):
        """Classifier should have a version string."""
        assert classifier.version is not None
        assert len(classifier.version) > 0
