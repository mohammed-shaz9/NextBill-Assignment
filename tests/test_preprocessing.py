"""Unit tests for the text preprocessing pipeline."""

from preprocessing.text_processor import clean_text


class TestCleanText:
    """Tests for the clean_text function."""

    def test_lowercasing(self):
        result = clean_text("COURIER DELIVERY Charges")
        assert result == result.lower()

    def test_removes_special_characters(self):
        result = clean_text("Invoice #INV-2024-001 for $500")
        assert "#" not in result
        assert "$" not in result
        assert "-" not in result

    def test_removes_digits(self):
        result = clean_text("Bill 12345 for 500 items")
        assert "12345" not in result
        assert "500" not in result

    def test_removes_domain_stopwords(self):
        """Domain stopwords like 'invoice', 'bill', 'payment' should be removed."""
        result = clean_text("Invoice for courier delivery")
        assert "invoice" not in result.split()

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_whitespace_only(self):
        assert clean_text("   ") == ""

    def test_preserves_meaningful_words(self):
        result = clean_text("courier delivery warehouse")
        assert "courier" in result or "delivery" in result or "warehouse" in result

    def test_lemmatization(self):
        """Words should be lemmatized (e.g., 'charges' → 'charge')."""
        result = clean_text("shipping charges and deliveries")
        # Lemmatizer should reduce 'deliveries' → 'delivery'
        assert "delivery" in result or "deliveries" not in result
