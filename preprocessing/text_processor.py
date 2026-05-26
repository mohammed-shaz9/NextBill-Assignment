"""
Text Preprocessing Pipeline for Invoice Classification

Handles text normalization, cleaning, and tokenization tailored
for Indian business invoice text (GST, HSN, courier brands, etc.).
"""

import re
import nltk

# Ensure required NLTK data is available
for resource in ["punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Words that appear across ALL categories and therefore carry no signal
_DOMAIN_STOPWORDS = {
    "invoice", "bill", "payment", "charges", "fee", "fees", "cost",
    "costs", "amount", "total", "paid", "received", "dated", "vendor",
    "purchase", "order", "monthly", "annual", "quarterly", "rs", "inr",
    "rupees", "number", "ref", "reference",
}

_ENGLISH_STOPWORDS = set(stopwords.words("english"))
_ALL_STOPWORDS = _ENGLISH_STOPWORDS | _DOMAIN_STOPWORDS
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Remove special characters / numbers
      3. Tokenize
      4. Remove stopwords (English + domain-specific)
      5. Lemmatize
      6. Re-join

    Args:
        text: Raw invoice description string.

    Returns:
        Cleaned, lemmatized string ready for vectorization.
    """
    if not text or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove special characters and digits (keep only letters + spaces)
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords + lemmatize
    cleaned_tokens = [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in _ALL_STOPWORDS and len(tok) > 1
    ]

    return " ".join(cleaned_tokens)
