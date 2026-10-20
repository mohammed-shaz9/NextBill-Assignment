"""
Text preprocessing for invoice classification.

No NLTK: lowercasing + regex tokenization + stopword removal is enough, and
char n-grams in the model cover plurals/typos that a lemmatizer would.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words that appear across ALL categories and therefore carry no signal
_DOMAIN_STOPWORDS = {
    "invoice",
    "bill",
    "payment",
    "charges",
    "charge",
    "fee",
    "fees",
    "cost",
    "costs",
    "amount",
    "total",
    "paid",
    "received",
    "dated",
    "vendor",
    "purchase",
    "order",
    "monthly",
    "annual",
    "quarterly",
    "rs",
    "inr",
    "rupees",
    "number",
    "ref",
    "reference",
    "gst",
    "reimbursement",
    "claim",
}
_ALL_STOPWORDS = ENGLISH_STOP_WORDS | _DOMAIN_STOPWORDS
_TOKEN = re.compile(r"[a-z]+")


def clean_text(text: str) -> str:
    """Lowercase, keep alphabetic tokens, drop stopwords and 1-char tokens."""
    if not text:
        return ""
    return " ".join(
        t for t in _TOKEN.findall(text.lower()) if len(t) > 1 and t not in _ALL_STOPWORDS
    )
