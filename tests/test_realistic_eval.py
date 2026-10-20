"""Honest check: hand-written invoices NOT from the generator templates."""

import pandas as pd

from app.classifier import InvoiceClassifier
from app.config import settings


def test_realistic_accuracy_floor():
    df = pd.read_csv(settings.BASE_DIR / "data" / "eval_realistic.csv")
    preds = [r["category"] for r in InvoiceClassifier().predict_many(df["text"].tolist())]
    wrong = [(t, y, p) for t, y, p in zip(df["text"], df["category"], preds) if y != p]
    acc = 1 - len(wrong) / len(df)
    print(f"realistic accuracy {acc:.2%}", *wrong, sep="\n")
    assert acc >= 0.75, wrong
