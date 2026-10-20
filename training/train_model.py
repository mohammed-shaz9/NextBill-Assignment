"""
Model training for invoice expense classification.

1. Deduplicate (raw synthetic data is ~30% duplicates -> leaks into test set).
2. Compare candidate pipelines with stratified 5-fold CV.
3. Refit the winner, report on a held-out split, save model + metrics.json.
"""

import json
import logging
import os
import sys

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import settings
from preprocessing.text_processor import clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("training")


def _features():
    """Word n-grams for semantics + char n-grams for brands, typos, plurals."""
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def candidates() -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline(
            [
                ("features", _features()),
                (
                    "clf",
                    LogisticRegression(C=10, max_iter=2000, class_weight="balanced"),
                ),
            ]
        ),
        "linear_svc_calibrated": Pipeline(
            [
                ("features", _features()),
                ("clf", CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)),
            ]
        ),
        "complement_nb": Pipeline([("features", _features()), ("clf", ComplementNB(alpha=0.3))]),
    }


def train() -> dict:
    if not os.path.exists(settings.DATA_PATH):
        logger.error("Training data not found. Run `python -m data.generate_data` first.")
        sys.exit(1)

    df = pd.read_csv(settings.DATA_PATH).dropna()
    feedback = settings.BASE_DIR / "data" / "feedback.csv"
    if feedback.exists():  # human-corrected labels from POST /feedback
        df = pd.concat([df, pd.read_csv(feedback)[["text", "category"]]])
    n_raw = len(df)
    df = df[df["category"].isin(settings.CATEGORIES)]
    df = df.loc[~df["text"].str.lower().str.strip().duplicated()]
    logger.info("Loaded %d rows -> %d unique", n_raw, len(df))

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["category"],
        test_size=0.2,
        random_state=42,
        stratify=df["category"],
    )

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = {}
    for name, pipe in candidates().items():
        scores[name] = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        ).mean()
        logger.info("CV macro-F1  %-24s %.4f", name, scores[name])

    best_name = max(scores, key=scores.get)
    best = candidates()[best_name].fit(X_train, y_train)
    y_pred = best.predict(X_test)
    logger.info("Selected %s\n%s", best_name, classification_report(y_test, y_pred))
    labels = sorted(y_test.unique())
    logger.info(
        "Confusion matrix (rows=true, cols=pred) %s\n%s",
        labels,
        confusion_matrix(y_test, y_pred, labels=labels),
    )

    metrics = {
        "model": best_name,
        "cv_macro_f1": {k: round(float(v), 4) for k, v in scores.items()},
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    # Ship a model trained on ALL unique data; held-out numbers above are the honest estimate.
    final = candidates()[best_name].fit(df["text"], df["category"])
    real = settings.BASE_DIR / "data" / "eval_realistic.csv"
    if real.exists():  # hand-written, out-of-template text: the number that matters
        r = pd.read_csv(real)
        metrics["realistic_accuracy"] = round(
            accuracy_score(r["category"], final.predict(r["text"])), 4
        )
    os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
    joblib.dump(final, settings.MODEL_PATH)
    settings.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved %s and %s: %s", settings.MODEL_PATH, settings.METRICS_PATH, metrics)
    return metrics


if __name__ == "__main__":
    train()
