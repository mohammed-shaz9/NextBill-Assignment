"""
Model Training Script for Invoice Expense Classification
Uses a unified scikit-learn Pipeline and GridSearchCV for hyperparameter tuning.
"""

import os
import sys
import joblib
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.text_processor import clean_text  # noqa: E402
from app.config import settings  # noqa: E402

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("training")


def _print_confusion_matrix(cm, labels) -> None:
    """Pretty-print a text-based confusion matrix."""
    col_width = max(len(l) for l in labels) + 2
    header = " " * col_width + "".join(l[:8].ljust(10) for l in labels)
    logger.info(header)
    for i, row in enumerate(cm):
        row_str = labels[i].ljust(col_width) + "".join(str(v).ljust(10) for v in row)
        logger.info(row_str)


def train() -> None:
    """Run the training pipeline with Grid Search."""
    # 1. Load data
    data_path = settings.DATA_PATH
    if not os.path.exists(data_path):
        logger.error("[ERROR] Training data not found. Run `python -m data.generate_data` first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    logger.info("[DATA] Loaded %d samples across %d categories", len(df), df["category"].nunique())
    logger.info("Category distribution:\n%s", df["category"].value_counts().to_string())

    # 2. Split data (raw text, preprocessing runs inside the pipeline)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["category"],
        test_size=0.2,
        random_state=42,
        stratify=df["category"],
    )
    logger.info("   Train size: %d | Test size: %d", len(X_train), len(X_test))

    # 3. Define Pipeline
    # Using clean_text as the preprocessor ensures that text preprocessing is done
    # inside the pipeline, preventing leakage and making inference transparent.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 4. Hyperparameter Grid
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [3000, 5000],
        "clf__C": [0.1, 1.0, 10.0]
    }

    logger.info("[STEP] Running GridSearchCV for parameter tuning...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    logger.info("[BEST] Optimal Parameters: %s", grid_search.best_params_)
    logger.info("[BEST] Best 5-fold CV weighted-F1: %.4f", grid_search.best_score_)

    # 5. Evaluate on Test Set
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)

    # Classification report
    report = classification_report(y_test, y_pred)
    logger.info("Test set classification report:\n%s", report)

    # Confusion matrix
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    logger.info("Confusion Matrix:")
    _print_confusion_matrix(cm, labels)

    # 6. Save Pipeline
    model_path = settings.MODEL_PATH
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(best_pipeline, model_path)
    logger.info("[SAVE] Best pipeline saved to %s", model_path)
    logger.info("[OK] Model training and serialization complete.")


if __name__ == "__main__":
    train()
