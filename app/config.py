"""
NextBill Invoice Expense Classifier - Configuration
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ---------- Paths ----------
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_PATH: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "training" / "trained_model.pkl"
    )
    DATA_PATH: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "training_data.csv"
    )

    # ---------- API Metadata ----------
    API_TITLE: str = "NextBill Invoice Expense Classifier"
    API_DESCRIPTION: str = (
        "An ML-powered REST API that classifies invoice text into expense categories. "
        "Built for NextBill — India's AI-powered autonomous finance engine for MSMEs."
    )
    API_VERSION: str = "1.0.0"

    # ---------- Security ----------
    ENVIRONMENT: str = Field(default="development")
    API_KEY: str = Field(default="nextbill_dev_secret_key_2026")
    API_KEY_NAME: str = "x-api-key"
    ALLOWED_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:3000",
        ]
    )

    # ---------- Supported Expense Categories ----------
    CATEGORIES: list[str] = [
        "Logistics",
        "Office Supplies",
        "Cloud/Software",
        "Utilities",
        "Travel",
        "Inventory",
    ]

    # ---------- Model ----------
    MODEL_VERSION: str = "1.0.0-tfidf-lr"
    CONFIDENCE_THRESHOLD: float = 0.25


# Global settings instance
settings = Settings()
