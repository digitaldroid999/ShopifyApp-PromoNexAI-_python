"""
Application configuration loaded from environment variables.
"""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _bool(value: str) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _int(value: str, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class Settings:
    """Runtime settings from environment."""

    # API
    DEBUG: bool = _bool(os.getenv("DEBUG", "false"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = _int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Remotion
    REMOTION_SERVER_URL: str = os.getenv("REMOTION_SERVER_URL", "http://localhost:5050")

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_DB: int = _int(os.getenv("REDIS_DB", "0"))

    # MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "eshop_scraper")

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = _int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

    # OpenAI (optional)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_MAX_TOKENS: int = _int(os.getenv("OPENAI_MAX_TOKENS", "100"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

    # Scheduler
    CLEANUP_INTERVAL_HOURS: int = _int(os.getenv("CLEANUP_INTERVAL_HOURS", "24"))
    CLEANUP_DAYS_THRESHOLD: int = _int(os.getenv("CLEANUP_DAYS_THRESHOLD", "2"))

    # Shopify app public folder (relative to this project root); composited images saved here
    COMPOSITED_IMAGES_OUTPUT_DIR: str = os.getenv(
        "COMPOSITED_IMAGES_OUTPUT_DIR",
        os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "promo-nex-ai", "public", "composited_images")),
    )


settings = Settings()
