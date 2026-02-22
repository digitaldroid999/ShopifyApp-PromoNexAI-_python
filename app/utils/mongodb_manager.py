"""
MongoDB connection manager. Singleton used for test_audio cache and audio generation.
"""

from typing import Any, Optional

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Lazy client/database so we don't require pymongo at import if unused
_client: Any = None
_database: Any = None


class MongoDBManager:
    """Singleton-style manager: shared connection and database."""

    def __init__(self) -> None:
        self._client: Any = None
        self._database: Any = None

    def ensure_connection(self) -> bool:
        """Connect to MongoDB if not already connected. Returns True if connected."""
        global _client, _database
        try:
            if _client is not None:
                # Quick ping to ensure still connected
                _client.admin.command("ping")
                return True
            try:
                from pymongo import MongoClient
            except ImportError:
                logger.error("pymongo not installed. pip install pymongo")
                return False
            _client = MongoClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=5000,
            )
            _client.admin.command("ping")
            _database = _client[settings.MONGODB_DATABASE]
            logger.info("MongoDB connected successfully")
            return True
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            return False

    @property
    def database(self) -> Any:
        """MongoDB database instance (e.g. database.test_audio)."""
        global _database
        if _database is None:
            self.ensure_connection()
        return _database


# Singleton instance for app.utils.mongodb_manager.mongodb_manager
mongodb_manager = MongoDBManager()
