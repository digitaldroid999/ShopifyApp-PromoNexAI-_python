"""
Centralized logging configuration for the API service.
"""

import logging
import sys
from typing import Optional

from app.config import settings


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Args:
        name: Logger name (typically __name__).
        level: Optional log level override (e.g. DEBUG, INFO).

    Returns:
        Configured Logger instance.
    """
    log_level = (level or getattr(settings, "LOG_LEVEL", "INFO")).upper()
    level_value = getattr(logging, log_level, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level_value)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level_value)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
