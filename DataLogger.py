"""Lightweight logging helpers for Black-Scholes runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path


LOGGER_NAME = "black_scholes"
LOG_FILE = Path(__file__).resolve().parent / "logs" / "black_scholes.log"


def get_logger():
    """Return the shared application logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    logger.addHandler(file_handler)
    return logger


def _format_payload(data):
    if data is None:
        return ""
    return f" | {json.dumps(data, default=str, sort_keys=True)}"


def log_event(message, data=None):
    """Log a normal application event."""
    get_logger().info("%s%s", message, _format_payload(data))


def log_error(message, error=None, data=None):
    """Log an error with the active exception context."""
    logger = get_logger()
    logger.error("%s%s", message, _format_payload(data))
    if error is not None:
        logger.error("%s: %s", type(error).__name__, error)
    else:
        logger.exception(message)