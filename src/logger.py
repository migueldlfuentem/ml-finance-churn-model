"""Central logging utilities for the churn project."""

import logging
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "ml_finance_churn.log"
LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = logging.INFO


def _configure_handler(handler: logging.Handler, level: int) -> logging.Handler:
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def get_logger(name: Optional[str] = None, level: int = DEFAULT_LEVEL) -> logging.Logger:
    """Return a configured logger shared across the project."""
    logger_name = name or "ml_finance_churn"
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(level)
        stream_handler = _configure_handler(logging.StreamHandler(), level)
        file_handler = _configure_handler(logging.FileHandler(LOG_FILE), level)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.propagate = False

    return logger
