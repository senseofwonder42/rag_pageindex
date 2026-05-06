import sys
from typing import Literal

from loguru import logger

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
    "| <level>{level: <8}</level> "
    "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    "- <level>{message}</level>"
)


def setup_logging(log_level: LogLevel = "INFO") -> None:
    """Configure loguru to output to stdout with formatted messages.

    Should be called once at program start to initialize logging. Removes
    any default handlers and adds a single handler with the standard format.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logger.remove()
    logger.add(sys.stdout, level=log_level, format=_FORMAT, enqueue=False)
