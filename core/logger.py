import logging
import sys
from core.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for the given module name.

    Usage:
        from core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Processing file: %s", path)

    All loggers share the same root handler so output is consistent
    across every module.  Log level is DEBUG when settings.debug is
    True, otherwise INFO.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called more than once
    if logger.handlers:
        return logger

    level = logging.DEBUG if settings.debug else logging.INFO
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent log records from bubbling up to the root logger
    logger.propagate = False

    return logger