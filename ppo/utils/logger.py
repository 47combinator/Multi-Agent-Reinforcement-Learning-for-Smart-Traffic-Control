"""Structured logger for the PPO traffic project."""
import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create or retrieve a named logger.

    Args:
        name    : Logger name (typically __name__ of the calling module).
        log_file: Optional path to write logs to a file.
        level   : Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    # ── Console handler ────────────────────────────────────────────────
    formatter = logging.Formatter(
        fmt   = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt = "%H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler (optional) ────────────────────────────────────────
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
