"""Logging configuration for the market sentiment demo."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configure console and optional file logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
