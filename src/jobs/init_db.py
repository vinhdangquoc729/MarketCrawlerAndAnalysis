"""Initialize PostgreSQL database schema."""
from __future__ import annotations

import logging

from src.storage.db import execute_schema
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    execute_schema()
    logger.info("Database schema initialized successfully.")


if __name__ == "__main__":
    main()
