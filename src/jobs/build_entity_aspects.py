"""Build entity_aspects and model_input from article_entities."""
from __future__ import annotations

import argparse
import logging

from src.nlp.aspect_extractor import extract_aspects
from src.nlp.sentiment_input_builder import build_model_input
from src.storage.db import (
    fetch_article_entities_without_aspects,
    upsert_entity_aspects,
)
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def build_aspects_for_entities(
    limit: int | None = None,
    batch_size: int = 2000,
    log_every: int = 5000,
) -> int:
    entities = fetch_article_entities_without_aspects(limit=limit)
    total = len(entities)
    logger.info("Loaded entity_rows=%s", total)

    total_saved = 0
    pending: list[dict] = []

    def flush() -> None:
        nonlocal total_saved
        if pending:
            total_saved += upsert_entity_aspects(pending)
            pending.clear()

    for i, entity in enumerate(entities, start=1):
        aspects = extract_aspects(entity["context"], max_aspects=2, allow_general=True)

        for aspect_result in aspects:
            model_input = build_model_input(
                ticker=entity["ticker"],
                aspect=aspect_result.aspect,
                context=entity["context"],
            )
            pending.append({
                "entity_id": entity["id"],
                "aspect": aspect_result.aspect,
                "aspect_keywords": aspect_result.keywords,
                "model_input": model_input,
            })

        if i % batch_size == 0:
            flush()

        if i % log_every == 0:
            logger.info("Progress %s/%s  saved=%s", i, total, total_saved)

    flush()
    logger.info("Build entity_aspects complete total=%s saved=%s", total, total_saved)
    return total_saved


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="Flush to DB every N entity rows. Default: 2000.")
    parser.add_argument("--log-every", type=int, default=5000,
                        help="Print progress every N entity rows. Default: 5000.")
    args = parser.parse_args()

    build_aspects_for_entities(
        limit=args.limit,
        batch_size=args.batch_size,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
