"""Run financial relevance filter on unprocessed articles."""
from __future__ import annotations

import argparse
import logging

from sqlalchemy import text

from src.preprocessing.relevance_filter import score_article_relevance
from src.storage.db import bulk_upsert_article_relevance, get_engine, load_ticker_aliases, query_dataframe
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

STATUS_BY_DECISION = {
    "process_sentiment": "relevant",
    "review_later": "review",
    "skip_sentiment": "skipped",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run relevance filter on unprocessed articles.")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Articles to score before flushing to DB. Default: 500.")
    parser.add_argument("--log-every", type=int, default=1000,
                        help="Print progress every N articles. Default: 1000.")
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()

    engine = get_engine()
    aliases = load_ticker_aliases(engine).to_dict(orient="records")

    logger.info("Loading unprocessed articles from DB ...")
    articles = query_dataframe(
        """
        SELECT a.article_id, a.title, a.sapo, a.content, a.category, a.crawl_at
        FROM articles a
        LEFT JOIN article_relevance r ON a.article_id = r.article_id
        WHERE r.article_id IS NULL
        ORDER BY a.crawl_at DESC
        """,
        engine=engine,
    ).to_dict(orient="records")

    total = len(articles)
    logger.info("Loaded %s articles — starting scoring batch_size=%s log_every=%s",
                total, args.batch_size, args.log_every)

    relevance_batch: list[dict] = []
    status_batch: list[dict] = []
    done = 0
    decision_counts: dict[str, int] = {}

    def flush() -> None:
        if not relevance_batch:
            return
        bulk_upsert_article_relevance(relevance_batch, engine=engine)
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE articles SET status = :status WHERE article_id = :article_id"),
                status_batch,
            )
        relevance_batch.clear()
        status_batch.clear()

    for article in articles:
        result = score_article_relevance(article, aliases)
        decision = result["decision"]

        relevance_batch.append(result)
        status_batch.append({
            "status": STATUS_BY_DECISION[decision],
            "article_id": article["article_id"],
        })
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        done += 1

        if done % args.batch_size == 0:
            flush()

        if done % args.log_every == 0:
            logger.info(
                "Progress %s/%s  process=%s  review=%s  skip=%s",
                done, total,
                decision_counts.get("process_sentiment", 0),
                decision_counts.get("review_later", 0),
                decision_counts.get("skip_sentiment", 0),
            )

    flush()  # remainder

    logger.info(
        "Relevance filter complete total=%s  process=%s  review=%s  skip=%s",
        done,
        decision_counts.get("process_sentiment", 0),
        decision_counts.get("review_later", 0),
        decision_counts.get("skip_sentiment", 0),
    )


if __name__ == "__main__":
    main()
