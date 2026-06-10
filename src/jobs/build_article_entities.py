"""Build article_entities from relevant articles."""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.nlp.context_builder import build_entity_contexts
from src.nlp.entity_linker import find_entities_in_sentences, make_alias_records
from src.nlp.sentence_splitter import build_full_text, split_sentences
from src.storage.db import (
    fetch_relevant_articles,
    fetch_ticker_aliases,
    upsert_article_entities,
)
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Per-process alias cache — loaded once via initializer
_ALIASES = None


def _init_worker(alias_rows: list[dict]) -> None:
    global _ALIASES
    _ALIASES = make_alias_records(alias_rows)


def _to_ticker_set(value) -> set[str] | None:
    if not value:
        return None
    if isinstance(value, list):
        return {str(x).upper() for x in value if x}
    if isinstance(value, tuple):
        return {str(x).upper() for x in value if x}
    if isinstance(value, str):
        cleaned = value.strip("{}")
        if not cleaned:
            return None
        return {x.strip().upper() for x in cleaned.split(",") if x.strip()}
    return None


def _process_article(article: dict, window: int) -> list[dict]:
    """Process one article using the per-process alias cache."""
    aliases = _ALIASES
    article_id = article["article_id"]

    full_text = build_full_text(
        article.get("title"),
        article.get("sapo"),
        article.get("content"),
    )
    sentences = split_sentences(full_text)
    if not sentences:
        return []

    restrict_tickers = _to_ticker_set(article.get("detected_tickers"))
    mentions = find_entities_in_sentences(
        sentences=sentences, aliases=aliases, restrict_tickers=restrict_tickers
    )
    if not mentions and restrict_tickers:
        mentions = find_entities_in_sentences(
            sentences=sentences, aliases=aliases, restrict_tickers=None
        )

    contexts = build_entity_contexts(
        article_id=article_id,
        sentences=sentences,
        mentions=mentions,
        window=window,
    )

    return [
        {
            "article_id": ctx.article_id,
            "ticker": ctx.ticker,
            "entity_text": ctx.entity_text,
            "alias_type": ctx.alias_type,
            "alias_weight": ctx.alias_weight,
            "sentence_index": ctx.sentence_index,
            "context": ctx.context,
            "relevance_weight": ctx.relevance_weight,
        }
        for ctx in contexts
    ]


def _process_article_single(article: dict, aliases, window: int) -> list[dict]:
    """Single-process variant (aliases passed directly)."""
    article_id = article["article_id"]

    full_text = build_full_text(
        article.get("title"),
        article.get("sapo"),
        article.get("content"),
    )
    sentences = split_sentences(full_text)
    if not sentences:
        return []

    restrict_tickers = _to_ticker_set(article.get("detected_tickers"))
    mentions = find_entities_in_sentences(
        sentences=sentences, aliases=aliases, restrict_tickers=restrict_tickers
    )
    if not mentions and restrict_tickers:
        mentions = find_entities_in_sentences(
            sentences=sentences, aliases=aliases, restrict_tickers=None
        )

    contexts = build_entity_contexts(
        article_id=article_id,
        sentences=sentences,
        mentions=mentions,
        window=window,
    )

    return [
        {
            "article_id": ctx.article_id,
            "ticker": ctx.ticker,
            "entity_text": ctx.entity_text,
            "alias_type": ctx.alias_type,
            "alias_weight": ctx.alias_weight,
            "sentence_index": ctx.sentence_index,
            "context": ctx.context,
            "relevance_weight": ctx.relevance_weight,
        }
        for ctx in contexts
    ]


VN30 = {
    "ACB","BID","CTG","DGC","FPT","GAS","GVR","HDB","HPG","LPB",
    "MBB","MSN","MWG","PLX","SAB","SHB","SSB","SSI","STB","TCB",
    "TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VPL","VRE",
}


def build_entities_for_articles(
    limit: int | None = None,
    window: int = 1,
    batch_size: int = 500,
    log_every: int = 1000,
    workers: int = 1,
    vn30_only: bool = False,
    since_date: str | None = None,
) -> int:
    articles = fetch_relevant_articles(limit=limit, since_date=since_date)
    alias_rows = fetch_ticker_aliases()
    if vn30_only:
        alias_rows = [r for r in alias_rows if r.get("ticker", "").upper() in VN30]
        logger.info("VN30-only mode: filtered to %s alias rows", len(alias_rows))
    aliases = make_alias_records(alias_rows)

    total = len(articles)
    logger.info("Loaded articles=%s aliases=%s workers=%s", total, len(aliases), workers)

    total_saved = 0
    done = 0
    pending_rows: list[dict] = []

    def flush() -> None:
        nonlocal total_saved
        if pending_rows:
            total_saved += upsert_article_entities(pending_rows)
            pending_rows.clear()

    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(alias_rows,),
        ) as pool:
            futures = {
                pool.submit(_process_article, article, window): article
                for article in articles
            }
            for future in as_completed(futures):
                rows = future.result()
                pending_rows.extend(rows)
                done += 1

                if done % batch_size == 0:
                    flush()
                if done % log_every == 0:
                    logger.info(
                        "Progress %s/%s  pending=%s  saved=%s",
                        done, total, len(pending_rows), total_saved,
                    )
    else:
        for article in articles:
            rows = _process_article_single(article, aliases, window)
            pending_rows.extend(rows)
            done += 1

            if done % batch_size == 0:
                flush()
            if done % log_every == 0:
                logger.info(
                    "Progress %s/%s  pending=%s  saved=%s",
                    done, total, len(pending_rows), total_saved,
                )

    flush()
    logger.info("Build article_entities complete total=%s saved=%s", done, total_saved)
    return total_saved


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Flush to DB every N articles. Default: 500.")
    parser.add_argument("--log-every", type=int, default=1000,
                        help="Print progress every N articles. Default: 1000.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes for NLP. Default: 1.")
    parser.add_argument("--vn30-only", action="store_true",
                        help="Restrict entity detection to VN30 tickers only.")
    parser.add_argument("--since-date", type=str, default=None,
                        help="Only process articles crawled on or after this date (YYYY-MM-DD).")
    args = parser.parse_args()

    build_entities_for_articles(
        limit=args.limit,
        window=args.window,
        batch_size=args.batch_size,
        log_every=args.log_every,
        workers=args.workers,
        vn30_only=args.vn30_only,
        since_date=args.since_date,
    )


if __name__ == "__main__":
    main()
