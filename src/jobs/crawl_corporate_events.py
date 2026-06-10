"""Crawl ticker-specific corporate events from CafeF and save to PostgreSQL.

Uses the CafeF Events API (same endpoint as stock_news/scripts/collect/scrape_events.py)
but stores results in the articles table with detected_tickers pre-populated.

Usage:
  # Crawl specific tickers
  python -m src.jobs.crawl_corporate_events --tickers FPT,VCB,HPG

  # Crawl all active tickers from ticker_master (run build_ticker_master first)
  python -m src.jobs.crawl_corporate_events --from-ticker-master

  # Limit pages and year
  python -m src.jobs.crawl_corporate_events --tickers FPT,VCB --max-pages 5 --min-year 2024
"""
from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from src.crawlers.cafef_events_crawler import CafeFEventsCrawler
from src.storage.db import bulk_upsert_articles, query_dataframe
from src.utils.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)

# VN30 tickers as a sensible default when neither --tickers nor --from-ticker-master is given
VN30_TICKERS = [
    "ACB", "BID", "CTG", "DGC", "FPT", "GAS", "GVR", "HDB", "HPG", "LPB",
    "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSB", "SSI", "STB", "TCB",
    "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]


def load_tickers_from_master(limit: int | None = None) -> list[str]:
    """Load active tickers from ticker_master table."""
    sql = "SELECT ticker FROM ticker_master WHERE is_active = TRUE ORDER BY ticker"
    if limit:
        sql += f" LIMIT {int(limit)}"
    df = query_dataframe(sql)
    if df.empty:
        logger.warning(
            "ticker_master is empty. Run build_ticker_master first, "
            "or use --tickers to specify tickers explicitly."
        )
        return []
    return df["ticker"].dropna().str.upper().str.strip().tolist()


def parse_tickers(tickers_str: str) -> list[str]:
    return [t.strip().upper() for t in tickers_str.split(",") if t.strip()]


def crawl_corporate_events(
    tickers: list[str],
    max_pages: int = 10,
    min_year: int | None = None,
) -> int:
    """Crawl corporate events for the given tickers and save to PostgreSQL."""
    if not tickers:
        logger.warning("No tickers provided — nothing to crawl.")
        return 0

    logger.info(
        "Starting corporate events crawl tickers=%s max_pages=%s min_year=%s",
        len(tickers), max_pages, min_year,
    )

    crawler = CafeFEventsCrawler()
    articles = crawler.crawl_tickers(
        tickers=tickers,
        max_pages=max_pages,
        min_year=min_year,
    )

    if not articles:
        logger.warning("No articles fetched.")
        return 0

    saved = bulk_upsert_articles(articles)

    logger.info(
        "Corporate events crawl complete tickers=%s total_fetched=%s saved=%s",
        len(tickers), len(articles), saved,
    )
    return saved


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Crawl ticker-specific corporate events from CafeF into PostgreSQL."
    )

    ticker_group = parser.add_mutually_exclusive_group()
    ticker_group.add_argument(
        "--tickers",
        type=str,
        default=os.getenv("MARKET_TICKERS"),
        help="Comma-separated tickers, e.g. FPT,VCB,HPG.",
    )
    ticker_group.add_argument(
        "--from-ticker-master",
        action="store_true",
        help="Load all active tickers from the ticker_master table.",
    )
    ticker_group.add_argument(
        "--vn30",
        action="store_true",
        help="Use the VN30 constituent list.",
    )

    parser.add_argument(
        "--limit-tickers",
        type=int,
        default=None,
        help="Limit number of tickers (useful with --from-ticker-master).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Max API pages per ticker (30 events/page). Default: 10.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Only fetch events published from this year onwards.",
    )

    args = parser.parse_args()

    if args.from_ticker_master:
        tickers = load_tickers_from_master(limit=args.limit_tickers)
    elif args.vn30:
        tickers = VN30_TICKERS[:args.limit_tickers] if args.limit_tickers else VN30_TICKERS
    elif args.tickers:
        tickers = parse_tickers(args.tickers)
        if args.limit_tickers:
            tickers = tickers[:args.limit_tickers]
    else:
        logger.info("No ticker source specified — defaulting to VN30.")
        tickers = VN30_TICKERS

    crawl_corporate_events(
        tickers=tickers,
        max_pages=args.max_pages,
        min_year=args.min_year,
    )


if __name__ == "__main__":
    main()
