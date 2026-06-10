"""Backfill CafeF articles into PostgreSQL."""
from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from src.crawlers.cafef_crawler import CafeFCrawler
from src.storage.db import bulk_upsert_articles
from src.utils.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for CafeF backfill job."""
    parser = argparse.ArgumentParser(
        description="Backfill CafeF articles into PostgreSQL."
    )

    parser.add_argument(
        "--max-per-category",
        type=int,
        default=int(os.getenv("MAX_ARTICLES_PER_CATEGORY", "200")),
        help="Maximum number of articles to crawl per category.",
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help=(
            "Optional category name. "
            "Example: doanh_nghiep, thi_truong_chung_khoan, ngan_hang, bao_cao_phan_tich."
        ),
    )

    parser.add_argument(
        "--timeline-pages",
        type=int,
        default=int(os.getenv("CAFEF_TIMELINE_MAX_PAGES", "10")),
        help="Number of CafeF timelinelist pages to crawl per category.",
    )

    parser.add_argument(
        "--timeline-start-page",
        type=int,
        default=int(os.getenv("CAFEF_TIMELINE_START_PAGE", "2")),
        help="Start page for CafeF timelinelist API.",
    )

    parser.add_argument(
        "--no-timeline",
        action="store_true",
        help="Disable CafeF timelinelist crawling and crawl only initial category HTML.",
    )

    parser.add_argument(
        "--min-year",
        type=int,
        default=2024,
        help="Only keep articles published from this year onwards. Default: 2024.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel threads for article fetching. Default: 10.",
    )

    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    crawler = CafeFCrawler()

    use_timeline = not args.no_timeline
    total_saved = 0

    def flush(batch: list) -> None:
        nonlocal total_saved
        n = bulk_upsert_articles(batch)
        total_saved += n

    if args.category:
        if args.category not in crawler.CATEGORIES:
            available = ", ".join(crawler.CATEGORIES.keys())
            raise ValueError(
                f"Unknown category: {args.category}. "
                f"Available categories: {available}"
            )

        logger.info(
            "Starting CafeF backfill category=%s max_per_category=%s "
            "use_timeline=%s timeline_start_page=%s timeline_pages=%s min_year=%s",
            args.category,
            args.max_per_category,
            use_timeline,
            args.timeline_start_page,
            args.timeline_pages,
            args.min_year,
        )

        articles = crawler.crawl_category(
            category_name=args.category,
            max_articles=args.max_per_category,
            use_timeline=use_timeline,
            timeline_start_page=args.timeline_start_page,
            timeline_max_pages=args.timeline_pages,
            min_year=args.min_year,
            on_batch=flush,
            workers=args.workers,
        )

    else:
        logger.info(
            "Starting CafeF backfill all categories max_per_category=%s "
            "use_timeline=%s timeline_start_page=%s timeline_pages=%s min_year=%s",
            args.max_per_category,
            use_timeline,
            args.timeline_start_page,
            args.timeline_pages,
            args.min_year,
        )

        articles = crawler.crawl_all_categories(
            max_articles_per_category=args.max_per_category,
            use_timeline=use_timeline,
            timeline_start_page=args.timeline_start_page,
            timeline_max_pages=args.timeline_pages,
            min_year=args.min_year,
            on_batch=flush,
            workers=args.workers,
        )

    # Final flush for remainder not divisible by batch_size
    remainder = articles[-(len(articles) % 100):] if articles and len(articles) % 100 != 0 else []
    if remainder:
        total_saved += bulk_upsert_articles(remainder)
    errors = sum(1 for article in articles if article.get("error_message"))

    logger.info(
        "Backfill complete total=%s saved=%s errors=%s",
        len(articles),
        total_saved,
        errors,
    )


if __name__ == "__main__":
    main()