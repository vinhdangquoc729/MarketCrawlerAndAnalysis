"""Run full Market Sentiment pipeline.

Pipeline:
1. Init DB schema
2. Build ticker master
3. Build ticker aliases
4. Crawl CafeF articles
5. Run relevance filter
6. Build article_entities
7. Build entity_aspects
8. Run model inference via FastAPI
9. Build sentiment aggregates
10. Optional: run market data + validation pipeline
11. Run QA report
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence

import requests
from dotenv import load_dotenv

from src.utils.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    mode: str
    category: str | None
    max_per_category: int
    timeline_start_page: int
    timeline_pages: int
    no_timeline: bool

    stage2_window: int
    stage2_limit_articles: int | None
    stage2_limit_aspects: int | None

    model_api_url: str
    model_version: str
    batch_size: int
    max_inference_batches: int | None

    run_market_validation: bool
    market_start_date: str | None
    market_end_date: str | None
    market_tickers: str | None
    market_limit_tickers: int | None
    skip_market_fetch: bool
    skip_validation_report: bool

    skip_init: bool
    skip_master: bool
    skip_crawl: bool
    skip_relevance: bool
    skip_stage2: bool
    skip_inference: bool
    skip_aggregates: bool
    skip_qa: bool

    reset_stage2: bool
    reset_aspects: bool


def run_command(command: Sequence[str], step_name: str) -> None:
    """Run a Python module command and stop if it fails."""
    logger.info("=" * 90)
    logger.info("START STEP: %s", step_name)
    logger.info("COMMAND: %s", " ".join(command))

    result = subprocess.run(list(command), check=False)

    if result.returncode != 0:
        logger.error("FAILED STEP: %s returncode=%s", step_name, result.returncode)
        raise RuntimeError(f"Pipeline failed at step: {step_name}")

    logger.info("DONE STEP: %s", step_name)


def check_model_api_health(model_api_url: str) -> None:
    """Check FastAPI model server health before inference."""
    url = model_api_url.rstrip("/")

    try:
        response = requests.get(f"{url}/health", timeout=10)

        if response.status_code != 200:
            raise RuntimeError(
                f"Model API health check failed: status={response.status_code}, body={response.text}"
            )

        logger.info("Model API health check OK: %s", response.json())

    except Exception as exc:
        raise RuntimeError(
            "Cannot connect to sentiment model FastAPI. "
            f"Please start API first. URL={url}. Error={exc}"
        ) from exc


def build_crawl_command(config: PipelineConfig) -> list[str]:
    """Build crawl command based on backfill/daily mode."""
    if config.mode == "daily":
        command = [
            sys.executable,
            "-m",
            "src.jobs.crawl_daily",
            "--max-per-category",
            str(config.max_per_category),
            "--timeline-start-page",
            str(config.timeline_start_page),
            "--timeline-pages",
            str(config.timeline_pages),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "src.jobs.crawl_backfill",
            "--max-per-category",
            str(config.max_per_category),
            "--timeline-start-page",
            str(config.timeline_start_page),
            "--timeline-pages",
            str(config.timeline_pages),
        ]

    if config.category:
        command.extend(["--category", config.category])

    if config.no_timeline:
        command.append("--no-timeline")

    return command


def build_article_entities_command(config: PipelineConfig) -> list[str]:
    """Build command for Stage 2 article_entities."""
    command = [
        sys.executable,
        "-m",
        "src.jobs.build_article_entities",
        "--window",
        str(config.stage2_window),
    ]

    if config.stage2_limit_articles is not None:
        command.extend(["--limit", str(config.stage2_limit_articles)])

    return command


def build_entity_aspects_command(config: PipelineConfig) -> list[str]:
    """Build command for Stage 2 entity_aspects."""
    command = [
        sys.executable,
        "-m",
        "src.jobs.build_entity_aspects",
    ]

    if config.stage2_limit_aspects is not None:
        command.extend(["--limit", str(config.stage2_limit_aspects)])

    return command


def build_inference_command(config: PipelineConfig) -> list[str]:
    """Build command for model inference."""
    command = [
        sys.executable,
        "-m",
        "src.jobs.run_model_inference",
        "--batch-size",
        str(config.batch_size),
        "--model-version",
        config.model_version,
    ]

    if config.max_inference_batches is not None:
        command.extend(["--max-batches", str(config.max_inference_batches)])

    return command


def build_market_validation_command(config: PipelineConfig) -> list[str]:
    """Build command for market data + validation pipeline."""
    command = [
        sys.executable,
        "-m",
        "src.jobs.run_market_validation_pipeline",
    ]

    if config.market_start_date:
        command.extend(["--start-date", config.market_start_date])

    if config.market_end_date:
        command.extend(["--end-date", config.market_end_date])

    if config.market_tickers:
        command.extend(["--tickers", config.market_tickers])

    if config.market_limit_tickers is not None:
        command.extend(["--limit-tickers", str(config.market_limit_tickers)])

    if config.skip_market_fetch:
        command.append("--skip-fetch")

    if config.skip_validation_report:
        command.append("--skip-report")

    return command


def reset_stage2_tables(reset_entities: bool, reset_aspects: bool) -> None:
    """Reset Stage 2 output tables if requested."""
    if not reset_entities and not reset_aspects:
        return

    from sqlalchemy import text
    from src.storage.db import get_engine

    engine = get_engine()

    with engine.begin() as conn:
        if reset_entities:
            logger.warning("Resetting article_entities and entity_aspects...")
            conn.execute(text("TRUNCATE TABLE entity_aspects RESTART IDENTITY CASCADE"))
            conn.execute(text("TRUNCATE TABLE article_entities RESTART IDENTITY CASCADE"))

        elif reset_aspects:
            logger.warning("Resetting entity_aspects only...")
            conn.execute(text("TRUNCATE TABLE entity_aspects RESTART IDENTITY CASCADE"))


def run_pipeline(config: PipelineConfig) -> None:
    """Run full pipeline."""
    logger.info("FULL PIPELINE CONFIG: %s", config)

    if not config.skip_init:
        run_command(
            [sys.executable, "-m", "src.jobs.init_db"],
            "init_db",
        )

    if not config.skip_master:
        run_command(
            [sys.executable, "-m", "src.master_data.build_ticker_master"],
            "build_ticker_master",
        )

        run_command(
            [sys.executable, "-m", "src.master_data.aliases_builder"],
            "aliases_builder",
        )

    if not config.skip_crawl:
        run_command(
            build_crawl_command(config),
            f"crawl_{config.mode}",
        )

    if not config.skip_relevance:
        run_command(
            [sys.executable, "-m", "src.jobs.run_relevance_filter"],
            "run_relevance_filter",
        )

    if not config.skip_stage2:
        reset_stage2_tables(
            reset_entities=config.reset_stage2,
            reset_aspects=config.reset_aspects,
        )

        run_command(
            build_article_entities_command(config),
            "build_article_entities",
        )

        run_command(
            build_entity_aspects_command(config),
            "build_entity_aspects",
        )

    if not config.skip_inference:
        check_model_api_health(config.model_api_url)

        run_command(
            build_inference_command(config),
            "run_model_inference",
        )

    if not config.skip_aggregates:
        run_command(
            [
                sys.executable,
                "-m",
                "src.jobs.build_sentiment_aggregates",
                "--model-version",
                config.model_version,
            ],
            "build_sentiment_aggregates",
        )

    if config.run_market_validation:
        run_command(
            build_market_validation_command(config),
            "run_market_validation_pipeline",
        )

    if not config.skip_qa:
        run_command(
            [sys.executable, "-m", "src.jobs.qa_report"],
            "qa_report",
        )

    logger.info("=" * 90)
    logger.info("FULL PIPELINE COMPLETE")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run full Market Sentiment pipeline."
    )

    parser.add_argument(
        "--mode",
        choices=["backfill", "daily"],
        default="backfill",
        help="Crawl mode.",
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional category: doanh_nghiep, ngan_hang, thi_truong_chung_khoan, bao_cao_phan_tich.",
    )

    parser.add_argument(
        "--max-per-category",
        type=int,
        default=int(os.getenv("MAX_ARTICLES_PER_CATEGORY", "100")),
        help="Maximum articles per category.",
    )

    parser.add_argument(
        "--timeline-start-page",
        type=int,
        default=int(os.getenv("CAFEF_TIMELINE_START_PAGE", "2")),
        help="CafeF timelinelist start page.",
    )

    parser.add_argument(
        "--timeline-pages",
        type=int,
        default=int(os.getenv("CAFEF_TIMELINE_MAX_PAGES", "5")),
        help="Number of CafeF timelinelist pages to crawl.",
    )

    parser.add_argument(
        "--no-timeline",
        action="store_true",
        help="Disable CafeF timeline API crawling.",
    )

    parser.add_argument(
        "--stage2-window",
        type=int,
        default=1,
        help="Number of previous/next sentences used for context.",
    )

    parser.add_argument(
        "--stage2-limit-articles",
        type=int,
        default=None,
        help="Limit relevant articles processed in build_article_entities.",
    )

    parser.add_argument(
        "--stage2-limit-aspects",
        type=int,
        default=None,
        help="Limit article_entities processed in build_entity_aspects.",
    )

    parser.add_argument(
        "--model-api-url",
        type=str,
        default=os.getenv("SENTIMENT_MODEL_API_URL", "http://127.0.0.1:8000"),
        help="FastAPI sentiment model URL.",
    )

    parser.add_argument(
        "--model-version",
        type=str,
        default=os.getenv("SENTIMENT_MODEL_VERSION", "finetuned-v1"),
        help="Model version name stored in entity_sentiments.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("MODEL_INFERENCE_BATCH_SIZE", "64")),
        help="Inference batch size.",
    )

    parser.add_argument(
        "--max-inference-batches",
        type=int,
        default=None,
        help="Optional max inference batches for testing.",
    )

    parser.add_argument(
        "--run-market-validation",
        action="store_true",
        help="Run market data fetching, market feature building and sentiment validation.",
    )

    parser.add_argument(
        "--market-start-date",
        type=str,
        default=os.getenv("MARKET_START_DATE"),
        help="Market data start date, format YYYY-MM-DD.",
    )

    parser.add_argument(
        "--market-end-date",
        type=str,
        default=os.getenv("MARKET_END_DATE"),
        help="Market data end date, format YYYY-MM-DD.",
    )

    parser.add_argument(
        "--market-tickers",
        type=str,
        default=os.getenv("MARKET_TICKERS"),
        help="Comma-separated tickers for market validation, e.g. FPT,VCB,HPG.",
    )

    parser.add_argument(
        "--market-limit-tickers",
        type=int,
        default=None,
        help="Limit number of tickers fetched from ticker_master.",
    )

    parser.add_argument(
        "--skip-market-fetch",
        action="store_true",
        help="Skip fetching market data and only rebuild market features/views/reports.",
    )

    parser.add_argument(
        "--skip-validation-report",
        action="store_true",
        help="Skip generating validation report CSV.",
    )

    parser.add_argument("--skip-init", action="store_true")
    parser.add_argument("--skip-master", action="store_true")
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--skip-relevance", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-aggregates", action="store_true")
    parser.add_argument("--skip-qa", action="store_true")

    parser.add_argument(
        "--reset-stage2",
        action="store_true",
        help="Clear article_entities and entity_aspects before Stage 2.",
    )

    parser.add_argument(
        "--reset-aspects",
        action="store_true",
        help="Clear only entity_aspects before Stage 2.",
    )

    return parser


def main() -> None:
    setup_logging()

    parser = build_parser()
    args = parser.parse_args()

    if args.reset_stage2 and args.reset_aspects:
        raise ValueError("Use either --reset-stage2 or --reset-aspects, not both.")

    config = PipelineConfig(
        mode=args.mode,
        category=args.category,
        max_per_category=args.max_per_category,
        timeline_start_page=args.timeline_start_page,
        timeline_pages=args.timeline_pages,
        no_timeline=args.no_timeline,
        stage2_window=args.stage2_window,
        stage2_limit_articles=args.stage2_limit_articles,
        stage2_limit_aspects=args.stage2_limit_aspects,
        model_api_url=args.model_api_url,
        model_version=args.model_version,
        batch_size=args.batch_size,
        max_inference_batches=args.max_inference_batches,
        run_market_validation=args.run_market_validation,
        market_start_date=args.market_start_date,
        market_end_date=args.market_end_date,
        market_tickers=args.market_tickers,
        market_limit_tickers=args.market_limit_tickers,
        skip_market_fetch=args.skip_market_fetch,
        skip_validation_report=args.skip_validation_report,
        skip_init=args.skip_init,
        skip_master=args.skip_master,
        skip_crawl=args.skip_crawl,
        skip_relevance=args.skip_relevance,
        skip_stage2=args.skip_stage2,
        skip_inference=args.skip_inference,
        skip_aggregates=args.skip_aggregates,
        skip_qa=args.skip_qa,
        reset_stage2=args.reset_stage2,
        reset_aspects=args.reset_aspects,
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()