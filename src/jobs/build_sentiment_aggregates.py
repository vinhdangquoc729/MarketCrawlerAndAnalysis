"""Build article-level and daily sentiment aggregates."""
from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv
from sqlalchemy import text

from src.storage.db import get_engine
from src.utils.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)


def build_article_ticker_sentiment(model_version: str, start_date: str | None = None) -> None:
    """Aggregate aspect-level predictions into article+ticker sentiment."""
    engine = get_engine()

    published_filter = ""
    params: dict = {"model_version": model_version}
    if start_date:
        # 7-day buffer: articles just before start_date may map to a trading day >= start_date
        published_filter = "AND a.published_at >= CAST(:pub_cutoff AS date) - INTERVAL '7 days'"
        params["pub_cutoff"] = start_date

    sql = f"""
    INSERT INTO article_ticker_sentiment (
        article_id,
        ticker,
        sentiment_score,
        final_sentiment,
        confidence,
        aspect_count,
        positive_count,
        neutral_count,
        negative_count
    )
    SELECT
        es.article_id,
        es.ticker,
        AVG(es.sentiment_score * COALESCE(es.confidence, 1.0)) AS sentiment_score,
        CASE
            WHEN AVG(es.sentiment_score * COALESCE(es.confidence, 1.0)) > 0.25 THEN 'positive'
            WHEN AVG(es.sentiment_score * COALESCE(es.confidence, 1.0)) < -0.25 THEN 'negative'
            ELSE 'neutral'
        END AS final_sentiment,
        AVG(es.confidence) AS confidence,
        COUNT(*) AS aspect_count,
        SUM(CASE WHEN es.sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive_count,
        SUM(CASE WHEN es.sentiment_label = 'neutral' THEN 1 ELSE 0 END) AS neutral_count,
        SUM(CASE WHEN es.sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative_count
    FROM entity_sentiments es
    JOIN articles a ON es.article_id = a.article_id
    WHERE es.model_version = :model_version
      AND es.inference_status = 'success'
      {published_filter}
    GROUP BY es.article_id, es.ticker
    ON CONFLICT (article_id, ticker)
    DO UPDATE SET
        sentiment_score = EXCLUDED.sentiment_score,
        final_sentiment = EXCLUDED.final_sentiment,
        confidence = EXCLUDED.confidence,
        aspect_count = EXCLUDED.aspect_count,
        positive_count = EXCLUDED.positive_count,
        neutral_count = EXCLUDED.neutral_count,
        negative_count = EXCLUDED.negative_count
    """

    with engine.begin() as conn:
        conn.execute(text(sql), params)


def build_daily_sentiment_index(start_date: str | None = None) -> None:
    """
    Aggregate article+ticker sentiment into daily ticker index.

    Logic:
    - If article_date is a trading day, use that date.
    - If article_date is weekend/holiday/non-trading day, map to next trading date.
    - When start_date is given, only rebuilds rows >= start_date (incremental update).
      A 7-day article buffer is applied so weekend articles that map forward are included.
    """
    engine = get_engine()

    params: dict = {}
    if start_date:
        # Delete only the date range being rebuilt
        delete_sql = "DELETE FROM daily_sentiment_index WHERE date >= CAST(:start_date AS date)"
        params["start_date"] = start_date
        # Articles published up to 7 days before start_date may forward-map into the range
        article_date_filter = "AND DATE(a.published_at AT TIME ZONE 'Asia/Ho_Chi_Minh') >= CAST(:start_date AS date) - INTERVAL '7 days'"
    else:
        delete_sql = "DELETE FROM daily_sentiment_index"
        article_date_filter = ""

    sql = f"""
    WITH article_base AS (
        SELECT
            ats.article_id,
            ats.ticker,
            ats.sentiment_score,
            ats.final_sentiment,
            ats.confidence,
            ats.positive_count,
            ats.neutral_count,
            ats.negative_count,
            tm.sector,
            DATE(
                a.published_at
                AT TIME ZONE 'Asia/Ho_Chi_Minh'
            ) AS article_date
        FROM article_ticker_sentiment ats
        JOIN articles a
            ON ats.article_id = a.article_id
        LEFT JOIN ticker_master tm
            ON ats.ticker = tm.ticker
        WHERE a.published_at IS NOT NULL
          {article_date_filter}
    ),
    article_mapped AS (
        SELECT
            ab.*,
            (
                SELECT MIN(p.date::date)
                FROM market_prices p
                WHERE p.ticker = ab.ticker
                  AND p.date::date >= ab.article_date
            ) AS trading_date
        FROM article_base ab
    )
    INSERT INTO daily_sentiment_index (
        date,
        ticker,
        sector,
        sentiment_index,
        final_sentiment,
        article_count,
        positive_count,
        neutral_count,
        negative_count,
        avg_confidence
    )
    SELECT
        trading_date AS date,
        ticker,
        sector,
        AVG(sentiment_score * COALESCE(confidence, 1.0)) AS sentiment_index,
        CASE
            WHEN AVG(sentiment_score * COALESCE(confidence, 1.0)) > 0.25 THEN 'positive'
            WHEN AVG(sentiment_score * COALESCE(confidence, 1.0)) < -0.25 THEN 'negative'
            ELSE 'neutral'
        END AS final_sentiment,
        COUNT(*) AS article_count,
        SUM(positive_count) AS positive_count,
        SUM(neutral_count) AS neutral_count,
        SUM(negative_count) AS negative_count,
        AVG(confidence) AS avg_confidence
    FROM article_mapped
    WHERE trading_date IS NOT NULL
    GROUP BY
        trading_date,
        ticker,
        sector
    ON CONFLICT (date, ticker)
    DO UPDATE SET
        sector = EXCLUDED.sector,
        sentiment_index = EXCLUDED.sentiment_index,
        final_sentiment = EXCLUDED.final_sentiment,
        article_count = EXCLUDED.article_count,
        positive_count = EXCLUDED.positive_count,
        neutral_count = EXCLUDED.neutral_count,
        negative_count = EXCLUDED.negative_count,
        avg_confidence = EXCLUDED.avg_confidence
    """

    with engine.begin() as conn:
        conn.execute(text(delete_sql), params)
        conn.execute(text(sql), params)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-version",
        type=str,
        default=os.getenv("SENTIMENT_MODEL_VERSION") or "finetuned-v1",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Only rebuild sentiment from this date (YYYY-MM-DD). Skips full DELETE when set.",
    )
    args = parser.parse_args()

    build_article_ticker_sentiment(model_version=args.model_version, start_date=args.start_date)
    build_daily_sentiment_index(start_date=args.start_date)

    logger.info("Sentiment aggregates complete.")


if __name__ == "__main__":
    main()