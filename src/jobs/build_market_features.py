"""Build market features and validation view."""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import text

from src.market_data.market_features import build_market_features
from src.market_data.market_storage import (
    ensure_market_schema,
    load_market_prices,
    upsert_market_features,
)
from src.storage.db import get_engine
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def create_sentiment_market_views() -> None:
    """Create or recreate sentiment-market validation views."""
    engine = get_engine()

    sql = """
    DROP VIEW IF EXISTS sentiment_aspect_market_dataset CASCADE;
    DROP VIEW IF EXISTS sentiment_market_forward_dataset CASCADE;

    CREATE VIEW sentiment_market_forward_dataset AS
    SELECT
        s.date,
        s.ticker,
        s.sector,
        s.sentiment_index,
        s.final_sentiment,
        s.article_count,
        s.positive_count,
        s.neutral_count,
        s.negative_count,
        s.avg_confidence,
        s.trust_level,
        m.close,
        m.volume,
        m.return_1d,
        m.return_3d,
        m.return_5d,
        m.forward_return_1d,
        m.forward_return_3d,
        m.forward_return_5d,
        m.volume_change_5d,
        m.volatility_5d,
        m.log_return,
        m.volume_growth,
        m.clv
    FROM daily_sentiment_index s
    JOIN market_features m
        ON s.date = m.date
       AND s.ticker = m.ticker;

    CREATE VIEW sentiment_aspect_market_dataset AS
    WITH aspect_daily AS (
        SELECT
            DATE(COALESCE(a.published_at, a.crawl_at) AT TIME ZONE 'Asia/Ho_Chi_Minh') AS date,
            ae.ticker,
            ea.aspect,
            AVG(es.sentiment_score * COALESCE(es.confidence, 1.0)) AS aspect_sentiment_score,
            AVG(es.confidence) AS avg_confidence,
            COUNT(*) AS sample_count,
            SUM(CASE WHEN es.sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive_count,
            SUM(CASE WHEN es.sentiment_label = 'neutral' THEN 1 ELSE 0 END) AS neutral_count,
            SUM(CASE WHEN es.sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative_count
        FROM entity_sentiments es
        JOIN entity_aspects ea
            ON es.entity_aspect_id = ea.id
        JOIN article_entities ae
            ON es.entity_id = ae.id
        JOIN articles a
            ON es.article_id = a.article_id
        WHERE es.inference_status = 'success'
        GROUP BY
            DATE(COALESCE(a.published_at, a.crawl_at) AT TIME ZONE 'Asia/Ho_Chi_Minh'),
            ae.ticker,
            ea.aspect
    )
    SELECT
        ad.date,
        ad.ticker,
        tm.sector,
        ad.aspect,
        ad.aspect_sentiment_score,
        ad.avg_confidence,
        ad.sample_count,
        ad.positive_count,
        ad.neutral_count,
        ad.negative_count,
        m.close,
        m.forward_return_1d,
        m.forward_return_3d,
        m.forward_return_5d
    FROM aspect_daily ad
    JOIN market_features m
        ON ad.date = m.date
       AND ad.ticker = m.ticker
    LEFT JOIN ticker_master tm
        ON ad.ticker = tm.ticker;
    """

    with engine.begin() as conn:
        conn.execute(text(sql))

    logger.info("Created sentiment market validation views.")


def main() -> None:
    import argparse
    from datetime import date, timedelta

    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Only recompute features from this date (YYYY-MM-DD). "
             "Loads an extra 10-day buffer for rolling calculations.",
    )
    args = parser.parse_args()

    ensure_market_schema()

    # Load with a 10-day buffer so rolling windows (return_5d, volatility_5d,
    # forward_return_5d) have enough history to produce correct values.
    load_since = None
    if args.start_date:
        buffer_date = date.fromisoformat(args.start_date) - timedelta(days=10)
        load_since = buffer_date.isoformat()

    price_df = load_market_prices(since_date=load_since)
    logger.info("Loaded market_prices rows=%s (since=%s)", len(price_df), load_since)

    feature_df = build_market_features(price_df)

    # When incremental: only upsert rows at or after start_date (buffer rows
    # were needed for calculations but shouldn't overwrite existing features)
    if args.start_date:
        feature_df = feature_df[
            pd.to_datetime(feature_df["date"]).dt.date
            >= date.fromisoformat(args.start_date)
        ]

    saved = upsert_market_features(feature_df)
    logger.info("Market features built rows=%s saved=%s", len(feature_df), saved)

    create_sentiment_market_views()


if __name__ == "__main__":
    main()