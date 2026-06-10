"""Export daily panel CSV for analysis scripts.

Queries daily_sentiment_index JOIN market_features and writes a flat CSV
that can be consumed by stock_news-style analysis scripts.

Usage:
    python -m src.jobs.export_daily_panel
    python -m src.jobs.export_daily_panel --start-date 2025-01-01 --tickers FPT,VCB,HPG
    python -m src.jobs.export_daily_panel --output data/my_panel.csv
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.storage.db import get_engine
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT = "data/processed/daily_panel.csv"

_QUERY = """
SELECT
    s.date,
    s.ticker,
    s.sector,
    s.sentiment_index       AS sentiment_score,
    s.article_count         AS news_count,
    s.positive_count,
    s.neutral_count,
    s.negative_count,
    s.avg_confidence,
    s.trust_level,
    s.final_sentiment,
    m.close,
    m.volume,
    m.log_return,
    m.volume_growth,
    m.clv,
    m.return_1d,
    m.return_3d,
    m.return_5d,
    m.forward_return_1d,
    m.forward_return_3d,
    m.forward_return_5d,
    m.volume_change_5d,
    m.volatility_5d
FROM daily_sentiment_index s
JOIN market_features m
    ON s.date = m.date
   AND s.ticker = m.ticker
{where_clause}
ORDER BY s.ticker, s.date
"""


def export_panel(
    output_path: str = _DEFAULT_OUTPUT,
    start_date: str | None = None,
    end_date: str | None = None,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Query PostgreSQL and write the daily panel CSV.

    Adds ``target_up`` (1 if forward_return_1d > 0, else 0) as a convenience
    binary label used by most analysis scripts.
    """
    conditions: list[str] = []
    params: dict = {}

    if start_date:
        conditions.append("s.date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        conditions.append("s.date <= :end_date")
        params["end_date"] = end_date
    if tickers:
        conditions.append("s.ticker = ANY(:tickers)")
        params["tickers"] = tickers

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = _QUERY.format(where_clause=where_clause)

    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    if df.empty:
        logger.warning(
            "No rows returned — ensure sentiment aggregates and market features are populated "
            "and that the date/ticker filters match existing data."
        )
        return df

    df["target_up"] = (df["forward_return_1d"] > 0).astype(int)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    logger.info(
        "Exported daily panel: rows=%s tickers=%s -> %s",
        len(df), df["ticker"].nunique(), out.resolve(),
    )
    return df


def main() -> None:
    setup_logging()

    default_output = os.path.join(
        os.getenv("MARKET_VALIDATION_OUTPUT_DIR", "data/processed"),
        "daily_panel.csv",
    )

    parser = argparse.ArgumentParser(description="Export daily panel CSV for analysis.")
    parser.add_argument("--output", default=default_output, help="Output CSV path")
    parser.add_argument("--start-date", default=None, help="Filter start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Filter end date YYYY-MM-DD")
    parser.add_argument(
        "--tickers", default=None,
        help="Comma-separated tickers to include, e.g. FPT,VCB,HPG",
    )
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    export_panel(
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        tickers=tickers,
    )


if __name__ == "__main__":
    main()
