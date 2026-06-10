"""Analyze when news is published relative to market session hours.

Outputs:
  data/results/intraday/news_timing_hourly.csv   - article count by hour
  data/results/intraday/news_timing_buckets.csv  - breakdown by session bucket
  data/results/intraday/news_timing_by_ticker.csv

Usage:
  python -m src.analysis.intraday.news_timing
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analysis.intraday._loader import load_article_sentiment
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
OUT_DIR = Path("data/results/intraday")


def run(min_date: str = "2023-10-01") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_article_sentiment(min_date=min_date)

    # 1. Hourly distribution
    hourly = (
        df.groupby("pub_hour")
        .agg(articles=("article_id", "nunique"), article_ticker_rows=("article_id", "count"))
        .reset_index()
        .rename(columns={"pub_hour": "hour"})
    )
    hourly.to_csv(OUT_DIR / "news_timing_hourly.csv", index=False)
    print("=== Publication hour distribution ===")
    print(hourly.to_string(index=False))

    # 2. Session bucket breakdown
    BUCKET_ORDER = ["pre_open", "intraday", "post_close", "midnight_only"]
    buckets = (
        df.groupby("timing")
        .agg(
            articles=("article_id", "nunique"),
            article_ticker_rows=("article_id", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            pct_positive=("final_sentiment", lambda x: (x == "positive").mean()),
            pct_neutral=("final_sentiment", lambda x: (x == "neutral").mean()),
            pct_negative=("final_sentiment", lambda x: (x == "negative").mean()),
        )
        .reset_index()
    )
    buckets["timing"] = pd.Categorical(buckets["timing"], categories=BUCKET_ORDER, ordered=True)
    buckets = buckets.sort_values("timing")
    buckets["pct_of_total"] = buckets["article_ticker_rows"] / buckets["article_ticker_rows"].sum()
    buckets.to_csv(OUT_DIR / "news_timing_buckets.csv", index=False)
    print("\n=== Session bucket breakdown ===")
    print(buckets.to_string(index=False))

    # 3. Per-ticker bucket breakdown
    ticker_buckets = (
        df.groupby(["ticker", "timing"])
        .agg(count=("article_id", "count"), avg_sentiment=("sentiment_score", "mean"))
        .reset_index()
    )
    ticker_buckets.to_csv(OUT_DIR / "news_timing_by_ticker.csv", index=False)
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    setup_logging()
    run()
