"""Load and join article-sentiment data with intraday price bars.

For each article-ticker pair, finds the first 15m bar at or after publication
time and computes forward bar returns up to N bars ahead.
"""
from __future__ import annotations

import logging
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.storage.db import get_engine

logger = logging.getLogger(__name__)

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# HOSE session boundaries (Vietnam time)
MARKET_OPEN  = 9 * 60       # 09:00
MORNING_END  = 11 * 60 + 30 # 11:30
AFTERNOON_START = 13 * 60   # 13:00
MARKET_CLOSE = 14 * 60 + 45 # 14:45


def _minutes_since_midnight(ts: pd.Timestamp) -> int:
    return ts.hour * 60 + ts.minute


def classify_timing(ts: pd.Timestamp) -> str:
    """Classify publication time into market session buckets."""
    m = _minutes_since_midnight(ts)
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
        return "midnight_only"   # date-only scraped, no real time
    if m < MARKET_OPEN:
        return "pre_open"        # 00:01 – 08:59  → priced in at open
    if m < MARKET_CLOSE:
        return "intraday"        # 09:00 – 14:44  → priced in same session
    return "post_close"          # 14:45+          → priced in next open


def load_article_sentiment(min_date: str = "2023-10-01") -> pd.DataFrame:
    """Load article-ticker sentiment with publication timestamps."""
    engine = get_engine()
    sql = f"""
        SELECT
            a.article_id,
            ats.ticker,
            a.published_at AT TIME ZONE 'Asia/Ho_Chi_Minh' AS published_at_vn,
            ats.sentiment_score,
            ats.final_sentiment,
            ats.confidence,
            ats.positive_count,
            ats.neutral_count,
            ats.negative_count
        FROM article_ticker_sentiment ats
        JOIN articles a ON ats.article_id = a.article_id
        WHERE a.published_at IS NOT NULL
          AND a.published_at >= '{min_date}'
        ORDER BY a.published_at, ats.ticker
    """
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    df["published_at_vn"] = pd.to_datetime(df["published_at_vn"])
    if df["published_at_vn"].dt.tz is None:
        df["published_at_vn"] = df["published_at_vn"].dt.tz_localize(VN_TZ)

    df["timing"] = df["published_at_vn"].apply(classify_timing)
    df["pub_hour"] = df["published_at_vn"].dt.hour
    df["pub_date"] = df["published_at_vn"].dt.date

    logger.info("Loaded %d article-ticker rows with published_at >= %s", len(df), min_date)
    return df


def load_intraday_with_returns(forward_bars: int = 8) -> pd.DataFrame:
    """Load intraday_prices and compute per-bar log returns + N forward bar returns."""
    engine = get_engine()
    sql = """
        SELECT ts AT TIME ZONE 'Asia/Ho_Chi_Minh' AS ts_vn, ticker, open, high, low, close, volume
        FROM intraday_prices
        ORDER BY ticker, ts
    """
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    df["ts_vn"] = pd.to_datetime(df["ts_vn"])
    if df["ts_vn"].dt.tz is None:
        df["ts_vn"] = df["ts_vn"].dt.tz_localize(VN_TZ)

    df = df.sort_values(["ticker", "ts_vn"]).reset_index(drop=True)

    # Log return for each bar (vs previous bar's close)
    df["log_return"] = np.log(df["close"] / df.groupby("ticker")["close"].shift(1))

    # Forward bar returns: bar+1 ... bar+N
    for i in range(1, forward_bars + 1):
        df[f"fwd_bar_{i}"] = df.groupby("ticker")["log_return"].shift(-i)

    # Cumulative forward return over all N bars
    fwd_cols = [f"fwd_bar_{i}" for i in range(1, forward_bars + 1)]
    df["fwd_cum"] = df[fwd_cols].sum(axis=1, min_count=1)

    logger.info("Loaded %d intraday bars across %d tickers", len(df), df["ticker"].nunique())
    return df


def build_article_bar_dataset(
    min_date: str = "2023-10-01",
    forward_bars: int = 8,
) -> pd.DataFrame:
    """Join article sentiment to the first intraday bar at or after publication.

    Returns one row per article-ticker pair with the matched bar's returns.
    """
    articles = load_article_sentiment(min_date=min_date)
    bars = load_intraday_with_returns(forward_bars=forward_bars)

    # Use merge_asof (forward direction) to match each article to the first bar >= published_at
    articles_sorted = articles.sort_values("published_at_vn").reset_index(drop=True)
    bars_sorted = bars.sort_values("ts_vn").reset_index(drop=True)

    fwd_cols = [f"fwd_bar_{i}" for i in range(1, forward_bars + 1)]
    bar_cols = ["ts_vn", "ticker", "open", "close", "log_return", "fwd_cum"] + fwd_cols

    merged = pd.merge_asof(
        articles_sorted.rename(columns={"published_at_vn": "pub_ts"}),
        bars_sorted[bar_cols].rename(columns={"ts_vn": "bar_ts", "open": "bar_open",
                                               "close": "bar_close", "log_return": "bar_log_return"}),
        left_on="pub_ts",
        right_on="bar_ts",
        by="ticker",
        direction="forward",  # first bar at or after publication
    )

    merged = merged.rename(columns={"pub_ts": "published_at_vn"})
    merged["lag_to_bar_minutes"] = (
        (merged["bar_ts"] - merged["published_at_vn"]).dt.total_seconds() / 60
    )
    merged = merged.dropna(subset=["bar_ts"])

    df = merged
    logger.info(
        "Built article-bar dataset: %d rows, %d tickers, timing=%s",
        len(df), df["ticker"].nunique() if not df.empty else 0,
        df["timing"].value_counts().to_dict() if not df.empty else {},
    )
    return df
