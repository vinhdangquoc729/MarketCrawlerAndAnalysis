"""Fetch market data automatically from provider."""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from src.market_data.market_storage import ensure_market_schema, upsert_market_prices
from src.market_data.provider_vnstock import VnstockProvider
from src.storage.db import query_dataframe
from src.utils.logging_config import setup_logging

load_dotenv()

logger = logging.getLogger(__name__)

APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Ho_Chi_Minh")
VN_TZ = ZoneInfo(APP_TIMEZONE)

MARKET_FETCH_DELAY_SECONDS = float(os.getenv("MARKET_FETCH_DELAY_SECONDS", "4.0"))
MARKET_FETCH_RETRY_SLEEP_SECONDS = float(os.getenv("MARKET_FETCH_RETRY_SLEEP_SECONDS", "65.0"))
MARKET_FETCH_MAX_RETRIES = int(os.getenv("MARKET_FETCH_MAX_RETRIES", "3"))


def today_vn() -> date:
    """Return today's date using Vietnam timezone."""
    return datetime.now(VN_TZ).date()


def load_active_tickers(limit: int | None = None) -> list[str]:
    """Load active tickers from ticker_master."""
    sql = """
    SELECT ticker
    FROM ticker_master
    WHERE is_active = TRUE
    ORDER BY ticker
    """

    if limit:
        sql += f" LIMIT {int(limit)}"

    df = query_dataframe(sql)

    if df.empty:
        return []

    return df["ticker"].dropna().astype(str).str.upper().str.strip().tolist()


def load_sentiment_tickers(limit: int | None = None) -> list[str]:
    """Load tickers that already appear in daily_sentiment_index."""
    sql = """
    SELECT DISTINCT ticker
    FROM daily_sentiment_index
    WHERE ticker IS NOT NULL
    ORDER BY ticker
    """

    if limit:
        sql += f" LIMIT {int(limit)}"

    try:
        df = query_dataframe(sql)
    except Exception as exc:
        logger.warning("Could not load sentiment tickers: %s", exc)
        return []

    if df.empty:
        return []

    return df["ticker"].dropna().astype(str).str.upper().str.strip().tolist()


def parse_tickers(tickers_text: str | None) -> list[str] | None:
    """Parse comma-separated tickers from CLI or .env."""
    if not tickers_text:
        return None

    tickers = [
        item.strip().upper()
        for item in tickers_text.split(",")
        if item.strip()
    ]

    return tickers or None


def is_rate_limit_error(error_text: str) -> bool:
    """Detect rate limit errors from provider text."""
    text = error_text.lower()

    return (
        "rate limit" in text
        or "giới hạn" in text
        or "request limit" in text
        or "20/20" in text
        or "too many requests" in text
        or "429" in text
    )


def log_fetch_coverage(df: pd.DataFrame) -> None:
    """Log min/max date by ticker after fetching."""
    if df.empty:
        logger.warning("Fetched dataframe is empty.")
        return

    coverage = (
        df.groupby("ticker")
        .agg(
            min_date=("date", "min"),
            max_date=("date", "max"),
            rows=("date", "count"),
        )
        .reset_index()
        .sort_values("ticker")
    )

    logger.info("Fetched market data coverage:")

    for _, row in coverage.iterrows():
        logger.info(
            "Coverage ticker=%s min_date=%s max_date=%s rows=%s",
            row["ticker"],
            row["min_date"],
            row["max_date"],
            row["rows"],
        )


def warn_if_data_stale(df: pd.DataFrame, expected_end_date: str) -> None:
    """Warn when provider returns data much older than requested end date."""
    if df.empty:
        return

    expected = pd.to_datetime(expected_end_date).date()
    max_date = pd.to_datetime(df["date"]).dt.date.max()
    gap_days = (expected - max_date).days

    if gap_days >= 10:
        logger.warning(
            "Market data may be stale. Requested end_date=%s but provider max_date=%s gap_days=%s",
            expected,
            max_date,
            gap_days,
        )


def fetch_one_ticker(
    provider: VnstockProvider,
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """Fetch one ticker with retry and rate-limit handling."""
    for attempt in range(1, MARKET_FETCH_MAX_RETRIES + 1):
        try:
            logger.info(
                "Fetching market data ticker=%s start=%s end=%s attempt=%s/%s",
                ticker,
                start_date,
                end_date,
                attempt,
                MARKET_FETCH_MAX_RETRIES,
            )

            df = provider.fetch_ohlcv(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )

            if df is None or df.empty:
                logger.warning("No rows returned ticker=%s", ticker)
                return None

            logger.info(
                "Fetched ticker=%s rows=%s min_date=%s max_date=%s",
                ticker,
                len(df),
                df["date"].min(),
                df["date"].max(),
            )

            return df

        except Exception as exc:
            error_text = str(exc)

            logger.warning(
                "Failed to fetch ticker=%s attempt=%s/%s error=%s",
                ticker,
                attempt,
                MARKET_FETCH_MAX_RETRIES,
                exc,
            )

            if is_rate_limit_error(error_text) and attempt < MARKET_FETCH_MAX_RETRIES:
                logger.warning(
                    "Rate limit detected. Sleeping %s seconds before retry...",
                    MARKET_FETCH_RETRY_SLEEP_SECONDS,
                )
                time.sleep(MARKET_FETCH_RETRY_SLEEP_SECONDS)

            elif attempt < MARKET_FETCH_MAX_RETRIES:
                time.sleep(5)

    logger.warning("Skip ticker=%s after all retry attempts.", ticker)
    return None


def fetch_market_data(
    start_date: str,
    end_date: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
    use_sentiment_tickers: bool = False,
    fail_if_empty: bool = True,
    delay_seconds: float | None = None,
) -> int:
    """Fetch market data and upsert into market_prices."""
    ensure_market_schema()

    if not tickers:
        if use_sentiment_tickers:
            tickers = load_sentiment_tickers(limit=limit_tickers)

            if not tickers:
                logger.warning(
                    "No tickers found in daily_sentiment_index. Falling back to ticker_master."
                )
                tickers = load_active_tickers(limit=limit_tickers)
        else:
            tickers = load_active_tickers(limit=limit_tickers)

    if not tickers:
        message = "No tickers found for market data fetching."

        if fail_if_empty:
            raise RuntimeError(message)

        logger.warning(message)
        return 0

    logger.info(
        "Market fetch config start_date=%s end_date=%s total_tickers=%s tickers=%s",
        start_date,
        end_date,
        len(tickers),
        ",".join(tickers),
    )

    provider = VnstockProvider()
    all_rows: list[pd.DataFrame] = []

    for idx, ticker in enumerate(tickers, start=1):
        df = fetch_one_ticker(
            provider=provider,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        if df is not None and not df.empty:
            all_rows.append(df)

        if idx < len(tickers):
            _delay = delay_seconds if delay_seconds is not None else MARKET_FETCH_DELAY_SECONDS
            logger.info(
                "Sleeping %s seconds before next ticker to avoid API rate limit... progress=%s/%s",
                _delay,
                idx,
                len(tickers),
            )
            time.sleep(_delay)

    if not all_rows:
        message = (
            "No market data fetched. "
            "Check vnstock rate limit, provider API, network, or ticker list."
        )

        if fail_if_empty:
            raise RuntimeError(message)

        logger.warning(message)
        return 0

    result_df = pd.concat(all_rows, ignore_index=True)

    result_df["date"] = pd.to_datetime(result_df["date"]).dt.date
    result_df["ticker"] = result_df["ticker"].astype(str).str.upper().str.strip()

    result_df = result_df.dropna(subset=["date", "ticker", "close"])
    result_df = result_df.drop_duplicates(subset=["date", "ticker"], keep="last")

    log_fetch_coverage(result_df)
    warn_if_data_stale(result_df, expected_end_date=end_date)

    saved = upsert_market_prices(result_df)

    logger.info(
        "Market data fetch complete rows=%s saved=%s min_date=%s max_date=%s",
        len(result_df),
        saved,
        result_df["date"].min(),
        result_df["date"].max(),
    )

    return saved


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Fetch market OHLCV data.")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--tickers",
        type=str,
        default=os.getenv("MARKET_TICKERS"),
        help="Comma-separated tickers, e.g. FPT,VCB,HPG",
    )
    parser.add_argument("--limit-tickers", type=int, default=None)
    parser.add_argument(
        "--use-sentiment-tickers",
        action="store_true",
        help="Fetch only tickers that appear in daily_sentiment_index.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Do not fail when no market data is fetched.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=None,
        help="Override inter-ticker sleep delay. Default: MARKET_FETCH_DELAY_SECONDS env var.",
    )

    args = parser.parse_args()

    today = today_vn()

    end_date = args.end_date or today.isoformat()
    start_date = args.start_date or (
        today - timedelta(
            days=int(os.getenv("MARKET_DATA_DEFAULT_LOOKBACK_DAYS", "365"))
        )
    ).isoformat()

    tickers = parse_tickers(args.tickers)

    fetch_market_data(
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        limit_tickers=args.limit_tickers,
        use_sentiment_tickers=args.use_sentiment_tickers,
        fail_if_empty=not args.allow_empty,
        delay_seconds=args.delay_seconds,
    )


if __name__ == "__main__":
    main()