"""Fetch intraday OHLCV data from vnstock and store in intraday_prices table."""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from src.storage.db import get_engine, query_dataframe
from src.utils.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)

APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Ho_Chi_Minh")
VN_TZ = ZoneInfo(APP_TIMEZONE)

INTRADAY_FETCH_DELAY_SECONDS = float(os.getenv("INTRADAY_FETCH_DELAY_SECONDS", "5.0"))
INTRADAY_FETCH_RETRY_SLEEP_SECONDS = float(os.getenv("INTRADAY_FETCH_RETRY_SLEEP_SECONDS", "65.0"))
INTRADAY_FETCH_MAX_RETRIES = int(os.getenv("INTRADAY_FETCH_MAX_RETRIES", "3"))

VALID_INTERVALS = {"1m", "5m", "15m", "30m", "1H"}

CREATE_INTRADAY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS intraday_prices (
    ts          TIMESTAMPTZ     NOT NULL,
    ticker      TEXT            NOT NULL,
    interval    TEXT            NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    source      TEXT,
    created_at  TIMESTAMPTZ     DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts, ticker, interval)
);
CREATE INDEX IF NOT EXISTS idx_intraday_ticker_ts ON intraday_prices (ticker, ts);
"""


def today_vn() -> date:
    return datetime.now(VN_TZ).date()


def ensure_intraday_schema() -> None:
    eng = get_engine()
    raw_conn = eng.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            for statement in CREATE_INTRADAY_TABLE_SQL.strip().split(";"):
                stmt = statement.strip()
                if stmt:
                    cur.execute(stmt)
        raw_conn.commit()
        logger.info("Intraday schema ensured.")
    except Exception as exc:
        raw_conn.rollback()
        logger.error("ensure_intraday_schema failed: %s", exc)
        raise
    finally:
        raw_conn.close()


def load_tickers(limit: int | None = None) -> list[str]:
    """Load tickers from daily_sentiment_index, fallback to ticker_master."""
    for sql in [
        "SELECT DISTINCT ticker FROM daily_sentiment_index WHERE ticker IS NOT NULL ORDER BY ticker",
        "SELECT ticker FROM ticker_master WHERE is_active = TRUE ORDER BY ticker",
    ]:
        try:
            df = query_dataframe(sql + (f" LIMIT {limit}" if limit else ""))
            if not df.empty:
                return df["ticker"].dropna().astype(str).str.upper().str.strip().tolist()
        except Exception:
            continue
    return []


def parse_tickers(text: str | None) -> list[str] | None:
    if not text:
        return None
    tickers = [t.strip().upper() for t in text.split(",") if t.strip()]
    return tickers or None


def is_rate_limit_error(error_text: str) -> bool:
    text = error_text.lower()
    return any(k in text for k in ("rate limit", "too many requests", "429", "20/20", "gioi han"))


def fetch_one_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
) -> pd.DataFrame | None:
    """Fetch intraday bars for one ticker with retry logic."""
    for attempt in range(1, INTRADAY_FETCH_MAX_RETRIES + 1):
        try:
            logger.info(
                "Fetching intraday ticker=%s interval=%s start=%s end=%s attempt=%s/%s",
                ticker, interval, start_date, end_date, attempt, INTRADAY_FETCH_MAX_RETRIES,
            )
            from vnstock.api.quote import Quote
            q = Quote(symbol=ticker, source="VCI")
            df = q.history(start=start_date, end=end_date, interval=interval)

            if df is None or df.empty:
                logger.warning("No intraday rows returned ticker=%s interval=%s", ticker, interval)
                return None

            df = df.copy()
            df.columns = [c.lower() for c in df.columns]

            # Normalize timestamp column (vnstock may call it 'time' or 'date')
            if "time" in df.columns and "ts" not in df.columns:
                df = df.rename(columns={"time": "ts"})
            elif "date" in df.columns and "ts" not in df.columns:
                df = df.rename(columns={"date": "ts"})
            elif df.index.name in ("time", "date"):
                df = df.reset_index()
                df.columns = ["ts"] + list(df.columns[1:])

            required = {"open", "high", "low", "close", "volume", "ts"}
            missing = required - set(df.columns)
            if missing:
                raise RuntimeError(f"Missing columns {missing} for ticker={ticker}")

            df["ts"] = pd.to_datetime(df["ts"], utc=False)
            # Localize to Vietnam time if tz-naive
            if df["ts"].dt.tz is None:
                df["ts"] = df["ts"].dt.tz_localize(VN_TZ)
            else:
                df["ts"] = df["ts"].dt.tz_convert(VN_TZ)

            df["ticker"] = ticker.upper()
            df["interval"] = interval
            df["source"] = "VCI"

            df = df.dropna(subset=["ts", "close"])
            df = df.drop_duplicates(subset=["ts", "ticker", "interval"], keep="last")

            logger.info(
                "Fetched intraday ticker=%s interval=%s rows=%s min_ts=%s max_ts=%s",
                ticker, interval, len(df), df["ts"].min(), df["ts"].max(),
            )
            return df[["ts", "ticker", "interval", "open", "high", "low", "close", "volume", "source"]]

        except Exception as exc:
            error_text = str(exc)
            logger.warning(
                "Intraday fetch failed ticker=%s attempt=%s/%s error=%s",
                ticker, attempt, INTRADAY_FETCH_MAX_RETRIES, exc,
            )
            if is_rate_limit_error(error_text) and attempt < INTRADAY_FETCH_MAX_RETRIES:
                logger.warning("Rate limit — sleeping %ss", INTRADAY_FETCH_RETRY_SLEEP_SECONDS)
                time.sleep(INTRADAY_FETCH_RETRY_SLEEP_SECONDS)
            elif attempt < INTRADAY_FETCH_MAX_RETRIES:
                time.sleep(5)

    logger.warning("Skipping ticker=%s after all retries.", ticker)
    return None


def upsert_intraday_prices(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    sql = """
    INSERT INTO intraday_prices (ts, ticker, interval, open, high, low, close, volume, source)
    VALUES (%(ts)s, %(ticker)s, %(interval)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(source)s)
    ON CONFLICT (ts, ticker, interval) DO UPDATE SET
        open       = EXCLUDED.open,
        high       = EXCLUDED.high,
        low        = EXCLUDED.low,
        close      = EXCLUDED.close,
        volume     = EXCLUDED.volume,
        source     = EXCLUDED.source
    """

    import math

    def _f(v):
        if v is None:
            return None
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    rows = [
        {
            "ts": r["ts"],
            "ticker": r["ticker"],
            "interval": r["interval"],
            "open": _f(r.get("open")),
            "high": _f(r.get("high")),
            "low": _f(r.get("low")),
            "close": _f(r.get("close")),
            "volume": r.get("volume"),
            "source": r.get("source", "VCI"),
        }
        for r in df.to_dict(orient="records")
    ]

    eng = get_engine()
    raw_conn = eng.raw_connection()
    try:
        from psycopg2.extras import execute_batch
        with raw_conn.cursor() as cur:
            execute_batch(cur, sql, rows, page_size=500)
        raw_conn.commit()
        logger.info("upsert_intraday_prices saved=%s", len(rows))
        return len(rows)
    except Exception as exc:
        raw_conn.rollback()
        logger.error("upsert_intraday_prices failed: %s", exc)
        raise
    finally:
        raw_conn.close()


def fetch_intraday_data(
    start_date: str,
    end_date: str,
    interval: str,
    tickers: list[str] | None = None,
    limit_tickers: int | None = None,
) -> int:
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {VALID_INTERVALS}, got '{interval}'")

    ensure_intraday_schema()

    if not tickers:
        tickers = load_tickers(limit=limit_tickers)

    if not tickers:
        raise RuntimeError("No tickers found.")

    logger.info(
        "Intraday fetch config interval=%s start=%s end=%s tickers=%s",
        interval, start_date, end_date, ",".join(tickers),
    )

    total_saved = 0
    all_rows: list[pd.DataFrame] = []

    for idx, ticker in enumerate(tickers, start=1):
        df = fetch_one_ticker(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

        if df is not None and not df.empty:
            all_rows.append(df)

        if idx < len(tickers):
            logger.info(
                "Sleeping %ss before next ticker... progress=%s/%s",
                INTRADAY_FETCH_DELAY_SECONDS, idx, len(tickers),
            )
            time.sleep(INTRADAY_FETCH_DELAY_SECONDS)

    if all_rows:
        result_df = pd.concat(all_rows, ignore_index=True)
        total_saved = upsert_intraday_prices(result_df)
        logger.info(
            "Intraday fetch complete total_rows=%s saved=%s min_ts=%s max_ts=%s",
            len(result_df), total_saved, result_df["ts"].min(), result_df["ts"].max(),
        )
    else:
        logger.warning("No intraday data fetched for any ticker.")

    return total_saved


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Fetch intraday OHLCV data.")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--interval",
        type=str,
        default=os.getenv("INTRADAY_INTERVAL") or "15m",
        choices=sorted(VALID_INTERVALS),
        help="Bar interval: 1m, 5m, 15m, 30m, 1H (default: 15m)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=os.getenv("MARKET_TICKERS"),
        help="Comma-separated tickers, e.g. FPT,VCB. Defaults to daily_sentiment_index tickers.",
    )
    parser.add_argument("--limit-tickers", type=int, default=None)
    args = parser.parse_args()

    today = today_vn()
    # Default lookback: 30 days (intraday history is limited)
    default_lookback = int(os.getenv("INTRADAY_DEFAULT_LOOKBACK_DAYS", "30"))
    end_date = args.end_date or today.isoformat()
    start_date = args.start_date or (today - timedelta(days=default_lookback)).isoformat()

    tickers = parse_tickers(args.tickers)

    fetch_intraday_data(
        start_date=start_date,
        end_date=end_date,
        interval=args.interval,
        tickers=tickers,
        limit_tickers=args.limit_tickers,
    )


if __name__ == "__main__":
    main()
