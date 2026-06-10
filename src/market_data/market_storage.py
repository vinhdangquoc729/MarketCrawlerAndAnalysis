"""Market data persistence layer."""
from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.storage.db import get_engine
from src.validation.validation_queries import (
    CREATE_MARKET_FEATURES_TABLE_SQL,
    CREATE_MARKET_PRICES_TABLE_SQL,
)

logger = logging.getLogger(__name__)


def ensure_market_schema() -> None:
    """Create market_prices and market_features tables if they do not exist."""
    eng = get_engine()
    raw_conn = eng.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.execute(CREATE_MARKET_PRICES_TABLE_SQL)
            cur.execute(CREATE_MARKET_FEATURES_TABLE_SQL)
        raw_conn.commit()
        logger.info("Market schema ensured.")
    except Exception as exc:
        raw_conn.rollback()
        logger.error("ensure_market_schema failed: %s", exc)
        raise
    finally:
        raw_conn.close()


def load_market_prices(
    engine: Engine | None = None,
    since_date: str | None = None,
) -> pd.DataFrame:
    """Load rows from market_prices ordered by ticker and date.

    since_date: if given (YYYY-MM-DD), only load rows on or after that date.
    """
    eng = engine or get_engine()
    where = "WHERE date >= :since_date" if since_date else ""
    sql = f"""
    SELECT date, ticker, open, high, low, close, volume, source
    FROM market_prices
    {where}
    ORDER BY ticker, date
    """
    try:
        with eng.connect() as conn:
            params = {"since_date": since_date} if since_date else None
            return pd.read_sql(text(sql), conn, params=params)
    except Exception as exc:
        logger.warning("load_market_prices failed: %s", exc)
        return pd.DataFrame()


def upsert_market_prices(df: pd.DataFrame, engine: Engine | None = None) -> int:
    """Insert or update rows in market_prices, keyed on (date, ticker)."""
    if df.empty:
        return 0

    eng = engine or get_engine()

    sql = """
    INSERT INTO market_prices (date, ticker, open, high, low, close, volume, source)
    VALUES (%(date)s, %(ticker)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(source)s)
    ON CONFLICT (date, ticker) DO UPDATE SET
        open       = EXCLUDED.open,
        high       = EXCLUDED.high,
        low        = EXCLUDED.low,
        close      = EXCLUDED.close,
        volume     = EXCLUDED.volume,
        source     = EXCLUDED.source,
        updated_at = CURRENT_TIMESTAMP
    """

    rows = [
        {
            "date": r.get("date"),
            "ticker": r.get("ticker"),
            "open": _float_or_none(r.get("open")),
            "high": _float_or_none(r.get("high")),
            "low": _float_or_none(r.get("low")),
            "close": _float_or_none(r.get("close")),
            "volume": r.get("volume"),
            "source": r.get("source", "vnstock"),
        }
        for r in df.to_dict(orient="records")
    ]

    raw_conn = eng.raw_connection()
    try:
        from psycopg2.extras import execute_batch
        with raw_conn.cursor() as cur:
            execute_batch(cur, sql, rows, page_size=100)
        raw_conn.commit()
        logger.info("upsert_market_prices saved=%s", len(rows))
        return len(rows)
    except Exception as exc:
        raw_conn.rollback()
        logger.error("upsert_market_prices failed: %s", exc)
        raise
    finally:
        raw_conn.close()


def upsert_market_features(df: pd.DataFrame, engine: Engine | None = None) -> int:
    """Insert or update rows in market_features, keyed on (date, ticker)."""
    if df.empty:
        return 0

    eng = engine or get_engine()

    sql = """
    INSERT INTO market_features
        (date, ticker, close, volume,
         return_1d, return_3d, return_5d,
         forward_return_1d, forward_return_3d, forward_return_5d,
         volume_change_5d, volatility_5d,
         log_return, volume_growth, clv)
    VALUES
        (%(date)s, %(ticker)s, %(close)s, %(volume)s,
         %(return_1d)s, %(return_3d)s, %(return_5d)s,
         %(forward_return_1d)s, %(forward_return_3d)s, %(forward_return_5d)s,
         %(volume_change_5d)s, %(volatility_5d)s,
         %(log_return)s, %(volume_growth)s, %(clv)s)
    ON CONFLICT (date, ticker) DO UPDATE SET
        close             = EXCLUDED.close,
        volume            = EXCLUDED.volume,
        return_1d         = EXCLUDED.return_1d,
        return_3d         = EXCLUDED.return_3d,
        return_5d         = EXCLUDED.return_5d,
        forward_return_1d = EXCLUDED.forward_return_1d,
        forward_return_3d = EXCLUDED.forward_return_3d,
        forward_return_5d = EXCLUDED.forward_return_5d,
        volume_change_5d  = EXCLUDED.volume_change_5d,
        volatility_5d     = EXCLUDED.volatility_5d,
        log_return        = EXCLUDED.log_return,
        volume_growth     = EXCLUDED.volume_growth,
        clv               = EXCLUDED.clv
    """

    rows = [
        {
            "date": r.get("date"),
            "ticker": r.get("ticker"),
            "close": _float_or_none(r.get("close")),
            "volume": r.get("volume"),
            "return_1d": _float_or_none(r.get("return_1d")),
            "return_3d": _float_or_none(r.get("return_3d")),
            "return_5d": _float_or_none(r.get("return_5d")),
            "forward_return_1d": _float_or_none(r.get("forward_return_1d")),
            "forward_return_3d": _float_or_none(r.get("forward_return_3d")),
            "forward_return_5d": _float_or_none(r.get("forward_return_5d")),
            "volume_change_5d": _float_or_none(r.get("volume_change_5d")),
            "volatility_5d": _float_or_none(r.get("volatility_5d")),
            "log_return": _float_or_none(r.get("log_return")),
            "volume_growth": _float_or_none(r.get("volume_growth")),
            "clv": _float_or_none(r.get("clv")),
        }
        for r in df.to_dict(orient="records")
    ]

    raw_conn = eng.raw_connection()
    try:
        from psycopg2.extras import execute_batch
        with raw_conn.cursor() as cur:
            execute_batch(cur, sql, rows, page_size=100)
        raw_conn.commit()
        logger.info("upsert_market_features saved=%s", len(rows))
        return len(rows)
    except Exception as exc:
        raw_conn.rollback()
        logger.error("upsert_market_features failed: %s", exc)
        raise
    finally:
        raw_conn.close()


def _float_or_none(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None
