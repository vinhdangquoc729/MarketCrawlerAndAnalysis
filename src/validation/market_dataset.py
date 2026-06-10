"""Build validation datasets/views for dashboard and market validation."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.storage.db import get_engine, query_dataframe
from src.validation.validation_queries import (
    CREATE_MARKET_FEATURES_TABLE_SQL,
    CREATE_MARKET_PRICES_TABLE_SQL,
    CREATE_SENTIMENT_ASPECT_MARKET_VIEW_SQL,
    CREATE_SENTIMENT_EVIDENCE_VIEW_SQL,
    CREATE_SENTIMENT_MARKET_FORWARD_VIEW_SQL,
    VALIDATION_VIEW_COUNT_SQL,
)

logger = logging.getLogger(__name__)

REQUIRED_PRICE_COLUMNS = {
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def ensure_validation_schema() -> None:
    """Create market validation tables."""
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text(CREATE_MARKET_PRICES_TABLE_SQL))
        conn.execute(text(CREATE_MARKET_FEATURES_TABLE_SQL))

    logger.info("Validation schema is ready.")


def normalize_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV price dataframe.

    Required output:
    date,ticker,open,high,low,close,volume,source
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "source",
            ]
        )

    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]

    rename_map = {
        "time": "date",
        "tradingdate": "date",
        "trading_date": "date",
        "datetime": "date",
        "vol": "volume",
    }
    df = df.rename(columns=rename_map)

    missing = REQUIRED_PRICE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required price columns: {sorted(missing)}. "
            f"Got columns={list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "source" not in df.columns:
        df["source"] = "unknown"

    df = df.dropna(subset=["date", "ticker", "close"])
    df = df.sort_values(["ticker", "date"])
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")

    return df[
        [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source",
        ]
    ]


def upsert_market_prices(df: pd.DataFrame) -> int:
    """Upsert OHLCV rows into market_prices."""
    if df.empty:
        logger.warning("No market prices to upsert.")
        return 0

    engine = get_engine()

    sql = """
    INSERT INTO market_prices (
        date,
        ticker,
        open,
        high,
        low,
        close,
        volume,
        source
    )
    VALUES (
        :date,
        :ticker,
        :open,
        :high,
        :low,
        :close,
        :volume,
        :source
    )
    ON CONFLICT (date, ticker)
    DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        source = EXCLUDED.source,
        updated_at = CURRENT_TIMESTAMP
    """

    rows = df.where(pd.notnull(df), None).to_dict(orient="records")

    with engine.begin() as conn:
        for row in rows:
            conn.execute(text(sql), row)

    logger.info("Upserted market_prices rows=%s", len(rows))
    return len(rows)


def import_market_prices_from_csv(csv_path: str | Path) -> int:
    """Import OHLCV data from CSV into market_prices."""
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Market price CSV not found: {path}")

    raw_df = pd.read_csv(path)
    normalized_df = normalize_price_dataframe(raw_df)

    ensure_validation_schema()
    return upsert_market_prices(normalized_df)


def load_market_prices() -> pd.DataFrame:
    """Load market_prices from database."""
    try:
        return query_dataframe(
            """
            SELECT
                date,
                ticker,
                open,
                high,
                low,
                close,
                volume,
                source
            FROM market_prices
            ORDER BY ticker, date
            """
        )
    except Exception as exc:
        logger.warning("Could not load market_prices: %s", exc)
        return pd.DataFrame()


def build_market_features_from_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    """Build return, forward return, volume change and volatility."""
    if price_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "close",
                "volume",
                "return_1d",
                "return_3d",
                "return_5d",
                "forward_return_1d",
                "forward_return_3d",
                "forward_return_5d",
                "volume_change_5d",
                "volatility_5d",
            ]
        )

    df = price_df.copy()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "close"])
    df = df.sort_values(["ticker", "date"])

    group = df.groupby("ticker", group_keys=False)

    df["return_1d"] = group["close"].pct_change(1)
    df["return_3d"] = group["close"].pct_change(3)
    df["return_5d"] = group["close"].pct_change(5)

    df["forward_return_1d"] = group["close"].shift(-1) / df["close"] - 1
    df["forward_return_3d"] = group["close"].shift(-3) / df["close"] - 1
    df["forward_return_5d"] = group["close"].shift(-5) / df["close"] - 1

    df["volume_ma5"] = group["volume"].transform(
        lambda series: series.rolling(window=5, min_periods=2).mean()
    )
    df["volume_change_5d"] = df["volume"] / df["volume_ma5"] - 1

    df["volatility_5d"] = group["return_1d"].transform(
        lambda series: series.rolling(window=5, min_periods=2).std()
    )

    keep_cols = [
        "date",
        "ticker",
        "close",
        "volume",
        "return_1d",
        "return_3d",
        "return_5d",
        "forward_return_1d",
        "forward_return_3d",
        "forward_return_5d",
        "volume_change_5d",
        "volatility_5d",
    ]

    return df[keep_cols]


def upsert_market_features(feature_df: pd.DataFrame) -> int:
    """Upsert engineered market features into market_features."""
    if feature_df.empty:
        logger.warning("No market features to upsert.")
        return 0

    engine = get_engine()

    sql = """
    INSERT INTO market_features (
        date,
        ticker,
        close,
        volume,
        return_1d,
        return_3d,
        return_5d,
        forward_return_1d,
        forward_return_3d,
        forward_return_5d,
        volume_change_5d,
        volatility_5d
    )
    VALUES (
        :date,
        :ticker,
        :close,
        :volume,
        :return_1d,
        :return_3d,
        :return_5d,
        :forward_return_1d,
        :forward_return_3d,
        :forward_return_5d,
        :volume_change_5d,
        :volatility_5d
    )
    ON CONFLICT (date, ticker)
    DO UPDATE SET
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        return_1d = EXCLUDED.return_1d,
        return_3d = EXCLUDED.return_3d,
        return_5d = EXCLUDED.return_5d,
        forward_return_1d = EXCLUDED.forward_return_1d,
        forward_return_3d = EXCLUDED.forward_return_3d,
        forward_return_5d = EXCLUDED.forward_return_5d,
        volume_change_5d = EXCLUDED.volume_change_5d,
        volatility_5d = EXCLUDED.volatility_5d
    """

    rows = feature_df.where(pd.notnull(feature_df), None).to_dict(orient="records")

    with engine.begin() as conn:
        for row in rows:
            conn.execute(text(sql), row)

    logger.info("Upserted market_features rows=%s", len(rows))
    return len(rows)


def create_validation_views() -> None:
    """Create/recreate dashboard validation views.

    DROP VIEW is used because PostgreSQL cannot remove columns using
    CREATE OR REPLACE VIEW when the old view has a different structure.
    """
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text(CREATE_SENTIMENT_EVIDENCE_VIEW_SQL))
        conn.execute(text(CREATE_SENTIMENT_MARKET_FORWARD_VIEW_SQL))
        conn.execute(text(CREATE_SENTIMENT_ASPECT_MARKET_VIEW_SQL))

    logger.info("Created validation/dashboard views.")


def build_market_dataset() -> dict[str, int]:
    """Build market_features and validation views from existing market_prices."""
    ensure_validation_schema()

    price_df = load_market_prices()
    logger.info("Loaded market_prices rows=%s", len(price_df))

    feature_df = build_market_features_from_prices(price_df)
    saved = upsert_market_features(feature_df)

    create_validation_views()

    return {
        "market_prices": len(price_df),
        "market_features": saved,
    }


def get_validation_counts() -> pd.DataFrame:
    """Return row counts of dashboard/validation objects."""
    try:
        return query_dataframe(VALIDATION_VIEW_COUNT_SQL)
    except Exception as exc:
        logger.warning("Could not query validation counts: %s", exc)
        return pd.DataFrame()


def get_market_coverage() -> pd.DataFrame:
    """Return market price coverage by ticker."""
    try:
        return query_dataframe(
            """
            SELECT
                ticker,
                MIN(date) AS min_date,
                MAX(date) AS max_date,
                COUNT(*) AS row_count
            FROM market_prices
            GROUP BY ticker
            ORDER BY ticker
            """
        )
    except Exception as exc:
        logger.warning("Could not query market coverage: %s", exc)
        return pd.DataFrame()


def get_sentiment_market_overlap() -> pd.DataFrame:
    """Return overlap summary between sentiment and market features."""
    try:
        return query_dataframe(
            """
            SELECT
                ticker,
                MIN(date) AS min_date,
                MAX(date) AS max_date,
                COUNT(*) AS row_count
            FROM sentiment_market_forward_dataset
            GROUP BY ticker
            ORDER BY row_count DESC, ticker
            """
        )
    except Exception as exc:
        logger.warning("Could not query sentiment-market overlap: %s", exc)
        return pd.DataFrame()