"""Build or refresh the ticker_master table from vnstock."""
from __future__ import annotations

import logging

import pandas as pd
from dotenv import load_dotenv

from src.storage.db import get_engine
from src.utils.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)


def _fetch_all_symbols() -> pd.DataFrame:
    """Fetch all listed symbols from vnstock."""
    from vnstock import Vnstock
    stock = Vnstock().stock(symbol="ACB", source="VCI")
    return stock.listing.all_symbols()


def _extract_ticker(row: pd.Series) -> str | None:
    for col in ("ticker", "symbol", "code"):
        if col in row.index and row[col]:
            val = str(row[col]).upper().strip()
            if val:
                return val
    return None


def _extract_company_name(row: pd.Series) -> str | None:
    for col in ("organ_name", "company_name", "full_name", "name"):
        if col in row.index and row[col]:
            val = str(row[col]).strip()
            if val:
                return val
    return None


def _extract_sector(row: pd.Series) -> str | None:
    for col in ("icb_name3", "icb_name2", "icb_name1", "com_group_code", "organ_type_code"):
        if col in row.index and row[col]:
            val = str(row[col]).strip()
            if val:
                return val
    return None


def build_ticker_master() -> int:
    """Fetch all symbols from vnstock and upsert into ticker_master.

    Returns the number of rows upserted.
    """
    logger.info("Fetching all symbols from vnstock...")
    df = _fetch_all_symbols()

    if df is None or df.empty:
        logger.warning("vnstock returned no symbols.")
        return 0

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    logger.info("Fetched symbols=%s columns=%s", len(df), list(df.columns))

    rows = []
    for _, row in df.iterrows():
        ticker = _extract_ticker(row)
        if not ticker:
            continue
        rows.append({
            "ticker": ticker,
            "company_name": _extract_company_name(row),
            "sector": _extract_sector(row),
            "is_active": True,
        })

    if not rows:
        logger.warning("No valid ticker rows after parsing vnstock response.")
        return 0

    sql = """
    INSERT INTO ticker_master (ticker, company_name, sector, is_active, updated_at)
    VALUES (%(ticker)s, %(company_name)s, %(sector)s, %(is_active)s, CURRENT_TIMESTAMP)
    ON CONFLICT (ticker) DO UPDATE SET
        company_name = EXCLUDED.company_name,
        sector       = EXCLUDED.sector,
        is_active    = EXCLUDED.is_active,
        updated_at   = CURRENT_TIMESTAMP
    """

    engine = get_engine()
    raw_conn = engine.raw_connection()
    try:
        from psycopg2.extras import execute_batch
        with raw_conn.cursor() as cur:
            execute_batch(cur, sql, rows, page_size=100)
        raw_conn.commit()
        logger.info("ticker_master upserted rows=%s", len(rows))
        return len(rows)
    except Exception as exc:
        raw_conn.rollback()
        logger.error("build_ticker_master failed: %s", exc)
        raise
    finally:
        raw_conn.close()


def main() -> None:
    setup_logging()
    count = build_ticker_master()
    logger.info("build_ticker_master complete rows=%s", count)


if __name__ == "__main__":
    main()
