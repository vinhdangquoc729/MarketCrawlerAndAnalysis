"""Build ticker_aliases from ticker_master."""
from __future__ import annotations

import logging
import re

from dotenv import load_dotenv
from unidecode import unidecode

from src.storage.db import get_engine, query_dataframe
from src.utils.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)

# Common Vietnamese company-type prefixes/suffixes to strip for short names
_STRIP_PATTERNS = [
    r"\bCông\s+ty\s+Cổ\s+phần\b",
    r"\bCong\s+ty\s+Co\s+phan\b",
    r"\bCTCP\b",
    r"\bCT\.?\s*CP\b",
    r"\bJoint\s+Stock\s+Company\b",
    r"\bJSC\b",
    r"\bCo\.\s*Ltd\.?\b",
    r"\bLimited\b",
    r"\bCorporation\b",
    r"\bGroup\b",
]
_STRIP_RE = re.compile("|".join(_STRIP_PATTERNS), flags=re.IGNORECASE)


def _normalize(text: str) -> str:
    """Lowercase + remove Vietnamese diacritics + collapse whitespace."""
    return re.sub(r"\s+", " ", unidecode(text).lower()).strip()


def _make_short_name(company_name: str) -> str | None:
    """Strip common company-type tokens to get a short display name."""
    short = _STRIP_RE.sub("", company_name).strip(" -–—")
    short = re.sub(r"\s+", " ", short).strip()
    return short if short and short != company_name else None


def _generate_aliases(ticker: str, company_name: str | None) -> list[dict]:
    """Return all alias rows for one ticker."""
    aliases: list[dict] = []
    seen: set[str] = set()

    def add(alias: str, alias_type: str, weight: float) -> None:
        key = alias.lower().strip()
        if key and key not in seen:
            seen.add(key)
            aliases.append({
                "ticker": ticker,
                "alias": alias.strip(),
                "alias_type": alias_type,
                "weight": weight,
                "alias_normalized": _normalize(alias),
            })

    # Ticker symbol itself is the strongest signal
    add(ticker, "ticker", 1.0)

    if not company_name:
        return aliases

    add(company_name, "company_name", 0.9)

    short = _make_short_name(company_name)
    if short:
        add(short, "short_name", 0.85)

    # ASCII-normalized version of the full name for fuzzy matching
    normalized = _normalize(company_name)
    if normalized != company_name.lower():
        add(normalized, "company_name_normalized", 0.8)

    return aliases


def build_ticker_aliases() -> int:
    """Generate alias rows for all active tickers and upsert into ticker_aliases.

    Returns the number of rows upserted.
    """
    df = query_dataframe(
        "SELECT ticker, company_name FROM ticker_master WHERE is_active = TRUE"
    )

    if df.empty:
        logger.warning("ticker_master is empty — run build_ticker_master first.")
        return 0

    logger.info("Building aliases for tickers=%s", len(df))

    all_rows: list[dict] = []
    for _, row in df.iterrows():
        all_rows.extend(_generate_aliases(row["ticker"], row.get("company_name")))

    if not all_rows:
        return 0

    sql = """
    INSERT INTO ticker_aliases (ticker, alias, alias_type, weight, alias_normalized)
    VALUES (%(ticker)s, %(alias)s, %(alias_type)s, %(weight)s, %(alias_normalized)s)
    ON CONFLICT (ticker, alias) DO UPDATE SET
        alias_type       = EXCLUDED.alias_type,
        weight           = EXCLUDED.weight,
        alias_normalized = EXCLUDED.alias_normalized
    """

    engine = get_engine()
    raw_conn = engine.raw_connection()
    try:
        from psycopg2.extras import execute_batch
        with raw_conn.cursor() as cur:
            execute_batch(cur, sql, all_rows, page_size=200)
        raw_conn.commit()
        logger.info("ticker_aliases upserted rows=%s", len(all_rows))
        return len(all_rows)
    except Exception as exc:
        raw_conn.rollback()
        logger.error("build_ticker_aliases failed: %s", exc)
        raise
    finally:
        raw_conn.close()


def main() -> None:
    setup_logging()
    count = build_ticker_aliases()
    logger.info("aliases_builder complete rows=%s", count)


if __name__ == "__main__":
    main()
