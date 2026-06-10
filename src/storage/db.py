"""Database access layer for the market sentiment pipeline."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any


def _dt_now() -> datetime:
    return datetime.now(timezone.utc)

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()
logger = logging.getLogger(__name__)

_engine: Engine | None = None


def get_engine() -> Engine:
    """Return the singleton SQLAlchemy engine."""
    global _engine
    if _engine is None:
        host = os.getenv("DB_HOST") or "localhost"
        port = os.getenv("DB_PORT") or "5433"
        name = os.getenv("DB_NAME") or "market_sentiment"
        user = os.getenv("DB_USER") or "postgres"
        password = os.getenv("DB_PASSWORD") or ""
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
        _engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
        logger.info("Created DB engine host=%s port=%s db=%s", host, port, name)
    return _engine


# Core pipeline tables (market tables are created by market_storage.ensure_market_schema)
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    article_id       TEXT PRIMARY KEY,
    url              TEXT UNIQUE NOT NULL,
    title            TEXT,
    sapo             TEXT,
    content          TEXT,
    category         TEXT,
    crawl_at         TIMESTAMPTZ,
    published_at     TIMESTAMPTZ,
    status           TEXT NOT NULL DEFAULT 'raw',
    content_hash     TEXT,
    detected_tickers TEXT[],
    error_message    TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_articles_status   ON articles(status);
CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);
CREATE INDEX IF NOT EXISTS idx_articles_crawl_at ON articles(crawl_at DESC);

CREATE TABLE IF NOT EXISTS article_relevance (
    article_id        TEXT PRIMARY KEY REFERENCES articles(article_id) ON DELETE CASCADE,
    relevance_type    TEXT,
    relevance_score   FLOAT,
    decision          TEXT,
    reason            TEXT,
    detected_tickers  TEXT[],
    detected_keywords TEXT[],
    created_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ticker_master (
    ticker       TEXT PRIMARY KEY,
    company_name TEXT,
    sector       TEXT,
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ticker_aliases (
    id               SERIAL PRIMARY KEY,
    ticker           TEXT NOT NULL REFERENCES ticker_master(ticker) ON DELETE CASCADE,
    alias            TEXT NOT NULL,
    alias_type       TEXT,
    weight           FLOAT NOT NULL DEFAULT 1.0,
    alias_normalized TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, alias)
);
CREATE INDEX IF NOT EXISTS idx_ticker_aliases_ticker           ON ticker_aliases(ticker);
CREATE INDEX IF NOT EXISTS idx_ticker_aliases_alias_normalized ON ticker_aliases(alias_normalized);

CREATE TABLE IF NOT EXISTS article_entities (
    id               SERIAL PRIMARY KEY,
    article_id       TEXT NOT NULL REFERENCES articles(article_id) ON DELETE CASCADE,
    ticker           TEXT NOT NULL,
    entity_text      TEXT,
    alias_type       TEXT,
    alias_weight     FLOAT,
    sentence_index   INT,
    context          TEXT,
    relevance_weight FLOAT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(article_id, ticker, sentence_index)
);
CREATE INDEX IF NOT EXISTS idx_article_entities_article_id ON article_entities(article_id);
CREATE INDEX IF NOT EXISTS idx_article_entities_ticker     ON article_entities(ticker);

CREATE TABLE IF NOT EXISTS entity_aspects (
    id              SERIAL PRIMARY KEY,
    entity_id       INT NOT NULL REFERENCES article_entities(id) ON DELETE CASCADE,
    aspect          TEXT NOT NULL,
    aspect_keywords TEXT[],
    aspect_score    FLOAT,
    model_input     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_id, aspect)
);
CREATE INDEX IF NOT EXISTS idx_entity_aspects_entity_id ON entity_aspects(entity_id);

CREATE TABLE IF NOT EXISTS entity_sentiments (
    id               SERIAL PRIMARY KEY,
    entity_aspect_id INT  NOT NULL REFERENCES entity_aspects(id) ON DELETE CASCADE,
    entity_id        INT  NOT NULL,
    article_id       TEXT NOT NULL,
    ticker           TEXT,
    aspect           TEXT,
    sentiment_label  TEXT,
    sentiment_score  FLOAT,
    confidence       FLOAT,
    prob_negative    FLOAT,
    prob_neutral     FLOAT,
    prob_positive    FLOAT,
    model_version    TEXT,
    inference_status TEXT NOT NULL DEFAULT 'success',
    error_message    TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_aspect_id, model_version)
);
CREATE INDEX IF NOT EXISTS idx_entity_sentiments_ticker           ON entity_sentiments(ticker);
CREATE INDEX IF NOT EXISTS idx_entity_sentiments_article_id       ON entity_sentiments(article_id);
CREATE INDEX IF NOT EXISTS idx_entity_sentiments_inference_status ON entity_sentiments(inference_status);

CREATE TABLE IF NOT EXISTS article_ticker_sentiment (
    article_id      TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    sentiment_score FLOAT,
    final_sentiment TEXT,
    confidence      FLOAT,
    aspect_count    INT,
    positive_count  INT,
    neutral_count   INT,
    negative_count  INT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(article_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_article_ticker_sentiment_ticker ON article_ticker_sentiment(ticker);

CREATE TABLE IF NOT EXISTS daily_sentiment_index (
    date            DATE NOT NULL,
    ticker          TEXT NOT NULL,
    sector          TEXT,
    sentiment_index FLOAT,
    final_sentiment TEXT,
    article_count   INT,
    positive_count  INT,
    neutral_count   INT,
    negative_count  INT,
    avg_confidence  FLOAT,
    trust_level     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_daily_sentiment_index_ticker ON daily_sentiment_index(ticker);
CREATE INDEX IF NOT EXISTS idx_daily_sentiment_index_date   ON daily_sentiment_index(date DESC);
"""


def execute_schema() -> None:
    """Create all core pipeline tables if they do not exist."""
    eng = get_engine()
    raw_conn = eng.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        raw_conn.commit()
        logger.info("Database schema executed successfully.")
    except Exception as exc:
        raw_conn.rollback()
        logger.error("execute_schema failed: %s", exc)
        raise
    finally:
        raw_conn.close()


def query_dataframe(sql: str, engine: Engine | None = None) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame."""
    eng = engine or get_engine()
    try:
        with eng.connect() as conn:
            return pd.read_sql(text(sql), conn)
    except Exception as exc:
        logger.warning("query_dataframe failed: %s", exc)
        return pd.DataFrame()


def bulk_upsert_articles(articles: list[dict[str, Any]]) -> int:
    """Insert or update articles in bulk, keyed on URL."""
    if not articles:
        return 0

    sql = """
    INSERT INTO articles
        (article_id, url, title, sapo, content, category,
         crawl_at, published_at, status, content_hash,
         detected_tickers, error_message)
    VALUES
        (%(article_id)s, %(url)s, %(title)s, %(sapo)s, %(content)s, %(category)s,
         %(crawl_at)s, %(published_at)s, %(status)s, %(content_hash)s,
         %(detected_tickers)s, %(error_message)s)
    ON CONFLICT (url) DO UPDATE SET
        title            = EXCLUDED.title,
        sapo             = EXCLUDED.sapo,
        content          = EXCLUDED.content,
        category         = EXCLUDED.category,
        published_at     = EXCLUDED.published_at,
        content_hash     = EXCLUDED.content_hash,
        detected_tickers = EXCLUDED.detected_tickers,
        error_message    = EXCLUDED.error_message,
        status           = CASE
            WHEN articles.status = 'raw' THEN EXCLUDED.status
            ELSE articles.status
        END,
        updated_at = CURRENT_TIMESTAMP
    """

    from src.utils.hashing import make_article_id

    rows = [
        {
            "article_id": a.get("article_id") or make_article_id("cafef", a["url"]),
            "url": a["url"],
            "title": a.get("title"),
            "sapo": a.get("sapo"),
            "content": a.get("content"),
            "category": a.get("category"),
            "crawl_at": a.get("crawl_at") or _dt_now(),
            "published_at": a.get("published_at"),
            "status": a.get("status", "raw"),
            "content_hash": a.get("content_hash"),
            "detected_tickers": a.get("detected_tickers"),
            "error_message": a.get("error_message"),
        }
        for a in articles
    ]

    eng = get_engine()
    raw_conn = eng.raw_connection()
    try:
        from psycopg2.extras import execute_batch
        with raw_conn.cursor() as cur:
            execute_batch(cur, sql, rows, page_size=100)
        raw_conn.commit()
        logger.info("bulk_upsert_articles saved=%s", len(rows))
        return len(rows)
    except Exception as exc:
        raw_conn.rollback()
        logger.error("bulk_upsert_articles failed: %s", exc)
        raise
    finally:
        raw_conn.close()


def load_ticker_aliases(engine: Engine | None = None) -> pd.DataFrame:
    """Load all ticker aliases as a DataFrame."""
    eng = engine or get_engine()
    sql = """
    SELECT ticker, alias, alias_type, weight, alias_normalized
    FROM ticker_aliases
    ORDER BY ticker, weight DESC
    """
    try:
        with eng.connect() as conn:
            return pd.read_sql(text(sql), conn)
    except Exception as exc:
        logger.warning("load_ticker_aliases failed: %s", exc)
        return pd.DataFrame(
            columns=["ticker", "alias", "alias_type", "weight", "alias_normalized"]
        )


def bulk_upsert_article_relevance(
    results: list[dict[str, Any]],
    engine: Engine | None = None,
) -> int:
    """Bulk insert/update article relevance records."""
    if not results:
        return 0
    eng = engine or get_engine()

    sql = """
    INSERT INTO article_relevance
        (article_id, relevance_type, relevance_score, decision, reason,
         detected_tickers, detected_keywords)
    VALUES
        (%(article_id)s, %(relevance_type)s, %(relevance_score)s, %(decision)s, %(reason)s,
         %(detected_tickers)s, %(detected_keywords)s)
    ON CONFLICT (article_id) DO UPDATE SET
        relevance_type    = EXCLUDED.relevance_type,
        relevance_score   = EXCLUDED.relevance_score,
        decision          = EXCLUDED.decision,
        reason            = EXCLUDED.reason,
        detected_tickers  = EXCLUDED.detected_tickers,
        detected_keywords = EXCLUDED.detected_keywords
    """

    rows = [
        {
            "article_id": r["article_id"],
            "relevance_type": r.get("relevance_type"),
            "relevance_score": r.get("relevance_score"),
            "decision": r["decision"],
            "reason": r.get("reason"),
            "detected_tickers": r.get("detected_tickers"),
            "detected_keywords": r.get("detected_keywords"),
        }
        for r in results
    ]

    raw_conn = eng.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.executemany(sql, rows)
        raw_conn.commit()
        return len(rows)
    except Exception as exc:
        raw_conn.rollback()
        logger.error("bulk_upsert_article_relevance failed: %s", exc)
        raise
    finally:
        raw_conn.close()


def upsert_article_relevance(result: dict[str, Any], engine: Engine | None = None) -> int:
    """Insert or update a single article relevance record."""
    eng = engine or get_engine()

    sql = """
    INSERT INTO article_relevance
        (article_id, relevance_type, relevance_score, decision, reason,
         detected_tickers, detected_keywords)
    VALUES
        (%(article_id)s, %(relevance_type)s, %(relevance_score)s, %(decision)s, %(reason)s,
         %(detected_tickers)s, %(detected_keywords)s)
    ON CONFLICT (article_id) DO UPDATE SET
        relevance_type    = EXCLUDED.relevance_type,
        relevance_score   = EXCLUDED.relevance_score,
        decision          = EXCLUDED.decision,
        reason            = EXCLUDED.reason,
        detected_tickers  = EXCLUDED.detected_tickers,
        detected_keywords = EXCLUDED.detected_keywords
    """

    row = {
        "article_id": result["article_id"],
        "relevance_type": result.get("relevance_type"),
        "relevance_score": result.get("relevance_score"),
        "decision": result["decision"],
        "reason": result.get("reason"),
        "detected_tickers": result.get("detected_tickers"),
        "detected_keywords": result.get("detected_keywords"),
    }

    raw_conn = eng.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.execute(sql, row)
        raw_conn.commit()
        return 1
    except Exception as exc:
        raw_conn.rollback()
        logger.error(
            "upsert_article_relevance failed article_id=%s: %s",
            result.get("article_id"),
            exc,
        )
        raise
    finally:
        raw_conn.close()


def fetch_relevant_articles(
    limit: int | None = None,
    since_date: str | None = None,
) -> list[dict[str, Any]]:
    """Return articles with decision='process_sentiment' for entity extraction.

    since_date: if given (YYYY-MM-DD), only return articles crawled on or after that date.
    """
    params: dict[str, Any] = {}
    since_clause = ""
    if since_date:
        since_clause = "AND a.crawl_at >= CAST(:since_date AS timestamptz)"
        params["since_date"] = since_date

    sql = f"""
    SELECT
        a.article_id,
        a.title,
        a.sapo,
        a.content,
        r.detected_tickers
    FROM articles a
    JOIN article_relevance r ON a.article_id = r.article_id
    WHERE r.decision = 'process_sentiment'
      {since_clause}
    ORDER BY a.crawl_at DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"

    try:
        eng = get_engine()
        with eng.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params or None)
        return df.to_dict(orient="records")
    except Exception as exc:
        logger.warning("fetch_relevant_articles failed: %s", exc)
        return []


def fetch_ticker_aliases() -> list[dict[str, Any]]:
    """Return all ticker alias rows as a list of dicts."""
    sql = """
    SELECT ticker, alias, alias_type, weight, alias_normalized
    FROM ticker_aliases
    """
    try:
        df = query_dataframe(sql)
        return df.to_dict(orient="records")
    except Exception as exc:
        logger.warning("fetch_ticker_aliases failed: %s", exc)
        return []


def upsert_article_entities(rows: list[dict[str, Any]]) -> int:
    """Insert or update article entity records."""
    if not rows:
        return 0

    sql = """
    INSERT INTO article_entities
        (article_id, ticker, entity_text, alias_type, alias_weight,
         sentence_index, context, relevance_weight)
    VALUES
        (%(article_id)s, %(ticker)s, %(entity_text)s, %(alias_type)s, %(alias_weight)s,
         %(sentence_index)s, %(context)s, %(relevance_weight)s)
    ON CONFLICT (article_id, ticker, sentence_index) DO UPDATE SET
        entity_text      = EXCLUDED.entity_text,
        alias_type       = EXCLUDED.alias_type,
        alias_weight     = EXCLUDED.alias_weight,
        context          = EXCLUDED.context,
        relevance_weight = EXCLUDED.relevance_weight
    """

    eng = get_engine()
    raw_conn = eng.raw_connection()
    try:
        from psycopg2.extras import execute_batch
        with raw_conn.cursor() as cur:
            execute_batch(cur, sql, rows, page_size=100)
        raw_conn.commit()
        return len(rows)
    except Exception as exc:
        raw_conn.rollback()
        logger.error("upsert_article_entities failed: %s", exc)
        raise
    finally:
        raw_conn.close()


def fetch_article_entities_without_aspects(limit: int | None = None) -> list[dict[str, Any]]:
    """Return article_entities that have no entity_aspects yet."""
    sql = """
    SELECT
        ae.id,
        ae.article_id,
        ae.ticker,
        ae.context
    FROM article_entities ae
    LEFT JOIN entity_aspects ea ON ae.id = ea.entity_id
    WHERE ea.entity_id IS NULL
    ORDER BY ae.id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"

    try:
        df = query_dataframe(sql)
        return df.to_dict(orient="records")
    except Exception as exc:
        logger.warning("fetch_article_entities_without_aspects failed: %s", exc)
        return []


def upsert_entity_aspects(rows: list[dict[str, Any]]) -> int:
    """Insert or update entity aspect records."""
    if not rows:
        return 0

    sql = """
    INSERT INTO entity_aspects
        (entity_id, aspect, aspect_keywords, aspect_score, model_input)
    VALUES
        (%(entity_id)s, %(aspect)s, %(aspect_keywords)s, %(aspect_score)s, %(model_input)s)
    ON CONFLICT (entity_id, aspect) DO UPDATE SET
        aspect_keywords = EXCLUDED.aspect_keywords,
        aspect_score    = EXCLUDED.aspect_score,
        model_input     = EXCLUDED.model_input
    """

    normalized = [
        {
            "entity_id": r["entity_id"],
            "aspect": r["aspect"],
            "aspect_keywords": r.get("aspect_keywords"),
            "aspect_score": r.get("aspect_score"),
            "model_input": r.get("model_input"),
        }
        for r in rows
    ]

    eng = get_engine()
    raw_conn = eng.raw_connection()
    try:
        from psycopg2.extras import execute_batch
        with raw_conn.cursor() as cur:
            execute_batch(cur, sql, normalized, page_size=100)
        raw_conn.commit()
        return len(normalized)
    except Exception as exc:
        raw_conn.rollback()
        logger.error("upsert_entity_aspects failed: %s", exc)
        raise
    finally:
        raw_conn.close()


def fetch_entity_aspects_for_inference(
    limit: int, model_version: str
) -> list[dict[str, Any]]:
    """Return entity_aspects not yet inferred for the given model version."""
    eng = get_engine()
    sql = text("""
    SELECT
        ea.id   AS entity_aspect_id,
        ae.id   AS entity_id,
        ae.article_id,
        ae.ticker,
        ea.aspect,
        ea.model_input
    FROM entity_aspects ea
    JOIN article_entities ae ON ea.entity_id = ae.id
    LEFT JOIN entity_sentiments es
           ON ea.id = es.entity_aspect_id
          AND es.model_version = :model_version
    WHERE es.entity_aspect_id IS NULL
    ORDER BY ea.id
    LIMIT :limit
    """)
    try:
        with eng.connect() as conn:
            result = conn.execute(sql, {"model_version": model_version, "limit": limit})
            keys = list(result.keys())
            return [dict(zip(keys, row)) for row in result.fetchall()]
    except Exception as exc:
        logger.warning("fetch_entity_aspects_for_inference failed: %s", exc)
        return []


def upsert_entity_sentiments(
    rows: list[dict[str, Any]],
    model_version: str,
    engine: Engine | None = None,
) -> int:
    """Insert or update entity sentiment predictions."""
    if not rows:
        return 0

    sql = """
    INSERT INTO entity_sentiments
        (entity_aspect_id, entity_id, article_id, ticker, aspect,
         sentiment_label, sentiment_score, confidence,
         prob_negative, prob_neutral, prob_positive,
         model_version, inference_status, error_message)
    VALUES
        (%(entity_aspect_id)s, %(entity_id)s, %(article_id)s, %(ticker)s, %(aspect)s,
         %(sentiment_label)s, %(sentiment_score)s, %(confidence)s,
         %(prob_negative)s, %(prob_neutral)s, %(prob_positive)s,
         %(model_version)s, %(inference_status)s, %(error_message)s)
    ON CONFLICT (entity_aspect_id, model_version) DO UPDATE SET
        sentiment_label  = EXCLUDED.sentiment_label,
        sentiment_score  = EXCLUDED.sentiment_score,
        confidence       = EXCLUDED.confidence,
        prob_negative    = EXCLUDED.prob_negative,
        prob_neutral     = EXCLUDED.prob_neutral,
        prob_positive    = EXCLUDED.prob_positive,
        inference_status = EXCLUDED.inference_status,
        error_message    = EXCLUDED.error_message
    """

    normalized = [{**r, "model_version": model_version} for r in rows]

    eng = engine or get_engine()
    raw_conn = eng.raw_connection()
    try:
        from psycopg2.extras import execute_batch
        with raw_conn.cursor() as cur:
            execute_batch(cur, sql, normalized, page_size=100)
        raw_conn.commit()
        return len(normalized)
    except Exception as exc:
        raw_conn.rollback()
        logger.error("upsert_entity_sentiments failed: %s", exc)
        raise
    finally:
        raw_conn.close()
