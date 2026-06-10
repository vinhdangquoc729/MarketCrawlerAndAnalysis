"""SQL queries for sentiment-market validation and dashboard views."""
from __future__ import annotations


CREATE_MARKET_PRICES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS market_prices (
    date DATE NOT NULL,
    ticker TEXT NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    source TEXT DEFAULT 'unknown',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_market_prices_ticker
ON market_prices(ticker);

CREATE INDEX IF NOT EXISTS idx_market_prices_date
ON market_prices(date);
"""


CREATE_MARKET_FEATURES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS market_features (
    date DATE NOT NULL,
    ticker TEXT NOT NULL,
    close FLOAT,
    volume BIGINT,
    return_1d FLOAT,
    return_3d FLOAT,
    return_5d FLOAT,
    forward_return_1d FLOAT,
    forward_return_3d FLOAT,
    forward_return_5d FLOAT,
    volume_change_5d FLOAT,
    volatility_5d FLOAT,
    log_return FLOAT,
    volume_growth FLOAT,
    clv FLOAT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_market_features_ticker
ON market_features(ticker);

CREATE INDEX IF NOT EXISTS idx_market_features_date
ON market_features(date);

ALTER TABLE market_features ADD COLUMN IF NOT EXISTS log_return FLOAT;
ALTER TABLE market_features ADD COLUMN IF NOT EXISTS volume_growth FLOAT;
ALTER TABLE market_features ADD COLUMN IF NOT EXISTS clv FLOAT;
"""


CREATE_SENTIMENT_EVIDENCE_VIEW_SQL = """
DROP VIEW IF EXISTS sentiment_evidence_view CASCADE;

CREATE VIEW sentiment_evidence_view AS
SELECT
    es.id AS sentiment_id,
    a.article_id,
    a.title,
    a.url,
    a.category,
    a.published_at,
    a.published_at AT TIME ZONE 'Asia/Ho_Chi_Minh' AS published_at_vn,
    ae.ticker,
    tm.company_name,
    tm.sector,
    ae.entity_text,
    ae.context,
    ea.aspect,
    ea.aspect_keywords,
    ea.aspect_score,
    es.sentiment_label,
    es.sentiment_score,
    es.confidence,
    es.prob_negative,
    es.prob_neutral,
    es.prob_positive,
    es.model_version,
    es.created_at
FROM entity_sentiments es
JOIN entity_aspects ea
    ON es.entity_aspect_id = ea.id
JOIN article_entities ae
    ON es.entity_id = ae.id
JOIN articles a
    ON es.article_id = a.article_id
LEFT JOIN ticker_master tm
    ON ae.ticker = tm.ticker
WHERE es.inference_status = 'success';
"""


CREATE_SENTIMENT_MARKET_FORWARD_VIEW_SQL = """
DROP VIEW IF EXISTS sentiment_market_forward_dataset CASCADE;

CREATE VIEW sentiment_market_forward_dataset AS
SELECT
    s.date,
    s.ticker,
    s.sector,
    s.sentiment_index,
    s.final_sentiment,
    s.article_count,
    s.positive_count,
    s.neutral_count,
    s.negative_count,
    s.avg_confidence,
    s.trust_level,
    m.close,
    m.volume,
    m.return_1d,
    m.return_3d,
    m.return_5d,
    m.forward_return_1d,
    m.forward_return_3d,
    m.forward_return_5d,
    m.volume_change_5d,
    m.volatility_5d,
    m.log_return,
    m.volume_growth,
    m.clv
FROM daily_sentiment_index s
JOIN market_features m
    ON s.date = m.date
   AND s.ticker = m.ticker;
"""


CREATE_SENTIMENT_ASPECT_MARKET_VIEW_SQL = """
DROP VIEW IF EXISTS sentiment_aspect_market_dataset CASCADE;

CREATE VIEW sentiment_aspect_market_dataset AS
WITH aspect_base AS (
    SELECT
        DATE(
            COALESCE(a.published_at, a.crawl_at)
            AT TIME ZONE 'Asia/Ho_Chi_Minh'
        ) AS article_date,
        ae.ticker,
        ea.aspect,
        es.sentiment_score,
        es.confidence,
        es.sentiment_label
    FROM entity_sentiments es
    JOIN entity_aspects ea
        ON es.entity_aspect_id = ea.id
    JOIN article_entities ae
        ON es.entity_id = ae.id
    JOIN articles a
        ON es.article_id = a.article_id
    WHERE es.inference_status = 'success'
      AND COALESCE(a.published_at, a.crawl_at) IS NOT NULL
),
aspect_mapped AS (
    SELECT
        ab.*,
        (
            SELECT MIN(mf.date)
            FROM market_features mf
            WHERE mf.ticker = ab.ticker
              AND mf.date >= ab.article_date
        ) AS trading_date
    FROM aspect_base ab
),
aspect_daily AS (
    SELECT
        trading_date AS date,
        ticker,
        aspect,
        AVG(sentiment_score * COALESCE(confidence, 1.0)) AS aspect_sentiment_score,
        AVG(confidence) AS avg_confidence,
        COUNT(*) AS sample_count,
        SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive_count,
        SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) AS neutral_count,
        SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative_count
    FROM aspect_mapped
    WHERE trading_date IS NOT NULL
    GROUP BY
        trading_date,
        ticker,
        aspect
)
SELECT
    ad.date,
    ad.ticker,
    tm.sector,
    ad.aspect,
    ad.aspect_sentiment_score,
    ad.avg_confidence,
    ad.sample_count,
    ad.positive_count,
    ad.neutral_count,
    ad.negative_count,
    m.close,
    m.forward_return_1d,
    m.forward_return_3d,
    m.forward_return_5d
FROM aspect_daily ad
JOIN market_features m
    ON ad.date = m.date
   AND ad.ticker = m.ticker
LEFT JOIN ticker_master tm
    ON ad.ticker = tm.ticker;
"""


VALIDATION_VIEW_COUNT_SQL = """
SELECT 'daily_sentiment_index' AS object_name, COUNT(*) AS row_count FROM daily_sentiment_index
UNION ALL
SELECT 'sentiment_evidence_view' AS object_name, COUNT(*) AS row_count FROM sentiment_evidence_view
UNION ALL
SELECT 'market_prices' AS object_name, COUNT(*) AS row_count FROM market_prices
UNION ALL
SELECT 'market_features' AS object_name, COUNT(*) AS row_count FROM market_features
UNION ALL
SELECT 'sentiment_market_forward_dataset' AS object_name, COUNT(*) AS row_count FROM sentiment_market_forward_dataset
UNION ALL
SELECT 'sentiment_aspect_market_dataset' AS object_name, COUNT(*) AS row_count FROM sentiment_aspect_market_dataset;
"""