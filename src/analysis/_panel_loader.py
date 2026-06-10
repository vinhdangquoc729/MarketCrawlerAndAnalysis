"""Shared panel-loading helper for all analysis scripts.

Reads a CSV produced by ``src.jobs.export_daily_panel`` and applies a
column alias so scripts that reference ``volatility`` find the value stored
as ``volatility_5d`` in the database export.
"""
from __future__ import annotations

import pandas as pd


def load_panel(
    path: str,
    required_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
    fill_missing_sentiment: float | None = None,
    news_days_only: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    # alias: volatility_5d -> volatility (stock_news scripts use "volatility")
    if "volatility" not in df.columns and "volatility_5d" in df.columns:
        df["volatility"] = df["volatility_5d"]

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise SystemExit(f"Panel missing columns: {missing}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in (numeric_cols or []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if fill_missing_sentiment is not None and "sentiment_score" in df.columns:
        df["sentiment_score"] = df["sentiment_score"].fillna(fill_missing_sentiment)

    df = df.dropna(subset=["ticker", "date"]).sort_values(["ticker", "date"])

    if news_days_only:
        df = df[df.get("news_count", pd.Series(0, index=df.index)).fillna(0) > 0]

    return df.reset_index(drop=True)
