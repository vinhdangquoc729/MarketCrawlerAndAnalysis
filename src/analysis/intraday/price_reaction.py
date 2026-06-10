"""Measure intraday price reaction to news sentiment.

For each article, finds the first 15m bar at or after publication and
computes correlation between sentiment_score and forward bar returns.

Outputs:
  data/results/intraday/price_reaction_summary.csv   - avg return by sentiment label x bar
  data/results/intraday/price_reaction_correlations.csv - correlation by bar lag x timing bucket
  data/results/intraday/price_reaction_by_ticker.csv

Usage:
  python -m src.analysis.intraday.price_reaction
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.intraday._loader import build_article_bar_dataset
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
OUT_DIR = Path("data/results/intraday")
FORWARD_BARS = 8


def _ttest(series: pd.Series) -> tuple[float, float]:
    clean = series.dropna()
    if len(clean) < 10:
        return np.nan, np.nan
    t, p = stats.ttest_1samp(clean, 0)
    return float(t), float(p)


def run(min_date: str = "2023-10-01") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_article_bar_dataset(min_date=min_date, forward_bars=FORWARD_BARS)
    if df.empty:
        print("No data. Check that intraday_prices overlaps with article dates.")
        return

    # Drop midnight-only (no real time signal)
    df = df[df["timing"] != "midnight_only"].copy()
    print(f"Working dataset: {len(df)} article-ticker rows, {df['ticker'].nunique()} tickers")
    print(f"Timing distribution:\n{df['timing'].value_counts().to_string()}\n")

    fwd_cols = [f"fwd_bar_{i}" for i in range(1, FORWARD_BARS + 1)]

    # 1. Average forward return by sentiment label x bar lag
    summary_rows = []
    for label in ["positive", "neutral", "negative"]:
        sub = df[df["final_sentiment"] == label]
        for col in ["bar_log_return"] + fwd_cols + ["fwd_cum"]:
            bar_idx = 0 if col == "bar_log_return" else (
                int(col.split("_")[-1]) if col.startswith("fwd_bar") else 99
            )
            t, p = _ttest(sub[col])
            summary_rows.append({
                "sentiment": label,
                "bar": col,
                "bar_idx": bar_idx,
                "n": sub[col].dropna().__len__(),
                "mean_return": sub[col].mean(),
                "median_return": sub[col].median(),
                "t_stat": t,
                "p_value": p,
            })

    summary = pd.DataFrame(summary_rows).sort_values(["bar_idx", "sentiment"])
    summary.to_csv(OUT_DIR / "price_reaction_summary.csv", index=False)

    print("=== Avg return by sentiment label x bar (bar_0 = matched bar, fwd_bar_1 = next bar) ===")
    pivot = summary[summary["bar_idx"] <= 4].pivot_table(
        index="bar", columns="sentiment", values="mean_return"
    )
    print(pivot.round(6).to_string())

    # 2. Pearson correlation: sentiment_score vs forward bar returns, by timing bucket
    corr_rows = []
    for timing in df["timing"].unique():
        sub = df[df["timing"] == timing]
        for col in ["bar_log_return"] + fwd_cols:
            bar_idx = 0 if col == "bar_log_return" else int(col.split("_")[-1])
            valid = sub[["sentiment_score", col]].dropna()
            if len(valid) < 20:
                continue
            r, p = stats.pearsonr(valid["sentiment_score"], valid[col])
            corr_rows.append({
                "timing": timing,
                "bar": col,
                "bar_idx": bar_idx,
                "n": len(valid),
                "correlation": r,
                "p_value": p,
            })

    corr_df = pd.DataFrame(corr_rows).sort_values(["timing", "bar_idx"])
    corr_df.to_csv(OUT_DIR / "price_reaction_correlations.csv", index=False)

    print("\n=== Sentiment-return correlation by timing bucket x bar lag ===")
    pivot2 = corr_df[corr_df["bar_idx"] <= 4].pivot_table(
        index="bar", columns="timing", values="correlation"
    )
    print(pivot2.round(4).to_string())

    # 3. Per-ticker correlation at bar_0 and fwd_bar_1
    ticker_rows = []
    for ticker, grp in df.groupby("ticker"):
        for col, label in [("bar_log_return", "bar_0"), ("fwd_bar_1", "fwd_bar_1")]:
            valid = grp[["sentiment_score", col]].dropna()
            if len(valid) < 10:
                continue
            r, p = stats.pearsonr(valid["sentiment_score"], valid[col])
            ticker_rows.append({"ticker": ticker, "bar": label, "n": len(valid), "correlation": r, "p_value": p})

    ticker_df = pd.DataFrame(ticker_rows)
    ticker_df.to_csv(OUT_DIR / "price_reaction_by_ticker.csv", index=False)

    print("\n=== Per-ticker correlation at bar_0 (top 10) ===")
    bar0 = ticker_df[ticker_df["bar"] == "bar_0"].sort_values("correlation", ascending=False)
    print(bar0.head(10)[["ticker", "n", "correlation", "p_value"]].to_string(index=False))
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    setup_logging()
    run()
