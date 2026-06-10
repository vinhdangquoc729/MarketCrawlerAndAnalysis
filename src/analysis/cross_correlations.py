"""
Compute lagged cross-correlations between daily sentiment and trading metrics.

This is the analysis layer of the pipeline — the input panel is produced by
``src.jobs.export_daily_panel`` (which reads from PostgreSQL). No model
inference or raw data loading happens here.

Lag convention:
  corr(x_{t-k}, y_t)
  k > 0 means x leads y by k trading days.
  Example: pair sentiment_score__log_return, lag=1 means yesterday's sentiment
  correlated with today's log return.

Usage:
  python -m src.analysis.cross_correlations
  python -m src.analysis.cross_correlations --panel data/processed/validation/daily_panel.csv
  python -m src.analysis.cross_correlations --news-days-only --ccf-output data/results/ccf_news_days.csv
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis._panel_loader import load_panel

PAIRS = [
    ("sentiment_score", "log_return"),
    ("sentiment_score", "target_up"),
    ("sentiment_score", "volume_growth"),
    ("sentiment_score", "volatility"),
    ("sentiment_score", "clv"),
    ("log_return", "volume_growth"),
]

_REQUIRED = ["ticker", "date", "sentiment_score", "log_return", "volume_growth", "clv"]
_NUMERIC = ["sentiment_score", "log_return", "volume_growth", "clv", "target_up", "volatility"]

_DEFAULT_PANEL = os.path.join(
    os.getenv("MARKET_VALIDATION_OUTPUT_DIR", "data/processed/validation"),
    "daily_panel.csv",
)


def corr_at_lag(group: pd.DataFrame, x_col: str, y_col: str, lag: int) -> tuple[float, int]:
    x = group[x_col].shift(lag)
    y = group[y_col]
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3:
        return np.nan, len(valid)
    if valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan, len(valid)
    return valid.iloc[:, 0].corr(valid.iloc[:, 1]), len(valid)


def compute_cross_correlations(panel: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    rows = []
    for ticker, group in panel.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        for x_col, y_col in PAIRS:
            if x_col not in group.columns or y_col not in group.columns:
                continue
            for lag in range(-max_lag, max_lag + 1):
                corr, n = corr_at_lag(group, x_col, y_col, lag)
                rows.append(
                    {
                        "ticker": ticker,
                        "pair": f"{x_col}__{y_col}",
                        "lag": lag,
                        "correlation": corr,
                        "n": n,
                    }
                )

    all_group = panel.sort_values(["ticker", "date"]).copy()
    for x_col, y_col in PAIRS:
        if x_col not in all_group.columns or y_col not in all_group.columns:
            continue
        for lag in range(-max_lag, max_lag + 1):
            corrs, ns = [], []
            for _, group in all_group.groupby("ticker"):
                corr, n = corr_at_lag(group.reset_index(drop=True), x_col, y_col, lag)
                if not math.isnan(corr):
                    corrs.append(corr)
                    ns.append(n)
            rows.append(
                {
                    "ticker": "ALL_MEAN",
                    "pair": f"{x_col}__{y_col}",
                    "lag": lag,
                    "correlation": float(np.mean(corrs)) if corrs else np.nan,
                    "n": int(np.sum(ns)) if ns else 0,
                }
            )
    return pd.DataFrame(rows)


def print_summary(panel: pd.DataFrame, ccf: pd.DataFrame) -> None:
    print("\nDaily panel:")
    print(f"  rows    : {len(panel)}")
    print(f"  tickers : {sorted(panel['ticker'].unique().tolist())}")
    print(f"  date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")
    news_rows = panel[panel.get("news_count", pd.Series(0, index=panel.index)).fillna(0) > 0]
    print(f"  rows with news: {len(news_rows)}")

    print("\nTop absolute correlations:")
    top = (
        ccf.dropna(subset=["correlation"])
        .assign(abs_corr=lambda d: d["correlation"].abs())
        .sort_values("abs_corr", ascending=False)
        .head(12)
    )
    if top.empty:
        print("  No valid correlations.")
    else:
        print(top[["ticker", "pair", "lag", "correlation", "n"]].to_string(index=False))


def run(
    panel_path: str,
    ccf_output: str,
    max_lag: int,
    news_days_only: bool,
) -> None:
    panel = load_panel(
        panel_path,
        required_cols=_REQUIRED,
        numeric_cols=_NUMERIC,
        news_days_only=news_days_only,
    )
    if "target_up" not in panel.columns:
        panel["target_up"] = np.where(
            panel["log_return"].notna(), (panel["log_return"] > 0).astype(float), np.nan
        )

    print(f"Loaded panel: {len(panel)} rows, {panel['ticker'].nunique()} tickers")

    ccf = compute_cross_correlations(panel, max_lag)

    out = Path(ccf_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ccf.to_csv(out, index=False, encoding="utf-8-sig")

    print_summary(panel, ccf)
    print(f"\nSaved CCF table: {ccf_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lagged cross-correlations: sentiment vs trading metrics")
    parser.add_argument("--panel", default=_DEFAULT_PANEL)
    parser.add_argument("--ccf-output", default="data/results/cross_correlations.csv")
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument(
        "--news-days-only", action="store_true",
        help="Restrict analysis to trading days that had at least one news article",
    )
    args = parser.parse_args()

    run(
        panel_path=args.panel,
        ccf_output=args.ccf_output,
        max_lag=args.max_lag,
        news_days_only=args.news_days_only,
    )
