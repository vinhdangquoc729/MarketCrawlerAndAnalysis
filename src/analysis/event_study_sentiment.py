"""
Event study for news sentiment and future stock returns.

Each trading day with news is treated as an event. Events are grouped by daily
sentiment:
  POS if sentiment_score > threshold
  NEG if sentiment_score < -threshold
  NEU otherwise

For each event, computes future cumulative log returns and abnormal returns:
  future_return_h  = sum log_return over t+1 ... t+h
  future_abret_h   = future_return_h - market average future return over same days

t-statistics use Fama-MacBeth: t-test across per-ticker means to correct for
cross-sectional correlation.

Usage:
  python -m src.analysis.event_study_sentiment
  python -m src.analysis.event_study_sentiment --threshold 0.2 --horizons 1 3 5 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from src.analysis._panel_loader import load_panel

_REQUIRED = ["ticker", "date", "log_return", "sentiment_score", "news_count"]
_NUMERIC = ["log_return", "sentiment_score", "news_count"]

_DEFAULT_PANEL = os.path.join(
    os.getenv("MARKET_VALIDATION_OUTPUT_DIR", "data/processed/validation"),
    "daily_panel.csv",
)


def add_market_return(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    daily_sum = panel.groupby("date")["log_return"].sum()
    daily_count = panel.groupby("date")["log_return"].count()
    panel["market_log_return"] = panel.apply(
        lambda r: (daily_sum[r["date"]] - r["log_return"]) / (daily_count[r["date"]] - 1)
        if daily_count[r["date"]] > 1 else np.nan,
        axis=1,
    )
    return panel


def sentiment_group(score: float, threshold: float) -> str:
    if pd.isna(score):
        return "NO_NEWS"
    if score > threshold:
        return "POS"
    if score < -threshold:
        return "NEG"
    return "NEU"


def add_future_returns(group: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    group = group.sort_values("date").reset_index(drop=True).copy()
    for h in horizons:
        future = sum(group["log_return"].shift(-i) for i in range(1, h + 1))
        future_market = sum(group["market_log_return"].shift(-i) for i in range(1, h + 1))
        group[f"future_return_{h}"] = future
        group[f"future_abret_{h}"] = future - future_market
    return group


def summarize(events: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    use_fama_macbeth = "ticker" not in group_cols
    for keys, group in events.dropna(subset=[value_col]).groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group[value_col].to_numpy(dtype=float)
        if use_fama_macbeth:
            ticker_means = group.groupby("ticker")[value_col].mean().dropna()
            n_tickers = len(ticker_means)
            t_stat, p_value = (ttest_1samp(ticker_means.values, popmean=0.0) if n_tickers >= 2
                               else (np.nan, np.nan))
        else:
            n_tickers = 1
            t_stat, p_value = (ttest_1samp(values, popmean=0.0, nan_policy="omit") if len(values) >= 2
                               else (np.nan, np.nan))
        rows.append({
            **dict(zip(group_cols, keys)),
            "metric": value_col, "n": len(values), "n_tickers": n_tickers,
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "t_stat_vs_0": float(t_stat) if not pd.isna(t_stat) else np.nan,
            "p_value_vs_0": float(p_value) if not pd.isna(p_value) else np.nan,
        })
    return pd.DataFrame(rows)


def run(
    panel_path: str,
    output_events: str,
    output_summary: str,
    horizons: list[int],
    threshold: float,
    include_no_news: bool,
) -> None:
    panel = add_market_return(load_panel(panel_path, required_cols=_REQUIRED, numeric_cols=_NUMERIC))
    panel = pd.concat(
        [add_future_returns(group, horizons) for _, group in panel.groupby("ticker")],
        ignore_index=True,
    )
    panel["sentiment_group"] = panel["sentiment_score"].map(lambda s: sentiment_group(s, threshold))

    events = panel[panel["news_count"].fillna(0) > 0].copy()
    if include_no_news:
        events = panel.copy()

    out_events = Path(output_events)
    out_events.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_events, index=False, encoding="utf-8-sig")

    summaries = []
    for h in horizons:
        for metric in [f"future_return_{h}", f"future_abret_{h}"]:
            summaries.append(summarize(events, metric, ["sentiment_group"]))
            summaries.append(summarize(events, metric, ["ticker", "sentiment_group"]))
    summary = pd.concat(summaries, ignore_index=True)
    summary = summary.sort_values(["metric", "sentiment_group"])

    Path(output_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_summary, index=False, encoding="utf-8-sig")

    print(f"Saved event rows to {output_events}")
    print(f"Saved summary to {output_summary}")
    print("\nOverall event-study summary:")
    overall = summary[~summary.get("ticker", pd.Series(dtype=str)).notna()] if "ticker" in summary.columns else summary
    cols = ["metric", "sentiment_group", "n", "n_tickers", "mean", "median", "t_stat_vs_0", "p_value_vs_0"]
    available_cols = [c for c in cols if c in summary.columns]
    print(summary[summary.get("ticker", pd.Series(np.nan, index=summary.index)).isna()][available_cols].to_string(index=False))
    print("\n(t_stat/p_value: Fama-MacBeth t-test trên mean per ticker, n_tickers = số ticker)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event study by sentiment group")
    parser.add_argument("--panel", default=_DEFAULT_PANEL)
    parser.add_argument("--output-events", default="data/results/event_study_rows.csv")
    parser.add_argument("--output-summary", default="data/results/event_study_summary.csv")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--include-no-news", action="store_true")
    args = parser.parse_args()

    run(
        panel_path=args.panel,
        output_events=args.output_events,
        output_summary=args.output_summary,
        horizons=args.horizons,
        threshold=args.threshold,
        include_no_news=args.include_no_news,
    )
