"""
Visualize news sentiment and stock price changes.

Input should be a daily panel produced by ``src.jobs.export_daily_panel``.

Plots:
  1. Per-ticker time series: close price, sentiment_score, log_return
  2. Scatter: lagged sentiment_score vs log_return
  3. Scatter: lagged sentiment_score vs target_up with jitter

Lag convention:
  lag=1 means yesterday's sentiment is plotted against today's price move.

Usage:
  python -m src.analysis.visualize_sentiment_price --ticker FPT --lag 1
  python -m src.analysis.visualize_sentiment_price --tickers FPT VCB HPG --lag 1
  python -m src.analysis.visualize_sentiment_price --news-days-only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis._panel_loader import load_panel

_REQUIRED = ["ticker", "date", "close", "log_return", "sentiment_score"]
_NUMERIC = ["close", "log_return", "sentiment_score", "target_up"]

_DEFAULT_PANEL = os.path.join(
    os.getenv("MARKET_VALIDATION_OUTPUT_DIR", "data/processed/validation"),
    "daily_panel.csv",
)


def prepare_lagged(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True).copy()
        group["sentiment_lagged"] = group["sentiment_score"].shift(lag)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def plot_ticker_timeseries(df: pd.DataFrame, ticker: str, output_dir: Path) -> None:
    data = df[df["ticker"] == ticker].sort_values("date").copy()
    if data.empty:
        print(f"No rows for {ticker}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(data["date"], data["close"], color="#1f77b4", linewidth=1.6)
    axes[0].set_ylabel("Close")
    axes[0].set_title(f"{ticker}: price, sentiment, and return")

    axes[1].bar(
        data["date"], data["sentiment_score"],
        color=np.where(data["sentiment_score"].fillna(0) >= 0, "#2ca02c", "#d62728"),
        width=1.0, alpha=0.75,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Sentiment")

    axes[2].bar(
        data["date"], data["log_return"],
        color=np.where(data["log_return"].fillna(0) >= 0, "#2ca02c", "#d62728"),
        width=1.0, alpha=0.75,
    )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Log return")
    axes[2].set_xlabel("Date")

    fig.tight_layout()
    path = output_dir / f"{ticker}_sentiment_price_timeseries.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved {path}")


def plot_scatter_return(df: pd.DataFrame, lag: int, output_dir: Path, tickers: list[str]) -> None:
    data = df[df["ticker"].isin(tickers)].dropna(subset=["sentiment_lagged", "log_return"]).copy()
    if data.empty:
        print("No rows for scatter return plot")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    for ticker, group in data.groupby("ticker"):
        ax.scatter(group["sentiment_lagged"], group["log_return"], s=22, alpha=0.55, label=ticker)

    x = data["sentiment_lagged"].to_numpy()
    y = data["log_return"].to_numpy()
    if len(data) >= 3 and np.nanstd(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.4, label="OLS line")
        corr = np.corrcoef(x, y)[0, 1]
    else:
        corr = np.nan

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title(f"Lagged sentiment vs log return (lag={lag}, r={corr:.3f})")
    ax.set_xlabel(f"Sentiment score at t-{lag}" if lag >= 0 else f"Sentiment score at t+{abs(lag)}")
    ax.set_ylabel("Log return at t")
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    path = output_dir / f"sentiment_vs_log_return_lag{lag}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved {path}")


def plot_scatter_target_up(df: pd.DataFrame, lag: int, output_dir: Path, tickers: list[str]) -> None:
    data = df[df["ticker"].isin(tickers)].dropna(subset=["sentiment_lagged", "target_up"]).copy()
    if data.empty:
        print("No rows for target_up plot")
        return

    rng = np.random.default_rng(42)
    y_jitter = data["target_up"].to_numpy() + rng.normal(0, 0.035, len(data))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = np.where(data["target_up"].to_numpy() > 0, "#2ca02c", "#d62728")
    ax.scatter(data["sentiment_lagged"], y_jitter, s=20, alpha=0.5, c=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks([0, 1], ["Down/flat", "Up"])
    ax.set_xlabel(f"Sentiment score at t-{lag}" if lag >= 0 else f"Sentiment score at t+{abs(lag)}")
    ax.set_title(f"Lagged sentiment vs target_up (lag={lag})")
    fig.tight_layout()
    path = output_dir / f"sentiment_vs_target_up_lag{lag}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved {path}")


def run(
    panel_path: str,
    tickers: list[str] | None,
    lag: int,
    output_dir: str,
    news_days_only: bool,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    panel = load_panel(panel_path, required_cols=_REQUIRED, numeric_cols=_NUMERIC, news_days_only=news_days_only)
    if "target_up" not in panel.columns:
        panel["target_up"] = np.where(
            panel["log_return"].notna(), (panel["log_return"] > 0).astype(float), np.nan
        )

    if not tickers:
        tickers = sorted(panel["ticker"].dropna().unique().tolist())[:5]
    tickers = [t.upper() for t in tickers]

    lagged = prepare_lagged(panel, lag)
    for ticker in tickers:
        plot_ticker_timeseries(panel, ticker, output)
    plot_scatter_return(lagged, lag, output, tickers)
    plot_scatter_target_up(lagged, lag, output, tickers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize sentiment and price changes")
    parser.add_argument("--panel", default=_DEFAULT_PANEL)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--ticker", default=None, help="Single ticker shortcut")
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--output-dir", default="data/results/plots")
    parser.add_argument("--news-days-only", action="store_true")
    args = parser.parse_args()

    selected = args.tickers
    if args.ticker:
        selected = [args.ticker]

    run(
        panel_path=args.panel,
        tickers=selected,
        lag=args.lag,
        output_dir=args.output_dir,
        news_days_only=args.news_days_only,
    )
