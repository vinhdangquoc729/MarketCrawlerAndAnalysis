"""
Granger-style test: does past sentiment add predictive power for future returns?

For each ticker and each lookback window, compare:
  Restricted:   R_t ~ past R + past volume_growth
  Unrestricted: R_t ~ past R + past volume_growth + past sentiment_score

Reports an F-statistic and p-value for the added sentiment lags.

Lookback windows are trading-day counts:
  10  ~= 2 weeks
  21  ~= 1 month
  63  ~= 3 months

Usage:
  python -m src.analysis.granger_sentiment
  python -m src.analysis.granger_sentiment --panel data/processed/validation/daily_panel.csv
  python -m src.analysis.granger_sentiment --news-days-only --output data/results/granger_news_days.csv
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from src.analysis._panel_loader import load_panel

DEFAULT_WINDOWS = [10, 21, 63]
TARGETS = ["log_return"]
CONTROL_COLS = ["log_return", "volume_growth"]
SENTIMENT_COL = "sentiment_score"

_REQUIRED = ["ticker", "date", "log_return", "volume_growth", SENTIMENT_COL]
_NUMERIC = ["log_return", "volume_growth", SENTIMENT_COL]

_DEFAULT_PANEL = os.path.join(
    os.getenv("MARKET_VALIDATION_OUTPUT_DIR", "data/processed/validation"),
    "daily_panel.csv",
)


def add_lags(group: pd.DataFrame, cols: list[str], max_lag: int) -> pd.DataFrame:
    lagged_cols = {}
    for col in cols:
        for lag in range(1, max_lag + 1):
            lagged_cols[f"{col}_lag{lag}"] = group[col].shift(lag)
    return pd.concat([group.copy(), pd.DataFrame(lagged_cols, index=group.index)], axis=1)


def fit_ols_rss(y: np.ndarray, x: np.ndarray) -> tuple[float, int]:
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    rss = float(np.sum(residuals**2))
    rank = int(np.linalg.matrix_rank(x))
    return rss, rank


def granger_for_group(
    group: pd.DataFrame,
    ticker: str,
    target: str,
    window: int,
    min_obs: int,
) -> dict:
    lagged = add_lags(group, CONTROL_COLS + [SENTIMENT_COL], window)

    restricted_cols = [
        f"{col}_lag{lag}"
        for col in CONTROL_COLS
        for lag in range(1, window + 1)
    ]
    sentiment_cols = [f"{SENTIMENT_COL}_lag{lag}" for lag in range(1, window + 1)]
    unrestricted_cols = restricted_cols + sentiment_cols

    data = lagged[[target] + unrestricted_cols].dropna()
    n = len(data)
    if n < max(min_obs, len(unrestricted_cols) + 5):
        return {
            "ticker": ticker, "target": target, "window": window, "n": n,
            "f_stat": np.nan, "p_value": np.nan,
            "rss_restricted": np.nan, "rss_unrestricted": np.nan,
            "df_num": window, "df_den": np.nan,
            "status": "too_few_observations",
        }

    y = data[target].to_numpy(dtype=float)
    x_r = data[restricted_cols].to_numpy(dtype=float)
    x_u = data[unrestricted_cols].to_numpy(dtype=float)

    rss_r, rank_r = fit_ols_rss(y, x_r)
    rss_u, rank_u = fit_ols_rss(y, x_u)
    df_num = rank_u - rank_r
    df_den = n - rank_u

    if df_num <= 0 or df_den <= 0 or rss_u <= 0:
        f_stat, p_value, status = np.nan, np.nan, "invalid_degrees_of_freedom"
    else:
        f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
        if f_stat < 0 and abs(f_stat) < 1e-12:
            f_stat = 0.0
        p_value = float(f_dist.sf(f_stat, df_num, df_den)) if f_stat >= 0 else np.nan
        status = "ok" if not math.isnan(p_value) else "invalid_f_stat"

    return {
        "ticker": ticker, "target": target, "window": window, "n": n,
        "f_stat": f_stat, "p_value": p_value,
        "rss_restricted": rss_r, "rss_unrestricted": rss_u,
        "df_num": df_num, "df_den": df_den, "status": status,
    }


def run(panel_path: str, output_path: str, windows: list[int], min_obs: int, news_days_only: bool) -> None:
    panel = load_panel(panel_path, required_cols=_REQUIRED, numeric_cols=_NUMERIC, news_days_only=news_days_only)
    rows = []
    for ticker, group in panel.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        for target in TARGETS:
            for window in windows:
                rows.append(granger_for_group(group, ticker, target, window, min_obs))

    result = pd.DataFrame(rows).sort_values(["p_value", "ticker", "window"], na_position="last")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Saved Granger results to {output_path}")
    print("\nSmallest p-values:")
    cols = ["ticker", "target", "window", "n", "f_stat", "p_value", "status"]
    print(result[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Granger-style sentiment -> return test")
    parser.add_argument("--panel", default=_DEFAULT_PANEL)
    parser.add_argument("--output", default="data/results/granger_sentiment_return.csv")
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument("--min-obs", type=int, default=80)
    parser.add_argument("--news-days-only", action="store_true")
    args = parser.parse_args()

    run(
        panel_path=args.panel,
        output_path=args.output,
        windows=args.windows,
        min_obs=args.min_obs,
        news_days_only=args.news_days_only,
    )
