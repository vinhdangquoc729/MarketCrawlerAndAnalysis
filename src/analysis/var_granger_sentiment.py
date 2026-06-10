"""
VAR-based Granger causality test.

Fits a VAR per ticker using:
  log_return, volume_growth, sentiment_score

Then tests whether sentiment_score Granger-causes log_return.

Usage:
  python -m src.analysis.var_granger_sentiment
  python -m src.analysis.var_granger_sentiment --news-days-only --fill-missing-sentiment nan
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis._panel_loader import load_panel

_REQUIRED = ["ticker", "date", "log_return", "volume_growth", "sentiment_score"]
_NUMERIC = ["log_return", "volume_growth", "sentiment_score"]

_DEFAULT_PANEL = os.path.join(
    os.getenv("MARKET_VALIDATION_OUTPUT_DIR", "data/processed/validation"),
    "daily_panel.csv",
)


def run(
    panel_path: str,
    output_path: str,
    lags: list[int],
    fill_missing_sentiment: float | None,
    min_obs: int,
    news_days_only: bool,
) -> None:
    try:
        from statsmodels.tsa.api import VAR
    except ImportError as exc:
        raise SystemExit("Missing dependency: statsmodels. Install with `pip install statsmodels`.") from exc

    panel = load_panel(
        panel_path,
        required_cols=_REQUIRED,
        numeric_cols=_NUMERIC,
        fill_missing_sentiment=fill_missing_sentiment,
        news_days_only=news_days_only,
    )
    rows = []
    cols = ["log_return", "volume_growth", "sentiment_score"]

    for ticker, group in panel.groupby("ticker"):
        data = group.sort_values("date")[cols].replace([np.inf, -np.inf], np.nan).dropna()
        for lag in lags:
            if len(data) < max(min_obs, lag * len(cols) + 10):
                rows.append({
                    "ticker": ticker, "lag": lag, "n": len(data),
                    "test_stat": np.nan, "p_value": np.nan,
                    "status": "too_few_observations",
                })
                continue
            try:
                result = VAR(data).fit(maxlags=lag, ic=None, trend="c")
                test = result.test_causality("log_return", ["sentiment_score"], kind="f")
                rows.append({
                    "ticker": ticker, "lag": lag, "n": len(data),
                    "test_stat": float(test.test_statistic),
                    "p_value": float(test.pvalue),
                    "status": "ok",
                })
            except Exception as exc:
                rows.append({
                    "ticker": ticker, "lag": lag, "n": len(data),
                    "test_stat": np.nan, "p_value": np.nan,
                    "status": f"failed: {exc}",
                })

    out_df = pd.DataFrame(rows).sort_values(["p_value", "ticker", "lag"], na_position="last")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Saved VAR Granger results to {output_path}")
    print("\nSmallest p-values:")
    print(out_df.head(20).to_string(index=False))


def _parse_fill(value: str) -> float | None:
    if value.lower() == "nan":
        return None
    return float(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAR Granger test: sentiment -> return")
    parser.add_argument("--panel", default=_DEFAULT_PANEL)
    parser.add_argument("--output", default="data/results/var_granger_sentiment.csv")
    parser.add_argument("--lags", type=int, nargs="+", default=[10, 21, 63])
    parser.add_argument("--fill-missing-sentiment", default="0")
    parser.add_argument("--min-obs", type=int, default=120)
    parser.add_argument("--news-days-only", action="store_true")
    args = parser.parse_args()

    run(
        panel_path=args.panel,
        output_path=args.output,
        lags=args.lags,
        fill_missing_sentiment=_parse_fill(args.fill_missing_sentiment),
        min_obs=args.min_obs,
        news_days_only=args.news_days_only,
    )
