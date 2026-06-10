"""
Compare binary sentiment direction with binary stock direction.

Definitions:
  sentiment_up = 1 if sentiment_score > threshold, else 0 (or NaN if no news)
  target_up    = 1 if log_return > 0, else 0

Lag convention:
  lag=1 compares sentiment_up from yesterday with target_up today.

Usage:
  python -m src.analysis.sentiment_target_confusion
  python -m src.analysis.sentiment_target_confusion --lags 0 1 2 3 4 5
  python -m src.analysis.sentiment_target_confusion --news-days-only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.analysis._panel_loader import load_panel

_REQUIRED = ["ticker", "date", "sentiment_score", "log_return"]
_NUMERIC = ["sentiment_score", "log_return", "target_up"]

_DEFAULT_PANEL = os.path.join(
    os.getenv("MARKET_VALIDATION_OUTPUT_DIR", "data/processed/validation"),
    "daily_panel.csv",
)


def evaluate_binary(y_true: pd.Series, y_pred: pd.Series) -> dict:
    valid = pd.concat([y_true, y_pred], axis=1).dropna()
    if valid.empty or valid.iloc[:, 0].nunique() < 2:
        return {
            "n": len(valid),
            "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan,
            "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan,
            "target_up_rate": np.nan, "sentiment_up_rate": np.nan,
        }
    y = valid.iloc[:, 0].astype(int)
    pred = valid.iloc[:, 1].astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(valid),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "target_up_rate": float(y.mean()),
        "sentiment_up_rate": float(pred.mean()),
    }


def make_sentiment_up(score: pd.Series, threshold: float, count_missing_as_down: bool) -> pd.Series:
    if count_missing_as_down:
        return (score > threshold).astype(float)
    return pd.Series(
        np.where(score.isna(), np.nan, score > threshold),
        index=score.index,
        dtype="float",
    )


def run(
    panel_path: str,
    output_path: str,
    lags: list[int],
    threshold: float,
    min_n: int,
    count_missing_as_down: bool,
    news_days_only: bool,
) -> None:
    panel = load_panel(
        panel_path, required_cols=_REQUIRED, numeric_cols=_NUMERIC, news_days_only=news_days_only,
    )
    if "target_up" not in panel.columns:
        panel["target_up"] = np.where(
            panel["log_return"].notna(), (panel["log_return"] > 0).astype(float), np.nan,
        )

    rows = []
    for ticker, group in panel.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True).copy()
        group["sentiment_up"] = make_sentiment_up(group["sentiment_score"], threshold, count_missing_as_down)
        for lag in lags:
            pred = group["sentiment_up"].shift(lag)
            metrics = evaluate_binary(group["target_up"], pred)
            if metrics["n"] >= min_n:
                rows.append({"ticker": ticker, "lag": lag, "threshold": threshold, **metrics})

    all_group = panel.sort_values(["ticker", "date"]).copy()
    all_group["sentiment_up"] = make_sentiment_up(all_group["sentiment_score"], threshold, count_missing_as_down)
    for lag in lags:
        parts = []
        for _, group in all_group.groupby("ticker"):
            group = group.sort_values("date").reset_index(drop=True).copy()
            parts.append(
                pd.DataFrame({
                    "target_up": group["target_up"],
                    "sentiment_up_lagged": group["sentiment_up"].shift(lag),
                })
            )
        combined = pd.concat(parts, ignore_index=True)
        metrics = evaluate_binary(combined["target_up"], combined["sentiment_up_lagged"])
        rows.append({"ticker": "ALL", "lag": lag, "threshold": threshold, **metrics})

    result = pd.DataFrame(rows).sort_values(["ticker", "lag"])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Saved confusion metrics to {output_path}")
    cols = ["ticker", "lag", "n", "tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "f1",
            "target_up_rate", "sentiment_up_rate"]
    print("\nALL rows:")
    print(result[result["ticker"] == "ALL"][cols].to_string(index=False))
    print("\nTop ticker/lag rows by F1:")
    print(result[result["ticker"] != "ALL"].sort_values("f1", ascending=False).head(20)[cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary sentiment vs target_up confusion matrix")
    parser.add_argument("--panel", default=_DEFAULT_PANEL)
    parser.add_argument("--output", default="data/results/sentiment_target_confusion.csv")
    parser.add_argument("--lags", type=int, nargs="+", default=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    parser.add_argument(
        "--threshold", "--sentiment-threshold", dest="threshold", type=float, default=0.0,
        help="Classify sentiment_up=1 when sentiment_score > threshold",
    )
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument(
        "--count-missing-as-down", action="store_true",
        help="Treat missing sentiment/no-news days as sentiment_up=0 instead of dropping them",
    )
    parser.add_argument("--news-days-only", action="store_true")
    args = parser.parse_args()

    run(
        panel_path=args.panel,
        output_path=args.output,
        lags=args.lags,
        threshold=args.threshold,
        min_n=args.min_n,
        count_missing_as_down=args.count_missing_as_down,
        news_days_only=args.news_days_only,
    )
