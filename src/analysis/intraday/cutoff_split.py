"""Compare predictive power of sentiment by publication timing.

Hypothesis: post-close articles (published after 14:45) should predict
next-session returns better than intraday articles (already priced in).

Outputs:
  data/results/intraday/cutoff_split_accuracy.csv   - binary direction accuracy by bucket
  data/results/intraday/cutoff_split_returns.csv    - mean forward returns by bucket x sentiment
  data/results/intraday/cutoff_split_correlation.csv

Usage:
  python -m src.analysis.intraday.cutoff_split
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.analysis.intraday._loader import build_article_bar_dataset
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
OUT_DIR = Path("data/results/intraday")
FORWARD_BARS = 8

BUCKET_ORDER = ["pre_open", "intraday", "post_close"]


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    valid = pd.concat([y_true, y_pred], axis=1).dropna()
    if len(valid) < 10 or valid.iloc[:, 0].nunique() < 2:
        return {"n": len(valid), "accuracy": np.nan, "precision": np.nan,
                "recall": np.nan, "f1": np.nan, "base_rate": np.nan}
    y = valid.iloc[:, 0].astype(int)
    p = valid.iloc[:, 1].astype(int)
    return {
        "n": len(valid),
        "accuracy": accuracy_score(y, p),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "base_rate": float(y.mean()),
    }


def run(min_date: str = "2023-10-01") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_article_bar_dataset(min_date=min_date, forward_bars=FORWARD_BARS)
    if df.empty:
        print("No data.")
        return

    df = df[df["timing"].isin(BUCKET_ORDER)].copy()
    df["sentiment_up"] = (df["sentiment_score"] > 0).astype(float)

    # Target: is next bar (fwd_bar_1) positive?
    df["next_bar_up"] = (df["fwd_bar_1"] > 0).astype(float)
    # For post-close: fwd_bar_1 is the first bar of next session (pre-open gap captured)
    # Cumulative 4-bar forward return direction
    df["fwd_4bar_up"] = (df[[f"fwd_bar_{i}" for i in range(1, 5)]].sum(axis=1) > 0).astype(float)

    # 1. Binary accuracy by timing bucket
    acc_rows = []
    for timing in BUCKET_ORDER:
        sub = df[df["timing"] == timing]
        for target_col, target_label in [("next_bar_up", "next_bar"), ("fwd_4bar_up", "fwd_4bar")]:
            m = _binary_metrics(sub[target_col], sub["sentiment_up"])
            acc_rows.append({"timing": timing, "target": target_label, **m})

    # All buckets combined
    for target_col, target_label in [("next_bar_up", "next_bar"), ("fwd_4bar_up", "fwd_4bar")]:
        m = _binary_metrics(df[target_col], df["sentiment_up"])
        acc_rows.append({"timing": "ALL", "target": target_label, **m})

    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(OUT_DIR / "cutoff_split_accuracy.csv", index=False)

    print("=== Binary accuracy: sentiment_up predicts next bar direction ===")
    print(acc_df.to_string(index=False))

    # 2. Mean forward returns by timing bucket x sentiment label
    ret_rows = []
    fwd_cols = [f"fwd_bar_{i}" for i in range(1, FORWARD_BARS + 1)]
    for timing in BUCKET_ORDER + ["ALL"]:
        sub = df if timing == "ALL" else df[df["timing"] == timing]
        for label in ["positive", "negative", "neutral"]:
            grp = sub[sub["final_sentiment"] == label]
            for col in ["bar_log_return"] + fwd_cols:
                clean = grp[col].dropna()
                if len(clean) < 5:
                    continue
                t, p = stats.ttest_1samp(clean, 0)
                ret_rows.append({
                    "timing": timing,
                    "sentiment": label,
                    "bar": col,
                    "n": len(clean),
                    "mean_return": clean.mean(),
                    "t_stat": t,
                    "p_value": p,
                })

    ret_df = pd.DataFrame(ret_rows)
    ret_df.to_csv(OUT_DIR / "cutoff_split_returns.csv", index=False)

    print("\n=== Mean fwd_bar_1 return by timing x sentiment ===")
    pivot = ret_df[ret_df["bar"] == "fwd_bar_1"].pivot_table(
        index="timing", columns="sentiment", values="mean_return"
    ).round(6)
    print(pivot.to_string())

    # 3. Pearson correlation: sentiment_score vs fwd_bar_1 and fwd_4bars, by bucket
    corr_rows = []
    for timing in BUCKET_ORDER + ["ALL"]:
        sub = df if timing == "ALL" else df[df["timing"] == timing]
        for col, label in [("fwd_bar_1", "fwd_bar_1"), ("fwd_4bar_up", "fwd_4bar_up"),
                           ("bar_log_return", "bar_log_return")]:
            valid = sub[["sentiment_score", col]].dropna()
            if len(valid) < 20:
                continue
            r, p = stats.pearsonr(valid["sentiment_score"], valid[col])
            corr_rows.append({
                "timing": timing, "target": label,
                "n": len(valid), "correlation": r, "p_value": p,
            })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_DIR / "cutoff_split_correlation.csv", index=False)

    print("\n=== Sentiment-return correlation by timing bucket ===")
    print(corr_df.pivot_table(index="timing", columns="target", values="correlation").round(4).to_string())
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    setup_logging()
    run()
