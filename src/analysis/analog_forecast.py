"""
Analog forecasting: find historical trading-day patterns similar to today.

Feature engineering (avoids temporal proximity bias):
  - log_return, volume_growth: within-window z-score — captures *dynamics*
    (was it a week that went down-down-flat-up-up?) not absolute level.
    Raw values bias toward recent windows because market regimes cluster.
  - sentiment_score: kept as absolute value — +0.3 is meaningfully different
    from -0.3 regardless of window context.

After within-window normalisation, inter-window distances are computed
on the combined (normalised return/volume, raw sentiment) vector.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

FEATURE_COLS = ["log_return", "volume_growth", "sentiment_score"]
# Normalise these per-window (removes level bias); sentiment stays absolute
_WITHIN_WINDOW_COLS_IDX = [0, 1]  # log_return, volume_growth
START_YEAR = 2024


def _fill_features(grp: pd.DataFrame) -> pd.DataFrame:
    grp = grp.copy()
    for col in FEATURE_COLS:
        grp[col] = pd.to_numeric(grp[col], errors="coerce").fillna(0.0)
    return grp


def _encode_window(mat: np.ndarray) -> np.ndarray:
    """
    Encode a (window, n_features) matrix into a flat comparison vector.

    log_return and volume_growth are within-window z-scored so similarity
    reflects *dynamics* (shape of the pattern) rather than absolute level.
    sentiment_score is kept as-is — absolute direction matters.
    """
    out = mat.astype(float).copy()
    for idx in _WITHIN_WINDOW_COLS_IDX:
        vals = out[:, idx]
        mu, sigma = vals.mean(), vals.std()
        out[:, idx] = (vals - mu) / sigma if sigma > 1e-8 else np.zeros_like(vals)
    return out.flatten()


def run_analog_forecast(
    panel: pd.DataFrame,
    as_of_date=None,
    window: int = 5,
    top_k: int = 15,
    horizon: int = 5,
    start_year: int = START_YEAR,
) -> pd.DataFrame:
    """
    Find historical analog windows for each ticker and compute forward-return stats.

    Parameters
    ----------
    panel       : DataFrame with ticker, date, log_return, volume_growth,
                  sentiment_score, and optionally news_count.
    as_of_date  : Reference date. Default = max date in panel.
    window      : Lookback window (trading days) for feature vector.
    top_k       : Number of nearest-neighbour analogs to retrieve.
    horizon     : Forward horizon (trading days) to measure outcomes.
    start_year  : Only use data from this year onwards.

    Feature encoding per window:
      log_return, volume_growth → within-window z-score (captures dynamics,
        not absolute level; avoids temporal proximity bias).
      sentiment_score → globally z-scored across candidates (preserves
        direction: positive vs negative sentiment still distinguishable).

    Returns
    -------
    DataFrame, one row per ticker, sorted by confidence descending.
    Includes a `fwd_paths` column (list of cumulative-return paths)
    for spaghetti chart rendering.
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"].dt.year >= start_year]
    panel = panel.sort_values(["ticker", "date"])

    if as_of_date is None:
        as_of_date = panel["date"].max()
    as_of_date = pd.Timestamp(as_of_date)

    results = []

    for ticker, grp in panel.groupby("ticker"):
        grp = _fill_features(grp.sort_values("date").reset_index(drop=True))
        n = len(grp)

        # Last row on or before as_of_date
        mask = grp["date"] <= as_of_date
        if mask.sum() < window:
            continue
        cur_end = int(mask.sum()) - 1
        cur_start = cur_end - window + 1

        current_mat = grp.iloc[cur_start:cur_end + 1][FEATURE_COLS].values
        if current_mat.shape[0] != window:
            continue
        # Within-window encode: return/volume z-scored per window, sentiment absolute
        current_flat = _encode_window(current_mat)

        # Historical candidates: end at most (cur_end - horizon) so forward
        # data exists and does not bleed into the "future" we are predicting
        hist_end = cur_end - horizon
        if hist_end < window - 1:
            continue

        hist_values = grp.iloc[:hist_end + 1][FEATURE_COLS].values
        n_cands = hist_end + 1 - window + 1
        if n_cands < max(top_k, 5):
            continue

        # Encode each candidate window the same way
        cand_matrix = np.stack([
            _encode_window(hist_values[i:i + window])
            for i in range(n_cands)
        ])  # (n_cands, window * 3)

        # Global z-score on sentiment column group only (columns 2W..3W-1)
        # so that sentiment scale is comparable across different tickers
        sent_start = 2 * window
        sent_cols = cand_matrix[:, sent_start:]
        s_mean = sent_cols.mean(axis=0)
        s_std = sent_cols.std(axis=0)
        s_std[s_std < 1e-8] = 1.0
        cand_matrix[:, sent_start:] = (sent_cols - s_mean) / s_std
        current_flat[sent_start:] = (current_flat[sent_start:] - s_mean) / s_std

        dists = np.linalg.norm(cand_matrix - current_flat, axis=1)
        k = min(top_k, n_cands)
        top_idx = np.argsort(dists)[:k]

        # Collect forward outcomes for each analog
        forward_returns: list[float] = []
        fwd_paths: list[list[float]] = []
        analog_details: list[dict] = []
        for rank, ci in enumerate(top_idx):
            analog_end = ci + window - 1          # row in grp where analog ends
            fwd_start = analog_end + 1
            fwd_end = analog_end + horizon         # inclusive
            if fwd_end >= n:
                continue
            daily = grp.iloc[fwd_start:fwd_end + 1]["log_return"].values
            if len(daily) == horizon:
                fwd_total = float(daily.sum())
                forward_returns.append(fwd_total)
                fwd_paths.append(daily.cumsum().tolist())
                # Raw window values for this analog (shape: window × 3)
                raw_window = grp.iloc[ci:ci + window][FEATURE_COLS].values
                detail: dict = {
                    "rank": rank + 1,
                    "analog_date": grp.iloc[analog_end]["date"],
                    "distance": float(dists[ci]),
                    "fwd_return": fwd_total,
                }
                for day_i in range(window):
                    detail[f"ret_{day_i - window + 1}"] = float(raw_window[day_i, 0])
                    detail[f"vol_{day_i - window + 1}"] = float(raw_window[day_i, 1])
                    detail[f"sent_{day_i - window + 1}"] = float(raw_window[day_i, 2])
                analog_details.append(detail)

        if len(forward_returns) < 5:
            continue

        fwd = np.array(forward_returns)
        win_rate = float(np.mean(fwd > 0))
        avg_fwd = float(np.mean(fwd))
        median_fwd = float(np.median(fwd))
        n_analogs = len(fwd)

        today_sentiment = float(grp.iloc[cur_end]["sentiment_score"])
        today_news = 0
        if "news_count" in grp.columns:
            raw = grp.iloc[cur_end].get("news_count", 0)
            today_news = int(raw) if raw and not (isinstance(raw, float) and np.isnan(raw)) else 0
        recent_return = float(grp.iloc[cur_start:cur_end + 1]["log_return"].sum())

        if win_rate >= 0.6 and avg_fwd > 0:
            signal = "BULLISH"
        elif win_rate <= 0.4 and avg_fwd < 0:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        # Confidence: deviation of win_rate from 50 %, scaled by sample completeness
        confidence = abs(win_rate - 0.5) * 2.0 * min(1.0, n_analogs / top_k)

        results.append({
            "ticker": ticker,
            "signal": signal,
            "today_sentiment": today_sentiment,
            "today_news_count": today_news,
            "recent_5d_return": recent_return,
            "win_rate": win_rate,
            "avg_fwd_5d": avg_fwd,
            "median_fwd_5d": median_fwd,
            "n_analogs": n_analogs,
            "confidence": confidence,
            "fwd_paths": fwd_paths,
            "analog_details": analog_details,
            "current_window": current_mat.tolist(),  # (window × 3) raw values for today
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("confidence", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from src.analysis._panel_loader import load_panel

    _REQUIRED = ["ticker", "date", "log_return", "volume_growth", "sentiment_score"]
    _NUMERIC = ["log_return", "volume_growth", "sentiment_score"]

    parser = argparse.ArgumentParser(description="Analog forecast signal per ticker")
    parser.add_argument("--panel", default=os.path.join(
        os.getenv("MARKET_VALIDATION_OUTPUT_DIR", "data/processed/validation"),
        "daily_panel.csv",
    ))
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--as-of-date", default=None)
    args = parser.parse_args()

    panel = load_panel(args.panel, required_cols=_REQUIRED, numeric_cols=_NUMERIC)
    result = run_analog_forecast(
        panel,
        as_of_date=args.as_of_date,
        window=args.window,
        top_k=args.top_k,
        horizon=args.horizon,
    )
    display_cols = ["ticker", "signal", "today_sentiment", "today_news_count",
                    "recent_5d_return", "win_rate", "avg_fwd_5d", "n_analogs", "confidence"]
    print(result[display_cols].to_string(index=False))
