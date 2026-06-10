"""Market feature engineering from OHLCV price data."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

import numpy as np

_OUT_COLS = [
    "date", "ticker", "close", "volume",
    "return_1d", "return_3d", "return_5d",
    "forward_return_1d", "forward_return_3d", "forward_return_5d",
    "volume_change_5d", "volatility_5d",
    "log_return", "volume_growth", "clv",
]


def build_market_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features from daily OHLCV data per ticker.

    Features:
    - return_Nd: N-day trailing pct change in close
    - forward_return_Nd: N-day forward pct change (shifted back by N days)
    - volume_change_5d: volume / 5-day moving-average volume
    - volatility_5d: 5-day rolling std of daily log returns
    - log_return: daily natural log return ln(close / prev_close)
    - volume_growth: daily log volume change ln(volume / prev_volume)
    - clv: Close Location Value = ((close-low) - (high-close)) / (high-low), range [-1, 1]
    """
    if price_df.empty:
        logger.warning("build_market_features called with empty DataFrame")
        return pd.DataFrame(columns=_OUT_COLS)

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    groups = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.copy().reset_index(drop=True)

        close = g["close"].astype(float)
        volume = g["volume"].astype(float)
        high = g["high"].astype(float) if "high" in g.columns else close
        low = g["low"].astype(float) if "low" in g.columns else close

        g["return_1d"] = close.pct_change(1)
        g["return_3d"] = close.pct_change(3)
        g["return_5d"] = close.pct_change(5)

        # Forward returns: value at row i = return earned from close[i] to close[i+N]
        g["forward_return_1d"] = close.pct_change(1).shift(-1)
        g["forward_return_3d"] = close.pct_change(3).shift(-3)
        g["forward_return_5d"] = close.pct_change(5).shift(-5)

        vol_ma5 = volume.rolling(5, min_periods=1).mean().replace(0, float("nan"))
        g["volume_change_5d"] = volume / vol_ma5

        # log_return: ln(close / prev_close)
        prev_close = close.shift(1)
        g["log_return"] = np.where(
            (close > 0) & (prev_close > 0),
            np.log(close / prev_close),
            np.nan,
        )

        g["volatility_5d"] = pd.Series(g["log_return"]).rolling(5, min_periods=2).std()

        # volume_growth: ln(volume / prev_volume)
        prev_vol = volume.shift(1)
        g["volume_growth"] = np.where(
            (volume > 0) & (prev_vol > 0),
            np.log(volume / prev_vol),
            np.nan,
        )

        # clv: Close Location Value
        price_range = high - low
        g["clv"] = np.where(
            price_range > 0,
            ((close - low) - (high - close)) / price_range,
            0.0,
        )

        groups.append(g)

    out = pd.concat(groups, ignore_index=True)
    out["date"] = out["date"].dt.date

    return out[_OUT_COLS].copy()
