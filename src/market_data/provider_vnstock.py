"""vnstock market data provider."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class VnstockProvider:
    """Wraps the vnstock library to fetch daily OHLCV price data."""

    def fetch_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        """Fetch daily OHLCV for one ticker.

        Returns a DataFrame with columns: date, ticker, open, high, low, close, volume
        or None if no data is returned. Raises RuntimeError on provider failure.
        """
        try:
            from vnstock import Vnstock
            stock = Vnstock().stock(symbol=ticker, source="VCI")
            df = stock.quote.history(start=start_date, end=end_date, interval="1D")
        except Exception as exc:
            raise RuntimeError(
                f"vnstock fetch failed ticker={ticker}: {exc}"
            ) from exc

        if df is None or df.empty:
            return None

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Handle 'time' column name used in some vnstock versions
        if "time" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"time": "date"})

        # Handle date as index
        if df.index.name in ("time", "date") and "date" not in df.columns:
            df = df.reset_index()
            df.columns = ["date"] + list(df.columns[1:])

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(
                f"vnstock response missing columns={missing} for ticker={ticker}. "
                f"Got: {list(df.columns)}"
            )

        if "date" not in df.columns:
            raise RuntimeError(
                f"vnstock response has no date column for ticker={ticker}. "
                f"Got: {list(df.columns)}"
            )

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["ticker"] = ticker.upper()

        return df[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()
