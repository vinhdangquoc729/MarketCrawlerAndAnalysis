"""Correlation and validation reports for sentiment model outputs."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.storage.db import query_dataframe

load_dotenv()
logger = logging.getLogger(__name__)

RETURN_COLS = [
    "forward_return_1d",
    "forward_return_3d",
    "forward_return_5d",
]


def safe_corr(x: pd.Series, y: pd.Series, min_samples: int = 3) -> float | None:
    """Compute correlation safely.

    Returns None when sample is too small or series has no variance.
    """
    data = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(data) < min_samples:
        return None

    if data["x"].nunique() <= 1 or data["y"].nunique() <= 1:
        return None

    value = data["x"].corr(data["y"])

    if pd.isna(value):
        return None

    return float(value)


def load_sentiment_market_dataset() -> pd.DataFrame:
    """Load ticker-level sentiment-market validation dataset."""
    try:
        return query_dataframe(
            """
            SELECT *
            FROM sentiment_market_forward_dataset
            WHERE sentiment_index IS NOT NULL
            ORDER BY date ASC, ticker ASC
            """
        )
    except Exception as exc:
        logger.warning("Could not load sentiment_market_forward_dataset: %s", exc)
        return pd.DataFrame()


def load_aspect_market_dataset() -> pd.DataFrame:
    """Load aspect-level sentiment-market validation dataset."""
    try:
        return query_dataframe(
            """
            SELECT *
            FROM sentiment_aspect_market_dataset
            WHERE aspect_sentiment_score IS NOT NULL
            ORDER BY date ASC, ticker ASC, aspect ASC
            """
        )
    except Exception as exc:
        logger.warning("Could not load sentiment_aspect_market_dataset: %s", exc)
        return pd.DataFrame()


def compute_overall_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall correlation between sentiment_index and forward returns."""
    rows: list[dict] = []

    for return_col in RETURN_COLS:
        if df.empty or return_col not in df.columns:
            rows.append(
                {
                    "level": "overall",
                    "group": "all",
                    "target": return_col,
                    "correlation": None,
                    "sample_size": 0,
                    "note": "missing_or_empty_dataset",
                }
            )
            continue

        clean = df[["sentiment_index", return_col]].dropna()

        rows.append(
            {
                "level": "overall",
                "group": "all",
                "target": return_col,
                "correlation": safe_corr(
                    clean["sentiment_index"],
                    clean[return_col],
                    min_samples=3,
                ),
                "sample_size": len(clean),
                "note": "ok" if len(clean) >= 3 else "too_few_samples",
            }
        )

    return pd.DataFrame(rows)


def compute_group_return_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average forward returns by final_sentiment group."""
    output_cols = [
        "final_sentiment",
        "sample_size",
        "avg_sentiment_index",
        "avg_forward_return_1d",
        "avg_forward_return_3d",
        "avg_forward_return_5d",
        "median_forward_return_1d",
        "median_forward_return_3d",
        "median_forward_return_5d",
        "avg_confidence",
        "avg_article_count",
    ]

    if df.empty or "final_sentiment" not in df.columns:
        return pd.DataFrame(columns=output_cols)

    result = (
        df.groupby("final_sentiment", as_index=False)
        .agg(
            sample_size=("ticker", "count"),
            avg_sentiment_index=("sentiment_index", "mean"),
            avg_forward_return_1d=("forward_return_1d", "mean"),
            avg_forward_return_3d=("forward_return_3d", "mean"),
            avg_forward_return_5d=("forward_return_5d", "mean"),
            median_forward_return_1d=("forward_return_1d", "median"),
            median_forward_return_3d=("forward_return_3d", "median"),
            median_forward_return_5d=("forward_return_5d", "median"),
            avg_confidence=("avg_confidence", "mean"),
            avg_article_count=("article_count", "mean"),
        )
        .sort_values("avg_sentiment_index", ascending=False)
    )

    return result[output_cols]


def compute_sector_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation by sector."""
    rows: list[dict] = []

    if df.empty or "sector" not in df.columns:
        return pd.DataFrame(
            columns=["level", "group", "target", "correlation", "sample_size", "note"]
        )

    for sector, group_df in df.groupby("sector", dropna=False):
        sector_name = sector if pd.notna(sector) else "UNKNOWN"

        for return_col in RETURN_COLS:
            if return_col not in group_df.columns:
                rows.append(
                    {
                        "level": "sector",
                        "group": sector_name,
                        "target": return_col,
                        "correlation": None,
                        "sample_size": 0,
                        "note": "missing_target",
                    }
                )
                continue

            clean = group_df[["sentiment_index", return_col]].dropna()
            corr = safe_corr(clean["sentiment_index"], clean[return_col], min_samples=3)

            rows.append(
                {
                    "level": "sector",
                    "group": sector_name,
                    "target": return_col,
                    "correlation": corr,
                    "sample_size": len(clean),
                    "note": "ok" if corr is not None else "too_few_or_no_variance",
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["target", "correlation"],
        ascending=[True, False],
        na_position="last",
    )


def compute_ticker_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation by ticker."""
    rows: list[dict] = []

    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame(
            columns=["level", "group", "target", "correlation", "sample_size", "note"]
        )

    for ticker, group_df in df.groupby("ticker"):
        for return_col in RETURN_COLS:
            if return_col not in group_df.columns:
                rows.append(
                    {
                        "level": "ticker",
                        "group": ticker,
                        "target": return_col,
                        "correlation": None,
                        "sample_size": 0,
                        "note": "missing_target",
                    }
                )
                continue

            clean = group_df[["sentiment_index", return_col]].dropna()
            corr = safe_corr(clean["sentiment_index"], clean[return_col], min_samples=5)

            rows.append(
                {
                    "level": "ticker",
                    "group": ticker,
                    "target": return_col,
                    "correlation": corr,
                    "sample_size": len(clean),
                    "note": "ok" if corr is not None else "too_few_or_no_variance",
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["target", "correlation"],
        ascending=[True, False],
        na_position="last",
    )


def compute_aspect_correlation(aspect_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aspect-level correlation."""
    rows: list[dict] = []

    if aspect_df.empty or "aspect" not in aspect_df.columns:
        return pd.DataFrame(
            columns=["level", "group", "target", "correlation", "sample_size", "note"]
        )

    for aspect, group_df in aspect_df.groupby("aspect"):
        for return_col in RETURN_COLS:
            if return_col not in group_df.columns:
                rows.append(
                    {
                        "level": "aspect",
                        "group": aspect,
                        "target": return_col,
                        "correlation": None,
                        "sample_size": 0,
                        "note": "missing_target",
                    }
                )
                continue

            clean = group_df[["aspect_sentiment_score", return_col]].dropna()
            corr = safe_corr(
                clean["aspect_sentiment_score"],
                clean[return_col],
                min_samples=3,
            )

            rows.append(
                {
                    "level": "aspect",
                    "group": aspect,
                    "target": return_col,
                    "correlation": corr,
                    "sample_size": len(clean),
                    "note": "ok" if corr is not None else "too_few_or_no_variance",
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["target", "correlation"],
        ascending=[True, False],
        na_position="last",
    )


def compute_hit_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute directional hit rate.

    Positive sentiment is expected to have positive forward return.
    Negative sentiment is expected to have negative forward return.
    Neutral is excluded.
    """
    rows: list[dict] = []

    if df.empty or "final_sentiment" not in df.columns:
        return pd.DataFrame(columns=["target", "hit_rate", "sample_size", "note"])

    directional_df = df[df["final_sentiment"].isin(["positive", "negative"])].copy()

    for return_col in RETURN_COLS:
        if return_col not in directional_df.columns:
            rows.append(
                {
                    "target": return_col,
                    "hit_rate": None,
                    "sample_size": 0,
                    "note": "missing_target",
                }
            )
            continue

        temp = directional_df[["final_sentiment", return_col]].dropna().copy()

        if temp.empty:
            rows.append(
                {
                    "target": return_col,
                    "hit_rate": None,
                    "sample_size": 0,
                    "note": "empty_after_dropna",
                }
            )
            continue

        temp["hit"] = (
            ((temp["final_sentiment"] == "positive") & (temp[return_col] > 0))
            | ((temp["final_sentiment"] == "negative") & (temp[return_col] < 0))
        )

        rows.append(
            {
                "target": return_col,
                "hit_rate": float(temp["hit"].mean()),
                "sample_size": len(temp),
                "note": "ok" if len(temp) >= 10 else "small_sample",
            }
        )

    return pd.DataFrame(rows)


def compute_market_coverage_report() -> pd.DataFrame:
    """Report market data coverage."""
    try:
        return query_dataframe(
            """
            SELECT
                ticker,
                MIN(date) AS min_date,
                MAX(date) AS max_date,
                COUNT(*) AS row_count
            FROM market_prices
            GROUP BY ticker
            ORDER BY ticker
            """
        )
    except Exception as exc:
        logger.warning("Could not compute market coverage report: %s", exc)
        return pd.DataFrame()


def compute_overlap_report() -> pd.DataFrame:
    """Report overlap between sentiment and market datasets."""
    try:
        return query_dataframe(
            """
            SELECT
                ticker,
                MIN(date) AS min_date,
                MAX(date) AS max_date,
                COUNT(*) AS row_count
            FROM sentiment_market_forward_dataset
            GROUP BY ticker
            ORDER BY row_count DESC, ticker
            """
        )
    except Exception as exc:
        logger.warning("Could not compute overlap report: %s", exc)
        return pd.DataFrame()


def generate_validation_report(
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate validation report dataframes and save to CSV."""
    output_dir = output_dir or os.getenv(
        "MARKET_VALIDATION_OUTPUT_DIR",
        "data/processed/validation",
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_sentiment_market_dataset()
    aspect_df = load_aspect_market_dataset()

    logger.info("Loaded sentiment_market rows=%s", len(df))
    logger.info("Loaded aspect_market rows=%s", len(aspect_df))

    reports = {
        "overall_correlation": compute_overall_correlation(df),
        "group_return_by_sentiment": compute_group_return_by_sentiment(df),
        "sector_correlation": compute_sector_correlation(df),
        "ticker_correlation": compute_ticker_correlation(df),
        "aspect_correlation": compute_aspect_correlation(aspect_df),
        "hit_rate": compute_hit_rate(df),
        "market_coverage": compute_market_coverage_report(),
        "sentiment_market_overlap": compute_overlap_report(),
    }

    for name, report_df in reports.items():
        csv_file = output_path / f"{name}.csv"
        report_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        logger.info("Saved %s rows=%s to %s", name, len(report_df), csv_file)

    return reports