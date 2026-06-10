"""Run sentiment-market validation reports."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.storage.db import query_dataframe
from src.utils.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)


RETURN_COLS = [
    "forward_return_1d",
    "forward_return_3d",
    "forward_return_5d",
]


def load_sentiment_market_dataset() -> pd.DataFrame:
    """Load ticker-level sentiment-market validation dataset."""
    try:
        return query_dataframe(
            """
            SELECT *
            FROM sentiment_market_forward_dataset
            WHERE sentiment_index IS NOT NULL
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
            """
        )
    except Exception as exc:
        logger.warning("Could not load sentiment_aspect_market_dataset: %s", exc)
        return pd.DataFrame()


def compute_overall_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall correlation between sentiment_index and forward returns."""
    rows = []

    for return_col in RETURN_COLS:
        if df.empty or return_col not in df.columns:
            rows.append(
                {
                    "level": "overall",
                    "group": "all",
                    "target": return_col,
                    "correlation": None,
                    "sample_size": 0,
                }
            )
            continue

        clean = df[["sentiment_index", return_col]].dropna()

        rows.append(
            {
                "level": "overall",
                "group": "all",
                "target": return_col,
                "correlation": clean["sentiment_index"].corr(clean[return_col])
                if len(clean) >= 3
                else None,
                "sample_size": len(clean),
            }
        )

    return pd.DataFrame(rows)


def compute_group_return_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average forward returns by sentiment group."""
    if df.empty or "final_sentiment" not in df.columns:
        return pd.DataFrame(
            columns=[
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
        )

    return (
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


def compute_sector_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation by sector."""
    rows = []

    if df.empty or "sector" not in df.columns:
        return pd.DataFrame(columns=["level", "group", "target", "correlation", "sample_size"])

    for sector, group_df in df.groupby("sector", dropna=False):
        sector_name = sector if pd.notna(sector) else "UNKNOWN"

        for return_col in RETURN_COLS:
            if return_col not in group_df.columns:
                sample_size = 0
                corr = None
            else:
                clean = group_df[["sentiment_index", return_col]].dropna()
                sample_size = len(clean)
                corr = clean["sentiment_index"].corr(clean[return_col]) if sample_size >= 3 else None

            rows.append(
                {
                    "level": "sector",
                    "group": sector_name,
                    "target": return_col,
                    "correlation": corr,
                    "sample_size": sample_size,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["target", "correlation"],
        ascending=[True, False],
        na_position="last",
    )


def compute_ticker_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation by ticker."""
    rows = []

    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame(columns=["level", "group", "target", "correlation", "sample_size"])

    for ticker, group_df in df.groupby("ticker"):
        for return_col in RETURN_COLS:
            if return_col not in group_df.columns:
                sample_size = 0
                corr = None
            else:
                clean = group_df[["sentiment_index", return_col]].dropna()
                sample_size = len(clean)
                corr = clean["sentiment_index"].corr(clean[return_col]) if sample_size >= 5 else None

            rows.append(
                {
                    "level": "ticker",
                    "group": ticker,
                    "target": return_col,
                    "correlation": corr,
                    "sample_size": sample_size,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["target", "correlation"],
        ascending=[True, False],
        na_position="last",
    )


def compute_aspect_correlation(aspect_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aspect-level correlation."""
    rows = []

    if aspect_df.empty or "aspect" not in aspect_df.columns:
        return pd.DataFrame(columns=["level", "group", "target", "correlation", "sample_size"])

    for aspect, group_df in aspect_df.groupby("aspect"):
        for return_col in RETURN_COLS:
            if return_col not in group_df.columns:
                sample_size = 0
                corr = None
            else:
                clean = group_df[["aspect_sentiment_score", return_col]].dropna()
                sample_size = len(clean)
                corr = (
                    clean["aspect_sentiment_score"].corr(clean[return_col])
                    if sample_size >= 3
                    else None
                )

            rows.append(
                {
                    "level": "aspect",
                    "group": aspect,
                    "target": return_col,
                    "correlation": corr,
                    "sample_size": sample_size,
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
    rows = []

    if df.empty or "final_sentiment" not in df.columns:
        return pd.DataFrame(columns=["target", "hit_rate", "sample_size"])

    directional_df = df[df["final_sentiment"].isin(["positive", "negative"])].copy()

    for return_col in RETURN_COLS:
        if return_col not in directional_df.columns:
            rows.append(
                {
                    "target": return_col,
                    "hit_rate": None,
                    "sample_size": 0,
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
            }
        )

    return pd.DataFrame(rows)


def generate_validation_report(output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Generate validation report dataframes and save to CSV."""
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
    }

    for name, report_df in reports.items():
        csv_file = output_path / f"{name}.csv"
        report_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        logger.info("Saved %s rows=%s to %s", name, len(report_df), csv_file)

    return reports


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run sentiment validation reports."
    )
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="Compatibility flag. Market data import is handled by run_market_validation_pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("MARKET_VALIDATION_OUTPUT_DIR", "data/processed/validation"),
        help="Output directory for validation CSV reports.",
    )
    args = parser.parse_args()

    reports = generate_validation_report(output_dir=args.output_dir)

    logger.info("===== VALIDATION SUMMARY =====")
    for name, df in reports.items():
        logger.info("%s rows=%s", name, len(df))

    logger.info("Sentiment validation complete.")


if __name__ == "__main__":
    main()