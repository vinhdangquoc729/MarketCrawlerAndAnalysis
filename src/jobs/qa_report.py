"""Generate data quality report for crawled articles."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from src.storage.db import query_dataframe
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def _explode_counter(series: pd.Series) -> Counter:
    counter: Counter = Counter()
    for value in series.dropna():
        if isinstance(value, list):
            counter.update(value)
        elif isinstance(value, str):
            cleaned = value.strip("{}[]")
            if cleaned:
                counter.update([x.strip().strip('"\'') for x in cleaned.split(",") if x.strip()])
    return counter


def main() -> None:
    setup_logging()
    articles = query_dataframe("SELECT * FROM articles")
    relevance = query_dataframe("SELECT * FROM article_relevance")

    print("\n===== QA REPORT =====")
    print(f"Tổng số bài: {len(articles)}")
    if not articles.empty:
        print("\nSố bài theo category:")
        print(articles.groupby("category").size().sort_values(ascending=False))
        print("\nSố bài theo status:")
        print(articles.groupby("status").size().sort_values(ascending=False))
        short_count = articles[articles["content"].fillna("").str.len() < 300].shape[0]
        duplicate_hash = articles[articles["content_hash"].notna()].duplicated("content_hash").sum()
        print(f"\nContent rỗng/quá ngắn (<300 ký tự): {short_count}")
        print(f"Duplicate content_hash: {duplicate_hash}")

    if not relevance.empty:
        print("\nSố bài theo decision:")
        print(relevance.groupby("decision").size().sort_values(ascending=False))
        print("\nSố bài theo relevance_type:")
        print(relevance.groupby("relevance_type").size().sort_values(ascending=False))
        print("\nTop detected tickers:")
        print(_explode_counter(relevance["detected_tickers"]).most_common(20))
        print("\nTop detected keywords:")
        print(_explode_counter(relevance["detected_keywords"]).most_common(20))

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {"metric": "total_articles", "value": len(articles)},
            {"metric": "total_relevance", "value": len(relevance)},
        ]
    )
    summary.to_csv(output_dir / "qa_report.csv", index=False)
    logger.info("Exported %s", output_dir / "qa_report.csv")


if __name__ == "__main__":
    main()
