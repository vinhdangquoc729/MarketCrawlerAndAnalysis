"""Ticker-specific corporate event crawler using CafeF's Events API.

Mirrors the scrape_events + fetch_articles approach from stock_news but saves
directly to PostgreSQL via bulk_upsert_articles, with detected_tickers
pre-populated from the API response.

Endpoint:
  GET /du-lieu/Ajax/Events_RelatedNews_New.aspx
      ?symbol={ticker}&floorID=0&configID=0&PageIndex={page}&PageSize=30&Type=2

Response: HTML <ul> with <li> items containing a <time> tag and an <a> tag.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from src.crawlers.cafef_parser import parse_article_detail, parse_cafef_datetime
from src.utils.hashing import make_article_id, make_content_hash

logger = logging.getLogger(__name__)

EVENTS_URL = (
    "https://cafef.vn/du-lieu/Ajax/Events_RelatedNews_New.aspx"
    "?symbol={symbol}&floorID=0&configID=0&PageIndex={page}&PageSize=30&Type=2"
)

BASE_URL = "https://cafef.vn"
CATEGORY = "corporate_events"
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


class CafeFEventsCrawler:
    """Crawler for ticker-specific corporate event articles from CafeF."""

    def __init__(
        self,
        delay_seconds: float | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        # Use env config if arguments are not provided.
        self.delay_seconds = (
            delay_seconds
            if delay_seconds is not None
            else float(os.getenv("CRAWL_DELAY_SECONDS", "1.0"))
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else int(os.getenv("CRAWL_TIMEOUT_SECONDS", "15"))
        )

        # Reuse session connections and set browser-like headers.
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html, */*",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
            "Referer": "https://cafef.vn/",
        })

    def _fetch(self, url: str, retries: int = 3) -> str:
        """Fetch HTML content with retry and simple backoff."""
        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                time.sleep(self.delay_seconds)

                resp = self.session.get(url, timeout=self.timeout_seconds)
                resp.raise_for_status()

                # Ensure Vietnamese text is decoded correctly.
                resp.encoding = resp.apparent_encoding or "utf-8"
                return resp.text

            except Exception as exc:
                last_exc = exc
                logger.warning("Fetch attempt=%s url=%s error=%s", attempt, url, exc)

                # Wait a little longer after each failed attempt.
                time.sleep(min(2 * attempt, 5))

        raise RuntimeError(f"Failed to fetch {url}: {last_exc}")

    def _parse_event_list(self, html: str, ticker: str) -> list[dict]:
        """Parse event links from CafeF Events API response."""
        soup = BeautifulSoup(html, "lxml")
        events = []

        for li in soup.find_all("li"):
            a_tag = li.find("a")
            if not a_tag:
                continue

            href = a_tag.get("href", "")
            url = BASE_URL + href.split("?")[0] if href.startswith("/") else href

            if not url or "cafef.vn" not in url:
                continue

            title = a_tag.get("title") or a_tag.get_text(strip=True)

            time_tag = li.find("time")
            raw_date = time_tag.get_text(strip=True) if time_tag else None
            published_at = parse_cafef_datetime(raw_date) if raw_date else None

            events.append({
                "ticker": ticker.upper(),
                "url": url,
                "title": title,
                "published_at": published_at,
            })

        return events

    def fetch_event_links(
        self,
        ticker: str,
        max_pages: int = 10,
        min_year: int | None = None,
        page_size: int = 30,
    ) -> list[dict]:
        """Fetch event links for one ticker across multiple API pages."""
        all_events: list[dict] = []
        seen: set[str] = set()

        for page in range(1, max_pages + 1):
            url = EVENTS_URL.format(
                symbol=ticker.upper(),
                page=page,
                size=page_size,
            )

            try:
                html = self._fetch(url)
            except Exception as exc:
                logger.warning(
                    "Events API failed ticker=%s page=%s: %s",
                    ticker,
                    page,
                    exc,
                )
                break

            events = self._parse_event_list(html, ticker)

            if not events:
                logger.info(
                    "No events returned ticker=%s page=%s — stopping.",
                    ticker,
                    page,
                )
                break

            added = 0
            stop_early = False

            for ev in events:
                if ev["url"] in seen:
                    continue

                # Stop when older data is reached.
                if min_year and ev["published_at"]:
                    year = ev["published_at"].year
                    if year < min_year:
                        stop_early = True
                        continue

                seen.add(ev["url"])
                all_events.append(ev)
                added += 1

            logger.info(
                "Events API ticker=%s page=%s found=%s added=%s total=%s",
                ticker,
                page,
                len(events),
                added,
                len(all_events),
            )

            if stop_early or len(events) < page_size:
                break

        return all_events

    def crawl_ticker(
        self,
        ticker: str,
        max_pages: int = 10,
        min_year: int | None = None,
    ) -> list[dict]:
        """Fetch event links and enrich each event with full article content."""
        links = self.fetch_event_links(
            ticker=ticker,
            max_pages=max_pages,
            min_year=min_year,
        )

        logger.info("Fetched event links ticker=%s count=%s", ticker, len(links))

        articles: list[dict] = []
        now = datetime.now(tz=timezone.utc)

        for idx, link in enumerate(links, start=1):
            url = link["url"]

            try:
                html = self._fetch(url)
                parsed = parse_article_detail(html, url=url, category=CATEGORY)

            except Exception as exc:
                logger.warning("Article fetch failed url=%s: %s", url, exc)

                # Fallback record keeps the article URL and event metadata.
                parsed = {
                    "source": "CafeF",
                    "url": url,
                    "category": CATEGORY,
                    "title": link.get("title", ""),
                    "sapo": "",
                    "content": "",
                    "author": "",
                    "published_at": link.get("published_at"),
                    "raw_html": "",
                }

            published_at = parsed.get("published_at") or link.get("published_at")

            # For /du-lieu/ URLs, the API title is more accurate than parsed page title.
            title = link.get("title") or parsed.get("title") or ""
            content = parsed.get("content") or ""

            article = {
                "article_id": make_article_id("cafef", url),
                "url": url,
                "title": title,
                "sapo": parsed.get("sapo") or "",
                "content": content,
                "category": CATEGORY,
                "crawl_at": now,
                "published_at": published_at,
                "status": "raw",
                "content_hash": make_content_hash(title, content, published_at),
                "detected_tickers": [ticker.upper()],
                "error_message": None,
            }

            logger.info(
                "Crawled event article %s/%s ticker=%s title=%s",
                idx,
                len(links),
                ticker,
                title[:80],
            )

            articles.append(article)

        return articles

    def crawl_tickers(
        self,
        tickers: list[str],
        max_pages: int = 10,
        min_year: int | None = None,
    ) -> list[dict]:
        """Crawl corporate event articles for multiple tickers."""
        all_articles: list[dict] = []

        for idx, ticker in enumerate(tickers, start=1):
            logger.info(
                "Crawling corporate events ticker=%s (%s/%s)",
                ticker,
                idx,
                len(tickers),
            )

            try:
                articles = self.crawl_ticker(
                    ticker=ticker,
                    max_pages=max_pages,
                    min_year=min_year,
                )

                all_articles.extend(articles)

                logger.info(
                    "Done ticker=%s articles=%s total_so_far=%s",
                    ticker,
                    len(articles),
                    len(all_articles),
                )

            except Exception:
                logger.exception("Failed to crawl ticker=%s", ticker)

        return all_articles