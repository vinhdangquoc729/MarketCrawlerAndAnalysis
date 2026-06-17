"""CafeF crawler with retry, delay, timeline pagination and parser fallback."""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

import requests
from dotenv import load_dotenv

from src.crawlers.cafef_parser import parse_article_detail, parse_article_links
from src.preprocessing.text_cleaner import is_valid_article

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CafeFCategory:
    """Cấu hình cho từng chuyên mục CafeF."""

    name: str
    url: str
    timeline_id: int | None = None


class CafeFCrawler:
    """Crawler dùng để thu thập bài viết từ các chuyên mục CafeF."""

    CATEGORIES: dict[str, CafeFCategory] = {
        "thi_truong_chung_khoan": CafeFCategory(
            name="thi_truong_chung_khoan",
            url="https://cafef.vn/thi-truong-chung-khoan.chn",
            timeline_id=18831,
        ),
        "doanh_nghiep": CafeFCategory(
            name="doanh_nghiep",
            url="https://cafef.vn/doanh-nghiep.chn",
            timeline_id=18836,
        ),
        "ngan_hang": CafeFCategory(
            name="ngan_hang",
            url="https://cafef.vn/tai-chinh-ngan-hang.chn",
            timeline_id=18834,
        ),
        "bao_cao_phan_tich": CafeFCategory(
            name="bao_cao_phan_tich",
            url="https://cafef.vn/du-lieu/phan-tich-bao-cao.chn",
            timeline_id=None,
        ),
    }

    # Giữ lại mapping cũ để các job trước đó vẫn dùng được.
    CATEGORY_URLS: dict[str, str] = {
        name: category.url for name, category in CATEGORIES.items()
    }

    def __init__(
        self,
        delay_seconds: float | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
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

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;"
                    "q=0.9,image/avif,image/webp,*/*;q=0.8"
                ),
                "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
                "Connection": "keep-alive",
            }
        )

    @staticmethod
    def build_timeline_url(timeline_id: int, page: int) -> str:
        """Tạo URL API timeline/load-more của CafeF."""
        return f"https://cafef.vn/timelinelist/{timeline_id}/{page}.chn"

    def fetch_html(self, url: str, retries: int = 3) -> str:
        """Fetch HTML với retry để hạn chế lỗi mạng tạm thời."""
        last_error: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                time.sleep(self.delay_seconds)

                resp = self.session.get(url, timeout=self.timeout_seconds)
                resp.raise_for_status()

                # Set encoding để đọc tiếng Việt ổn định hơn.
                resp.encoding = resp.apparent_encoding or "utf-8"
                return resp.text

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Fetch failed attempt=%s url=%s error=%s",
                    attempt,
                    url,
                    exc,
                )

                # Backoff nhẹ giữa các lần retry.
                time.sleep(min(2 * attempt, 5))

        raise RuntimeError(f"Failed to fetch {url}: {last_error}")

    def _resolve_category(
        self,
        category_name: str,
        category_url: str | None = None,
    ) -> CafeFCategory:
        """Lấy cấu hình chuyên mục, đồng thời hỗ trợ truyền URL ngoài."""
        if category_name not in self.CATEGORIES:
            if not category_url:
                raise ValueError(
                    f"Unknown category={category_name}. "
                    "Please provide category_url or add it to CATEGORIES."
                )

            return CafeFCategory(
                name=category_name,
                url=category_url,
                timeline_id=None,
            )

        category = self.CATEGORIES[category_name]

        if category_url:
            return CafeFCategory(
                name=category.name,
                url=category_url,
                timeline_id=category.timeline_id,
            )

        return category

    def crawl_category_links(
        self,
        category_name: str,
        category_url: str | None = None,
        max_links: int = 200,
        use_timeline: bool = True,
        timeline_start_page: int = 1,
        timeline_max_pages: int = 10,
        max_empty_pages: int = 3,
        min_year: int | None = None,
    ) -> list[dict[str, Any]]:
        """Crawl danh sách link bài viết từ trang chuyên mục và timeline."""
        category = self._resolve_category(
            category_name=category_name,
            category_url=category_url,
        )

        all_links: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        # Crawl link có sẵn trên trang chuyên mục.
        html = self.fetch_html(category.url)
        initial_links = parse_article_links(html, category.url)

        stop_from_initial = False
        for item in initial_links:
            url = item.get("url")
            if not url or url in seen_urls:
                continue

            if min_year and item.get("published_at") and item["published_at"].year < min_year:
                stop_from_initial = True
                continue

            seen_urls.add(url)
            all_links.append(item)

            if len(all_links) >= max_links:
                logger.info(
                    "Reached max_links=%s from initial category=%s",
                    max_links,
                    category_name,
                )
                return all_links

        logger.info(
            "Initial links category=%s found=%s total=%s",
            category_name,
            len(initial_links),
            len(all_links),
        )

        if stop_from_initial:
            logger.info(
                "Stopped at initial page due to min_year=%s category=%s",
                min_year,
                category_name,
            )
            return all_links

        if not use_timeline:
            return all_links

        if category.timeline_id is None:
            logger.info(
                "No timeline_id configured for category=%s. "
                "Only initial page links were crawled.",
                category_name,
            )
            return all_links

        empty_pages = 0

        # Crawl thêm các trang timeline.
        for page in range(
            timeline_start_page,
            timeline_start_page + timeline_max_pages,
        ):
            timeline_url = self.build_timeline_url(category.timeline_id, page)

            try:
                timeline_html = self.fetch_html(timeline_url)
            except Exception as exc:
                logger.warning(
                    "Timeline fetch failed category=%s timeline_id=%s page=%s error=%s",
                    category_name,
                    category.timeline_id,
                    page,
                    exc,
                )
                empty_pages += 1

                if empty_pages >= max_empty_pages:
                    logger.info(
                        "Stop timeline because consecutive_fetch_failed_or_empty_pages=%s "
                        "category=%s page=%s",
                        empty_pages,
                        category_name,
                        page,
                    )
                    break

                continue

            page_links = parse_article_links(timeline_html, timeline_url)

            if not page_links:
                empty_pages += 1
                logger.info(
                    "No links found in timeline category=%s timeline_id=%s page=%s "
                    "empty_pages=%s/%s",
                    category_name,
                    category.timeline_id,
                    page,
                    empty_pages,
                    max_empty_pages,
                )

                if empty_pages >= max_empty_pages:
                    logger.info(
                        "Stop timeline because consecutive_empty_pages=%s category=%s page=%s",
                        empty_pages,
                        category_name,
                        page,
                    )
                    break

                continue

            added = 0
            stop_early = False

            for item in page_links:
                url = item.get("url")
                if not url or url in seen_urls:
                    continue

                if min_year and item.get("published_at") and item["published_at"].year < min_year:
                    stop_early = True
                    continue

                seen_urls.add(url)
                all_links.append(item)
                added += 1

                if len(all_links) >= max_links:
                    logger.info(
                        "Reached max_links=%s category=%s page=%s",
                        max_links,
                        category_name,
                        page,
                    )
                    return all_links

            if stop_early:
                logger.info(
                    "Stopping timeline due to min_year=%s category=%s page=%s",
                    min_year,
                    category_name,
                    page,
                )
                break

            dates = [
                item["published_at"]
                for item in page_links
                if item.get("published_at")
            ]
            oldest = min(dates).strftime("%Y-%m-%d") if dates else "unknown"

            logger.info(
                "Timeline category=%s timeline_id=%s page=%s found=%s added=%s total=%s oldest=%s",
                category_name,
                category.timeline_id,
                page,
                len(page_links),
                added,
                len(all_links),
                oldest,
            )

            if added == 0:
                empty_pages += 1
            else:
                empty_pages = 0

            if empty_pages >= max_empty_pages:
                logger.info(
                    "Stop timeline because consecutive_no_new_pages=%s category=%s page=%s",
                    empty_pages,
                    category_name,
                    page,
                )
                break

        return all_links

    def crawl_category(
        self,
        category_name: str,
        category_url: str | None = None,
        max_articles: int = 200,
        use_timeline: bool = True,
        timeline_start_page: int = 1,
        timeline_max_pages: int = 10,
        min_year: int | None = None,
        on_batch: "Callable[[list[dict]], None] | None" = None,
        batch_size: int = 100,
        workers: int = 10,
    ) -> list[dict[str, Any]]:
        """Crawl link và nội dung chi tiết bài viết của một chuyên mục."""
        links = self.crawl_category_links(
            category_name=category_name,
            category_url=category_url,
            max_links=max_articles,
            use_timeline=use_timeline,
            timeline_start_page=timeline_start_page,
            timeline_max_pages=timeline_max_pages,
            min_year=min_year,
        )

        logger.info(
            "Found %s links for category=%s — crawling with workers=%s",
            len(links),
            category_name,
            workers,
        )

        articles: list[dict[str, Any]] = []
        done = 0

        def _fetch_one(item: dict) -> dict[str, Any]:
            """Crawl chi tiết một bài viết, nếu lỗi thì trả về record fallback."""
            try:
                article = self.crawl_article(item["url"], category_name)

                # Dùng published_at từ trang danh sách nếu trang chi tiết không parse được.
                if not article.get("published_at") and item.get("published_at"):
                    article["published_at"] = item["published_at"]

                if not is_valid_article(article):
                    article["error_message"] = "invalid_or_short_article"

                return article

            except Exception as exc:
                logger.warning("Article crawl failed url=%s: %s", item["url"], exc)

                return {
                    "source": "CafeF",
                    "url": item["url"],
                    "category": category_name,
                    "title": item.get("title", ""),
                    "sapo": "",
                    "content": "",
                    "author": "",
                    "published_at": item.get("published_at"),
                    "raw_html": "",
                    "error_message": str(exc),
                }

        # Dùng thread vì crawl bài viết là tác vụ I/O-bound.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_item = {pool.submit(_fetch_one, item): item for item in links}

            for future in as_completed(future_to_item):
                article = future.result()
                articles.append(article)
                done += 1

                pub = article.get("published_at")
                pub_str = pub.strftime("%Y-%m-%d") if pub else "unknown"

                logger.info(
                    "Crawled article %s/%s category=%s date=%s title=%s",
                    done,
                    len(links),
                    category_name,
                    pub_str,
                    str(article.get("title", ""))[:100],
                )

                if on_batch and done % batch_size == 0:
                    on_batch(articles[-batch_size:])
                    logger.info(
                        "Flushed batch of %s articles category=%s total=%s",
                        batch_size,
                        category_name,
                        done,
                    )

        return articles

    def crawl_article(self, url: str, category_name: str) -> dict[str, Any]:
        """Crawl và parse một bài viết CafeF."""
        html = self.fetch_html(url)
        return parse_article_detail(html, url=url, category=category_name)

    def crawl_all_categories(
        self,
        max_articles_per_category: int = 200,
        use_timeline: bool = True,
        timeline_start_page: int = 1,
        timeline_max_pages: int = 10,
        min_year: int | None = None,
        on_batch: "Callable[[list[dict]], None] | None" = None,
        batch_size: int = 100,
        workers: int = 10,
    ) -> list[dict[str, Any]]:
        """Crawl toàn bộ các chuyên mục đã cấu hình."""
        all_articles: list[dict[str, Any]] = []

        for name, category in self.CATEGORIES.items():
            try:
                all_articles.extend(
                    self.crawl_category(
                        category_name=name,
                        category_url=category.url,
                        max_articles=max_articles_per_category,
                        use_timeline=use_timeline,
                        timeline_start_page=timeline_start_page,
                        timeline_max_pages=timeline_max_pages,
                        min_year=min_year,
                        on_batch=on_batch,
                        batch_size=batch_size,
                        workers=workers,
                    )
                )
            except Exception:
                logger.exception("Category crawl failed category=%s", name)

        return all_articles