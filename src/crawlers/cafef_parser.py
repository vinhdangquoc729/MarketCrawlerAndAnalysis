"""CafeF HTML parser utilities.

This parser supports both:
1. Full CafeF category/article HTML pages.
2. HTML fragments returned by CafeF load-more API:
   https://cafef.vn/timelinelist/{timeline_id}/{page}.chn
"""
from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup, Tag
from dateutil import parser as date_parser

try:
    import trafilatura
except Exception:
    trafilatura = None

from src.preprocessing.text_cleaner import clean_text

SOURCE_NAME = "CafeF"
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# CafeF article URL usually ends with a long numeric article id, for example:
# https://cafef.vn/tv2-phat-di-thong-bao-khan-sau-khi-chu-tich-bi-khoi-to-18826052120051431.chn
#
# Use 10+ digits instead of 12+ to avoid missing valid article URLs.
ARTICLE_URL_RE = re.compile(
    r"https?://cafef\.vn/.+-\d{10,}\.chn(?:\?.*)?$",
    re.IGNORECASE,
)

DATE_RE_LIST = [
    # Timeline API usually returns ISO-like date:
    # 2026-05-23T07:28:00
    re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),

    # Vietnamese display formats:
    # 24/05/2026 - 00:07
    # 24/05/2026 - 00:07 AM
    re.compile(r"\d{1,2}/\d{1,2}/\d{4}\s*[-|]\s*\d{1,2}:\d{2}(?:\s*[AP]M)?", re.IGNORECASE),

    # 24-05-2026 - 00:07
    # 24-05-2026 - 00:07 AM
    re.compile(r"\d{1,2}-\d{1,2}-\d{4}\s*[-|]\s*\d{1,2}:\d{2}(?:\s*[AP]M)?", re.IGNORECASE),

    # 24/05/2026 00:07
    re.compile(r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}(?:\s*[AP]M)?", re.IGNORECASE),

    # 24-05-2026 00:07
    re.compile(r"\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2}(?:\s*[AP]M)?", re.IGNORECASE),
]

STOP_MARKERS = [
    "Từ Khóa:",
    "Từ Khóa",
    "CÙNG CHUYÊN MỤC",
    "Cùng chuyên mục",
    "Xem theo ngày",
    "Công ty Tin tức Lãnh đạo",
    "Link bài gốc",
    "Lấy link!",
    "Theo dõi CafeF",
    "Đọc thêm",
    "Tin liên quan",
]

NOISE_TEXT_MARKERS = [
    "TIN MỚI",
    "Chia sẻ",
    "Copy link",
    "Lấy link!",
    "Link bài gốc",
]

SPONSORED_MARKERS = [
    "NỘI DUNG ĐƯỢC TÀI TRỢ",
    "Nội dung được tài trợ",
    "Sản phẩm nổi bật",
]


def parse_cafef_datetime(value: str | None) -> datetime | None:
    """Parse CafeF datetime safely.

    Supports:
    - ISO: 2026-05-23T07:28:00
    - Vietnamese: 23-05-2026 - 07:28
    - Vietnamese: 23/05/2026 - 07:28
    - CafeF header: 12-05-2026 - 08:00 AM
    """
    if not value:
        return None

    text = clean_text(str(value))

    if not text:
        return None

    extracted = _extract_date_from_text(text)
    if extracted:
        text = extracted.strip()

    # 1. ISO format: YYYY-MM-DDTHH:MM:SS
    # Không được replace dấu "-" trong ISO.
    iso_match = re.search(
        r"(?P<dt>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",
        text,
    )
    if iso_match:
        try:
            dt = datetime.fromisoformat(iso_match.group("dt"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=VN_TZ)
            return dt
        except Exception:
            pass

    # 2. Vietnamese format: DD-MM-YYYY - HH:MM hoặc DD/MM/YYYY - HH:MM
    vn_match = re.search(
        r"(?P<day>\d{1,2})[/-](?P<month>\d{1,2})[/-](?P<year>\d{4})"
        r"\s*(?:-|\|)?\s*"
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
        r"(?:\s*(?P<ampm>AM|PM))?",
        text,
        flags=re.IGNORECASE,
    )

    if vn_match:
        try:
            day = int(vn_match.group("day"))
            month = int(vn_match.group("month"))
            year = int(vn_match.group("year"))
            hour = int(vn_match.group("hour"))
            minute = int(vn_match.group("minute"))
            ampm = vn_match.group("ampm")

            if ampm:
                ampm = ampm.upper()

                # CafeF có thể ghi 08:00 AM, 02:30 PM
                if ampm == "PM" and hour < 12:
                    hour += 12
                elif ampm == "AM" and hour == 12:
                    hour = 0

            return datetime(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                tzinfo=VN_TZ,
            )
        except Exception:
            pass

    # 3. Fallback cuối cùng
    try:
        dt = date_parser.parse(
            text,
            dayfirst=True,
            yearfirst=False,
            fuzzy=True,
        )

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=VN_TZ)

        return dt

    except Exception:
        return None

def _drop_noise(soup: BeautifulSoup) -> None:
    """Remove obvious non-content nodes.

    Be careful not to remove .VCSortableInPreviewMode globally because CafeF
    often uses it for images/tables inside the article body.
    """
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    for selector in [
        ".ads",
        ".advertisement",
        ".banner",
        ".sidebar",
        ".rightcol",
        ".right-column",
        ".related",
        ".tinlienquan",
        ".share",
        ".comment",
        "#admzone",
        ".box-category",
        ".box_docnhieu",
        ".footer",
        "footer",
    ]:
        for tag in soup.select(selector):
            tag.decompose()


def _is_article_url(url: str) -> bool:
    """Return True if URL looks like a real CafeF article URL."""
    return bool(ARTICLE_URL_RE.match(url))


def _truncate_at_stop_markers(text: str) -> str:
    """Cut article content before common CafeF non-body sections."""
    if not text:
        return ""

    cut_pos = len(text)

    for marker in STOP_MARKERS:
        pos = text.find(marker)
        if pos != -1:
            cut_pos = min(cut_pos, pos)

    return clean_text(text[:cut_pos])


def _remove_noise_lines(text: str) -> str:
    """Remove standalone noisy lines from extracted content."""
    lines = []

    for line in text.splitlines():
        line = clean_text(line)

        if not line:
            continue

        if line in NOISE_TEXT_MARKERS:
            continue

        if len(line) <= 2:
            continue

        lines.append(line)

    return clean_text("\n".join(lines))


def _extract_date_from_text(text: str) -> str | None:
    """Extract date-like string from a text block."""
    if not text:
        return None

    for pattern in DATE_RE_LIST:
        match = pattern.search(text)
        if match:
            return match.group(0)

    return None


def _nearest_block_text(node: Tag) -> str:
    """Get text from nearest parent block around a link.

    This helps extract title/sapo/time from CafeF category or timelinelist fragments.
    """
    current: Tag | None = node

    for _ in range(5):
        if current is None:
            break

        text = clean_text(current.get_text(" "))
        if len(text) >= 20:
            return text

        parent = current.parent
        current = parent if isinstance(parent, Tag) else None

    return clean_text(node.get_text(" "))


def _link_published_at(a: Tag) -> "datetime | None":
    """Find published_at for a link by inspecting the enclosing article block.

    The timeline HTML structure is:
      <div role="article" class="tlitem ...">
        <h3><a href="...">Title</a></h3>           ← a is here
        <div class="tlitem-flex">
          <span class="time time-ago" title="2026-06-07T00:01:00">...</span>
        </div>
      </div>

    _nearest_block_text returns from the <a> itself (title text ≥20 chars),
    so it never sees the sibling <span class="time">. This function walks up to
    the article container and looks for the time span explicitly.
    """
    current: Tag | None = a
    for _ in range(8):
        if current is None:
            break
        # CafeF article containers: role=article, or classes like tlitem / box-category-item
        if current.name in ("div", "li", "article") and current is not a:
            role = current.get("role", "")
            cls = " ".join(current.get("class") or [])
            if role == "article" or any(
                k in cls for k in ("tlitem", "box-category-item", "knswli", "list-news-item")
            ):
                for span in current.find_all(["span", "time"], recursive=True):
                    # Try title attribute first (most reliable on timeline pages)
                    title_val = span.get("title") or ""
                    text_val = clean_text(span.get_text(" "))
                    for val in (title_val, text_val):
                        if val:
                            dt = parse_cafef_datetime(val)
                            if dt:
                                return dt
                break
        parent = current.parent
        current = parent if isinstance(parent, Tag) else None
    return None


def parse_article_links(html: str, base_url: str) -> list[dict]:
    """Parse real CafeF article links from category page or timelinelist HTML.

    Returns:
        [
            {
                "url": "...",
                "title": "...",
                "published_at": "... or None",
            }
        ]
    """
    soup = BeautifulSoup(html or "", "lxml")
    _drop_noise(soup)

    items: list[dict] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()

        if not href:
            continue

        url = urljoin(base_url, href).split("#")[0]

        if "cafef.vn" not in url:
            continue

        if not _is_article_url(url):
            continue

        if url in seen:
            continue

        title = clean_text(a.get("title") or a.get_text(" "))
        block_text = _nearest_block_text(a)

        # Sometimes the anchor contains only an image or empty text.
        # In that case, try to infer title from nearby block text.
        if not title or len(title) < 12:
            title = block_text

        title = clean_text(title)

        # Avoid extremely long block text becoming title.
        if len(title) > 260:
            # Prefer the first reasonable sentence-like chunk.
            parts = re.split(r"\s{2,}|(?<=\.)\s+", title)
            title = clean_text(parts[0]) if parts else title

        if not title or len(title) < 12:
            continue

        # Skip sponsored blocks if marker appears very close to this link.
        # Do not over-skip globally because some valid enterprise news may appear after ads.
        block_text_lower = block_text.lower()
        if any(marker.lower() in block_text_lower for marker in SPONSORED_MARKERS):
            continue

        published_at = _link_published_at(a)
        if published_at is None:
            published_text = _extract_date_from_text(block_text)
            published_at = parse_cafef_datetime(published_text) if published_text else None

        seen.add(url)
        items.append(
            {
                "url": url,
                "title": title,
                "published_at": published_at,
            }
        )

    return items


def _first_text(soup: BeautifulSoup, selectors: list[str]) -> str:
    """Return first non-empty text from selectors."""
    for selector in selectors:
        node = soup.select_one(selector)

        if node:
            value = clean_text(node.get_text(" "))
            if value:
                return value

    return ""


def _meta_content(soup: BeautifulSoup, *names: str) -> str:
    """Read meta property/name content."""
    for name in names:
        node = soup.find("meta", attrs={"property": name}) or soup.find(
            "meta",
            attrs={"name": name},
        )

        if node and node.get("content"):
            return clean_text(node.get("content"))

    return ""


def _extract_content_by_selectors(soup: BeautifulSoup) -> str:
    """Extract article body using common CafeF selectors."""
    selectors = [
        ".detail-content",
        ".article-content",
        ".newscontent",
        ".contentdetail",
        ".body",
        ".detail__content",
        ".knc-content",
        ".detail-content-body",
        ".entry-content",
        "article",
    ]

    for selector in selectors:
        node = soup.select_one(selector)

        if not node:
            continue

        for bad in node.select(
            "script, style, iframe, .ads, .advertisement, .share, .related, .tinlienquan"
        ):
            bad.decompose()

        content = clean_text(node.get_text("\n"))
        content = _remove_noise_lines(content)
        content = _truncate_at_stop_markers(content)

        if len(content) >= 300:
            return content

    return ""


def _extract_content_fallback(html: str) -> str:
    """Fallback content extraction using trafilatura."""
    if trafilatura is None:
        return ""

    extracted = trafilatura.extract(
        html or "",
        include_comments=False,
        include_tables=False,
        include_images=False,
        favor_precision=True,
    )

    if not extracted:
        return ""

    content = clean_text(extracted)
    content = _remove_noise_lines(content)
    content = _truncate_at_stop_markers(content)

    return content

def _extract_header_published_text(soup: BeautifulSoup) -> str | None:
    """Extract CafeF published time from the article header/byline area.

    Example header:
    Ánh Dương | 12-05-2026 - 08:00 AM | Doanh nghiệp
    """
    candidates: list[str] = []

    # Ưu tiên các block gần h1 vì CafeF để ngày đăng gần tiêu đề.
    title_node = soup.select_one("h1")

    if title_node:
        current: Tag | None = title_node

        for _ in range(5):
            if current is None:
                break

            text = clean_text(current.get_text(" "))
            if text:
                candidates.append(text)

            parent = current.parent
            current = parent if isinstance(parent, Tag) else None

    # Thêm các selector thường chứa thời gian ở đầu bài
    for selector in [
        ".knc-time",
        ".time",
        ".date",
        ".pdate",
        ".detail-time",
        ".article-time",
        ".news-time",
    ]:
        for node in soup.select(selector):
            text = clean_text(node.get_text(" "))
            if text:
                candidates.append(text)

    for text in candidates:
        date_text = _extract_date_from_text(text)
        if date_text:
            return date_text

    return None

def _extract_published_at(soup: BeautifulSoup) -> datetime | None:
    """Extract article published time from meta/time/header/selectors."""
    meta_time = _meta_content(
        soup,
        "article:published_time",
        "og:updated_time",
        "pubdate",
        "date",
    )

    if meta_time:
        return parse_cafef_datetime(meta_time)

    time_node = soup.find("time")
    if time_node:
        dt_value = time_node.get("datetime") or clean_text(time_node.get_text(" "))
        if dt_value:
            return parse_cafef_datetime(dt_value)

    # Quan trọng: lấy ngày ở đầu bài trước.
    header_date_text = _extract_header_published_text(soup)
    if header_date_text:
        return parse_cafef_datetime(header_date_text)

    published_text = _first_text(
        soup,
        [
            ".time",
            ".date",
            ".pdate",
            ".knc-time",
            ".detail-time",
            ".article-time",
            ".news-time",
        ],
    )

    if published_text:
        return parse_cafef_datetime(published_text)

    # Fallback cuối: chỉ scan phần trước "Link bài gốc" và "CÙNG CHUYÊN MỤC"
    # để tránh bắt nhầm ngày của nguồn gốc hoặc bài liên quan.
    page_text = clean_text(soup.get_text(" "))

    for marker in [
        "Link bài gốc",
        "CÙNG CHUYÊN MỤC",
        "Cùng chuyên mục",
        "Xem theo ngày",
        "Từ Khóa:",
        "Từ Khóa",
    ]:
        if marker in page_text:
            page_text = page_text.split(marker)[0]

    date_text = _extract_date_from_text(page_text)

    return parse_cafef_datetime(date_text) if date_text else None
   


def parse_article_detail(html: str, url: str, category: str) -> dict:
    """Parse a CafeF article page into normalized article fields."""
    soup = BeautifulSoup(html or "", "lxml")
    _drop_noise(soup)

    title = _first_text(
        soup,
        [
            "h1",
            ".title",
            ".detail-title",
            ".article-title",
            ".news-title",
            ".knc-title",
        ],
    ) or _meta_content(soup, "og:title", "twitter:title", "title")

    sapo = _first_text(
        soup,
        [
            ".sapo",
            ".chapeau",
            ".detail-sapo",
            ".knc-sapo",
            ".article-sapo",
            ".news-sapo",
            "h2",
        ],
    ) or _meta_content(soup, "description", "og:description")

    content = _extract_content_by_selectors(soup)

    if len(content) < 300:
        content = _extract_content_fallback(html)

    content = _truncate_at_stop_markers(content)

    author = _first_text(
        soup,
        [
            ".author",
            ".p-author",
            ".detail-author",
            ".name",
            ".knc-author",
        ],
    )

    published_at = _extract_published_at(soup)

    return {
        "source": SOURCE_NAME,
        "url": url,
        "category": category,
        "title": clean_text(title),
        "sapo": clean_text(sapo),
        "content": clean_text(content),
        "author": clean_text(author),
        "published_at": published_at,
        "raw_html": html,
    }