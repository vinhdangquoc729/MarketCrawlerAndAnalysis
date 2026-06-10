"""Text cleaning utilities for Vietnamese financial news."""
from __future__ import annotations

import html
import re
import unicodedata
from urllib.parse import urlparse


BOILERPLATE_PATTERNS = [
    r"Đọc thêm\s*:.*$",
    r"Tin liên quan\s*:.*$",
    r"Theo CafeF\s*$",
    r"Theo Cafef\s*$",
    r"Nguồn\s*:.*$",
    r"Copy link\s*$",
    r"Lấy link!\s*$",
    r"Link bài gốc\s+.*$",
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

NOISE_LINES = {
    "TIN MỚI",
    "Chia sẻ",
    "Copy link",
    "Lấy link!",
    "Link bài gốc",
}


def normalize_vietnamese_text(text: str | None, keep_newlines: bool = False) -> str:
    """Normalize unicode, HTML entities and repeated whitespace.

    Args:
        text: Input text.
        keep_newlines: If True, preserve newlines for article-body parsing.
    """
    if not text:
        return ""

    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")

    if keep_newlines:
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
        return text.strip()

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_vietnamese_accents(text: str | None) -> str:
    """Remove Vietnamese accents for alias matching / keyword fallback.

    Example:
        'Hòa Phát' -> 'Hoa Phat'
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", text)


def truncate_at_stop_markers(text: str | None) -> str:
    """Cut text before common non-article sections."""
    cleaned = normalize_vietnamese_text(text, keep_newlines=True)

    if not cleaned:
        return ""

    cut_pos = len(cleaned)

    for marker in STOP_MARKERS:
        pos = cleaned.find(marker)
        if pos != -1:
            cut_pos = min(cut_pos, pos)

    return cleaned[:cut_pos].strip()


def remove_noise_lines(text: str | None) -> str:
    """Remove standalone noise lines like TIN MỚI / Chia sẻ / Copy link."""
    cleaned = normalize_vietnamese_text(text, keep_newlines=True)

    if not cleaned:
        return ""

    lines: list[str] = []

    for line in cleaned.splitlines():
        line = normalize_vietnamese_text(line)

        if not line:
            continue

        if line in NOISE_LINES:
            continue

        if len(line) <= 2:
            continue

        lines.append(line)

    return "\n".join(lines).strip()


def remove_boilerplate(text: str | None) -> str:
    """Remove common news boilerplate snippets."""
    cleaned = normalize_vietnamese_text(text, keep_newlines=True)

    if not cleaned:
        return ""

    cleaned = truncate_at_stop_markers(cleaned)

    for pattern in BOILERPLATE_PATTERNS:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        )

    cleaned = remove_noise_lines(cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)

    return cleaned.strip()


def clean_text(text: str | None, keep_newlines: bool = False) -> str:
    """Remove HTML residue, repeated whitespace and boilerplate.

    Args:
        text: Input text.
        keep_newlines: Preserve article paragraph breaks when True.
    """
    cleaned = normalize_vietnamese_text(text, keep_newlines=True)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = remove_boilerplate(cleaned)

    if keep_newlines:
        cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)
        return cleaned.strip()

    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_for_matching(text: str | None) -> str:
    """Normalize text for alias / keyword matching."""
    cleaned = clean_text(text)
    cleaned = remove_vietnamese_accents(cleaned)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9\s\-_/\.]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def is_valid_article(article: dict, min_content_length: int = 250) -> bool:
    """Return True when article has enough content for NLP."""
    title = clean_text(article.get("title"))
    content = clean_text(article.get("content"))
    url = article.get("url", "")

    parsed = urlparse(url)

    return bool(
        title
        and content
        and len(title) >= 8
        and len(content) >= min_content_length
        and parsed.scheme in {"http", "https"}
        and parsed.netloc
    )