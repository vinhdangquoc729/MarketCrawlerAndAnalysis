"""Hashing utilities."""
from __future__ import annotations

import hashlib


def _safe(value: object) -> str:
    return "" if value is None else str(value)


def make_content_hash(title: str | None, content: str | None, published_at: object | None = None) -> str:
    """Create a stable content hash from article fields."""
    raw = "||".join([_safe(title).strip(), _safe(content).strip(), _safe(published_at).strip()])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def make_article_id(source: str, url: str) -> str:
    """Create a stable article id from source and URL."""
    raw = f"{source.lower().strip()}::{url.strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()
