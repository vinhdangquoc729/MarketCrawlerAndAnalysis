"""Time utilities for Vietnam timezone."""
from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dateutil import parser as date_parser
from dotenv import load_dotenv

load_dotenv()

APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Ho_Chi_Minh")
VN_TZ = ZoneInfo(APP_TIMEZONE)


def now_vn() -> datetime:
    """Return current timezone-aware Vietnam datetime."""
    return datetime.now(VN_TZ)


def ensure_vn_timezone(dt: datetime | None) -> datetime | None:
    """Convert naive/aware datetime to Asia/Ho_Chi_Minh aware datetime.

    If dt is naive, assume it is already Vietnam local time.
    """
    if dt is None:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=VN_TZ)

    return dt.astimezone(VN_TZ)


def parse_datetime(value: str | None) -> datetime | None:
    """Parse datetime string and normalize to Vietnam timezone.

    CafeF usually displays local Vietnam time. If parsed datetime is naive,
    we attach Asia/Ho_Chi_Minh.
    """
    if not value:
        return None

    text = value.strip()

    if not text:
        return None

    try:
        dt = date_parser.parse(text, dayfirst=True, fuzzy=True)
        return ensure_vn_timezone(dt)
    except Exception:
        return None


def to_vn_date(dt: datetime | None):
    """Return Vietnam local date from datetime."""
    dt = ensure_vn_timezone(dt)
    return dt.date() if dt else None