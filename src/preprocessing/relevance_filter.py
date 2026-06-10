"""Financial relevance scoring before running sentiment analysis."""
from __future__ import annotations

import re
import unicodedata


def unidecode(value: str) -> str:
    """Lightweight accent stripping fallback."""
    value = unicodedata.normalize("NFD", value or "")
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
from dataclasses import dataclass
from typing import Any


FINANCIAL_KEYWORDS = [
    "cổ phiếu", "mã chứng khoán", "vn-index", "hnx-index", "upcom",
    "thanh khoản", "khối ngoại", "tự doanh", "lợi nhuận", "doanh thu",
    "báo cáo tài chính", "kết quả kinh doanh", "cổ tức",
    "phát hành", "trái phiếu", "nợ vay", "m&a", "sáp nhập",
    "lãi suất", "tỷ giá", "tín dụng", "nợ xấu",
    "ngân hàng nhà nước", "lạm phát", "gdp", "cpi",
    "bất động sản", "đầu tư công", "xuất khẩu", "giá thép", "giá dầu",
    "định giá", "khuyến nghị", "mua ròng", "bán ròng",
    "pháp lý", "thi hành án", "điều tra", "khởi tố", "vàng", "hàng hóa",
]

NOISE_KEYWORDS = [
    "túi hermès", "túi hermes", "xe sang", "đại gia", "showbiz",
    "hoa hậu", "ca sĩ", "diễn viên", "đời sống",
    "du lịch", "ẩm thực", "lifestyle",
]

GOOD_CATEGORIES = {"thi_truong_chung_khoan", "doanh_nghiep", "ngan_hang", "bao_cao_phan_tich"}
MEDIUM_CATEGORIES = {"bat_dong_san", "vi_mo", "tai_chinh_quoc_te", "thi_truong"}
AMBIGUOUS_TICKERS = {"CEO", "GAS", "POW"}
MACRO_TERMS = {"lãi suất", "tỷ giá", "cpi", "gdp", "ngân hàng nhà nước", "lạm phát"}
COMMODITY_TERMS = {"giá dầu", "giá thép", "vàng", "hàng hóa"}
LEGAL_TERMS = {"pháp lý", "thi hành án", "điều tra", "khởi tố", "trái phiếu", "ngân hàng"}
SECTOR_TERMS = {"bất động sản", "ngân hàng", "thép", "chứng khoán", "dầu khí", "bán lẻ"}


@dataclass(frozen=True)
class DetectedTicker:
    ticker: str
    alias: str
    alias_type: str
    weight: float


def normalize_text(value: str | None, remove_tone: bool = False) -> str:
    """Normalize text for matching."""
    text = unicodedata.normalize("NFC", value or "")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    if remove_tone:
        text = unidecode(text)
    return text


def _contains_phrase(text: str, phrase: str) -> bool:
    return normalize_text(phrase) in text


def detect_keywords(text: str, keywords: list[str]) -> list[str]:
    """Detect keywords in normalized text."""
    normalized = normalize_text(text)
    normalized_no_tone = normalize_text(text, remove_tone=True)
    found: list[str] = []
    for keyword in keywords:
        keyword_norm = normalize_text(keyword)
        keyword_no_tone = normalize_text(keyword, remove_tone=True)
        if keyword_norm in normalized or keyword_no_tone in normalized_no_tone:
            found.append(keyword)
    return sorted(set(found))


def _ticker_token_match(text_original: str, ticker: str) -> bool:
    pattern = rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])"
    return re.search(pattern, text_original) is not None


def detect_tickers(text: str, aliases: list[dict[str, Any]]) -> list[DetectedTicker]:
    """Detect tickers from alias rows.

    Ticker aliases are matched as independent uppercase tokens. Ambiguous tickers
    such as CEO/GAS/POW are accepted only when uppercase token exists.
    """
    normalized = normalize_text(text)
    normalized_no_tone = normalize_text(text, remove_tone=True)
    detected: dict[str, DetectedTicker] = {}

    for row in aliases:
        ticker = str(row["ticker"]).upper()
        alias = str(row["alias"])
        alias_norm = normalize_text(str(row.get("alias_normalized") or alias))
        alias_no_tone = normalize_text(alias, remove_tone=True)
        alias_type = str(row.get("alias_type") or "")
        weight = float(row.get("weight") or 1.0)

        matched = False
        if alias_type == "ticker" or alias.upper() == ticker:
            matched = _ticker_token_match(text, ticker)
            if ticker in AMBIGUOUS_TICKERS and not matched:
                continue
        else:
            matched = alias_norm in normalized or alias_no_tone in normalized_no_tone

        if matched:
            current = detected.get(ticker)
            candidate = DetectedTicker(ticker=ticker, alias=alias, alias_type=alias_type, weight=weight)
            if current is None or candidate.weight > current.weight:
                detected[ticker] = candidate

    return sorted(detected.values(), key=lambda x: x.ticker)


def _infer_relevance_type(detected: list[DetectedTicker], keywords: list[str], score: float) -> str:
    if score < 2:
        return "irrelevant"
    if detected:
        return "direct_stock_relevant"
    kws = set(keywords)
    if kws & MACRO_TERMS:
        return "macro_relevant"
    if kws & COMMODITY_TERMS:
        return "commodity_relevant"
    if kws & LEGAL_TERMS:
        return "legal_indirect"
    if kws & SECTOR_TERMS:
        return "sector_relevant"
    return "sector_relevant" if score >= 2 else "irrelevant"


def score_article_relevance(article: dict[str, Any], aliases: list[dict[str, Any]]) -> dict[str, Any]:
    """Score article financial relevance."""
    title = article.get("title") or ""
    sapo = article.get("sapo") or ""
    content = article.get("content") or ""
    category = article.get("category") or ""
    text = f"{title} {sapo} {content}"

    detected = detect_tickers(text, aliases)
    financial_keywords = detect_keywords(text, FINANCIAL_KEYWORDS)
    noise_keywords = detect_keywords(text, NOISE_KEYWORDS)

    score = 0.0
    if detected:
        direct = [d for d in detected if d.weight >= 0.9]
        indirect = [d for d in detected if d.weight < 0.9]
        if direct:
            score += 5
        if indirect and not direct:
            score += min(sum(max(2.0, d.weight * 5) for d in indirect), 4)
        elif indirect:
            score += min(sum(d.weight for d in indirect), 2)

    score += min(len(financial_keywords), 5)
    if category in GOOD_CATEGORIES:
        score += 2
    elif category in MEDIUM_CATEGORIES:
        score += 1
    score -= min(len(noise_keywords) * 2, 6)
    if not detected and not financial_keywords:
        score -= 4

    if score >= 5:
        decision = "process_sentiment"
    elif 2 <= score < 5:
        decision = "review_later"
    else:
        decision = "skip_sentiment"

    relevance_type = _infer_relevance_type(detected, financial_keywords, score)
    detected_tickers = [d.ticker for d in detected]
    reason_parts = [f"score={score:.1f}"]
    if detected_tickers:
        reason_parts.append(f"tickers={','.join(detected_tickers)}")
    if financial_keywords:
        reason_parts.append(f"financial_keywords={','.join(financial_keywords[:8])}")
    if noise_keywords:
        reason_parts.append(f"noise={','.join(noise_keywords)}")

    return {
        "article_id": article.get("article_id"),
        "relevance_type": relevance_type,
        "relevance_score": round(score, 2),
        "decision": decision,
        "reason": "; ".join(reason_parts),
        "detected_tickers": detected_tickers,
        "detected_keywords": financial_keywords,
    }
