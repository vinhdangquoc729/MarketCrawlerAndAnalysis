"""Vietnamese sentence splitting utilities."""
from __future__ import annotations

import re

from src.preprocessing.text_cleaner import clean_text


SENTENCE_END_RE = re.compile(r"(?<=[.!?。])\s+")


def build_full_text(title: str | None, sapo: str | None, content: str | None) -> str:
    """Combine title, sapo and content into one text for NLP."""
    parts = [
        clean_text(title),
        clean_text(sapo),
        clean_text(content),
    ]
    return clean_text(". ".join(part for part in parts if part))


def split_sentences(text: str | None, min_length: int = 10) -> list[str]:
    """Split Vietnamese financial news into sentences.

    This is a simple rule-based splitter for demo.
    It is good enough for sentence-level context extraction.
    """
    cleaned = clean_text(text)

    if not cleaned:
        return []

    cleaned = re.sub(r"\s+", " ", cleaned)
    raw_sentences = SENTENCE_END_RE.split(cleaned)

    sentences: list[str] = []

    for sentence in raw_sentences:
        sentence = clean_text(sentence)

        if len(sentence) >= min_length:
            sentences.append(sentence)

    return sentences