"""Entity linker for mapping article text mentions to stock tickers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from src.preprocessing.text_cleaner import clean_text, normalize_for_matching


COMMON_AMBIGUOUS_TICKERS = {
    "CEO", "GAS", "POW", "DIG", "FIT", "TIG", "PEN", "HOT", "TNT"
}
@dataclass(frozen=True)
class AliasRecord:
    ticker: str
    alias: str
    alias_normalized: str
    alias_type: str | None
    weight: float


@dataclass(frozen=True)
class EntityMention:
    ticker: str
    entity_text: str
    alias_type: str | None
    alias_weight: float
    sentence_index: int
    sentence: str
    start_char: int | None = None
    end_char: int | None = None


@lru_cache(maxsize=None)
def _compile_alias_pattern(alias: str, alias_type: str | None = None) -> re.Pattern:
    """Compile regex for matching alias in original text.

    Tickers should match as independent tokens.
    Company names/brands can match case-insensitively.
    """
    escaped = re.escape(alias)

    if alias_type == "ticker" or alias.isupper():
        return re.compile(rf"(?<![A-Z0-9]){escaped}(?![A-Z0-9])")

    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def _is_ambiguous_ticker_false_positive(
    alias: str,
    alias_type: str | None,
    sentence: str,
) -> bool:
    """Avoid obvious false positives for short ambiguous tickers.

    Example: CEO can be job title, not necessarily ticker CEO.
    """
    alias_upper = alias.upper()

    if alias_type != "ticker":
        return False

    if alias_upper not in COMMON_AMBIGUOUS_TICKERS:
        return False

    # If exact uppercase ticker appears, keep it.
    if re.search(rf"(?<![A-Z0-9]){re.escape(alias_upper)}(?![A-Z0-9])", sentence):
        return False

    return True


def find_entities_in_sentences(
    sentences: list[str],
    aliases: Iterable[AliasRecord],
    restrict_tickers: set[str] | None = None,
) -> list[EntityMention]:
    """Find ticker/entity mentions in sentences using alias mapping.

    Args:
        sentences: List of article sentences.
        aliases: Loaded ticker aliases.
        restrict_tickers: Optional set of tickers from article_relevance.detected_tickers.
            If provided, only aliases belonging to these tickers are used.
    """
    mentions: list[EntityMention] = []

    sorted_aliases = sorted(
        aliases,
        key=lambda a: len(a.alias),
        reverse=True,
    )

    for sentence_index, sentence in enumerate(sentences):
        original_sentence = clean_text(sentence)

        if not original_sentence:
            continue

        found_keys: set[tuple[str, str, int]] = set()

        for alias_record in sorted_aliases:
            if restrict_tickers and alias_record.ticker not in restrict_tickers:
                continue

            alias = alias_record.alias.strip()

            if not alias:
                continue

            if _is_ambiguous_ticker_false_positive(
                alias=alias,
                alias_type=alias_record.alias_type,
                sentence=original_sentence,
            ):
                continue

            pattern = _compile_alias_pattern(alias, alias_record.alias_type)

            for match in pattern.finditer(original_sentence):
                key = (alias_record.ticker, alias, match.start())

                if key in found_keys:
                    continue

                found_keys.add(key)

                mentions.append(
                    EntityMention(
                        ticker=alias_record.ticker,
                        entity_text=match.group(0),
                        alias_type=alias_record.alias_type,
                        alias_weight=float(alias_record.weight or 1.0),
                        sentence_index=sentence_index,
                        sentence=original_sentence,
                        start_char=match.start(),
                        end_char=match.end(),
                    )
                )

    return mentions


def make_alias_records(rows: list[dict]) -> list[AliasRecord]:
    """Convert DB rows to AliasRecord list."""
    records: list[AliasRecord] = []

    for row in rows:
        alias = row.get("alias") or ""

        records.append(
            AliasRecord(
                ticker=row["ticker"],
                alias=alias,
                alias_normalized=row.get("alias_normalized") or normalize_for_matching(alias),
                alias_type=row.get("alias_type"),
                weight=float(row.get("weight") or 1.0),
            )
        )

    return records