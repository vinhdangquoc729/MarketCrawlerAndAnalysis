"""Build ticker-specific context windows."""
from __future__ import annotations

from dataclasses import dataclass

from src.nlp.entity_linker import EntityMention
from src.preprocessing.text_cleaner import clean_text


@dataclass(frozen=True)
class EntityContext:
    article_id: str
    ticker: str
    entity_text: str
    alias_type: str | None
    alias_weight: float
    sentence_index: int
    context: str
    relevance_weight: float


def build_context_for_mention(
    sentences: list[str],
    mention: EntityMention,
    window: int = 1,
) -> str:
    """Build context around a mention using previous/current/next sentence."""
    start = max(0, mention.sentence_index - window)
    end = min(len(sentences), mention.sentence_index + window + 1)

    context = " ".join(sentences[start:end])
    return clean_text(context)


def compute_relevance_weight(alias_weight: float, context: str) -> float:
    """Compute relevance weight for entity context.

    Direct ticker/company aliases should have higher weight.
    Related entities such as leaders/brands can have lower weight.
    """
    base = float(alias_weight or 1.0)

    context_len = len(context)

    if context_len < 80:
        base *= 0.7

    return round(min(max(base, 0.1), 1.0), 3)


def build_entity_contexts(
    article_id: str,
    sentences: list[str],
    mentions: list[EntityMention],
    window: int = 1,
) -> list[EntityContext]:
    """Create entity-level contexts for an article."""
    contexts: list[EntityContext] = []
    seen: set[tuple[str, str, int]] = set()

    for mention in mentions:
        context = build_context_for_mention(sentences, mention, window=window)

        if not context:
            continue

        key = (
            mention.ticker,
            mention.entity_text.lower(),
            mention.sentence_index,
        )

        if key in seen:
            continue

        seen.add(key)

        relevance_weight = compute_relevance_weight(
            alias_weight=mention.alias_weight,
            context=context,
        )

        contexts.append(
            EntityContext(
                article_id=article_id,
                ticker=mention.ticker,
                entity_text=mention.entity_text,
                alias_type=mention.alias_type,
                alias_weight=mention.alias_weight,
                sentence_index=mention.sentence_index,
                context=context,
                relevance_weight=relevance_weight,
            )
        )

    return contexts