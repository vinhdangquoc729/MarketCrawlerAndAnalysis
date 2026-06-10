"""Build model input strings for sentiment inference."""
from __future__ import annotations

from src.preprocessing.text_cleaner import clean_text


def build_model_input(
    ticker: str,
    context: str,
    aspect: str | None = None,
) -> str:
    """Build structured model input matching the finetuned model's training format.

    Format: [STOCK] ticker [/STOCK] [TEXT] context [/TEXT]

    aspect is accepted for call-site compatibility but not embedded — the
    model was trained without aspect conditioning.
    """
    return f"[STOCK] {clean_text(ticker)} [/STOCK] [TEXT] {clean_text(context)} [/TEXT]"