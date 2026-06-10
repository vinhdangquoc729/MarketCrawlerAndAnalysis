"""Local in-process inference using a finetuned PhoBERT model.

Drop-in replacement for SentimentModelClient — same predict_batch interface,
no HTTP server required.

Requires: torch, transformers  (pip install torch transformers)
"""
from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.inference.model_client import ModelPrediction

_LABEL_NORMALIZE: dict[str, str] = {
    "NEG": "negative", "neg": "negative", "negative": "negative",
    "POS": "positive", "pos": "positive", "positive": "positive",
    "NEU": "neutral",  "neu": "neutral",  "neutral":  "neutral",
}


class LocalSentimentModel:
    """Run finetuned PhoBERT sentiment inference in-process (CPU or CUDA).

    Usage:
        model = LocalSentimentModel(model_path="path/to/finetuned_model")
        predictions = model.predict_batch(items)   # same as SentimentModelClient
    """

    def __init__(
        self,
        model_path: str | None = None,
        max_len: int = 256,
        device: str | None = None,
        inference_batch_size: int = 32,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch and transformers are required for local inference. "
                "pip install torch transformers"
            )

        self.model_path = (
            model_path or os.getenv("SENTIMENT_MODEL_PATH", "")
        )
        if not self.model_path:
            raise ValueError(
                "model_path is required. Pass --local-model-path or set "
                "SENTIMENT_MODEL_PATH in .env"
            )

        self.max_len = max_len
        self.inference_batch_size = inference_batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info(
            "Loading local sentiment model path=%s device=%s",
            self.model_path, self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()

        # Build index → standard label map from model config
        id2label: dict = self.model.config.id2label  # e.g. {0: "NEG", 1: "POS", 2: "NEU"}
        self._id2label = {
            int(idx): _LABEL_NORMALIZE.get(lbl, lbl.lower())
            for idx, lbl in id2label.items()
        }
        self._neg_idx = self._find_idx("negative")
        self._pos_idx = self._find_idx("positive")
        self._neu_idx = self._find_idx("neutral")
        logger.info("Local model ready. label_map=%s", self._id2label)

    def _find_idx(self, standard_label: str) -> int:
        for idx, lbl in self._id2label.items():
            if lbl == standard_label:
                return idx
        return 0

    def predict_batch(self, items: list[dict[str, Any]]) -> list[ModelPrediction]:
        """Predict sentiment for a list of entity_aspect items.

        Each item must have: entity_aspect_id, ticker, aspect, model_input.
        Returns the same list[ModelPrediction] as SentimentModelClient.
        """
        if not items:
            return []

        texts = [str(item.get("model_input") or "") for item in items]
        all_probs: list[list[float]] = []

        for start in range(0, len(texts), self.inference_batch_size):
            chunk = texts[start : start + self.inference_batch_size]
            enc = self.tokenizer(
                chunk,
                max_length=self.max_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits

            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            all_probs.extend(probs)

        predictions: list[ModelPrediction] = []
        for item, probs in zip(items, all_probs):
            best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
            label = self._id2label.get(best_idx, "neutral")
            p_neg = float(probs[self._neg_idx]) if self._neg_idx < len(probs) else 0.0
            p_pos = float(probs[self._pos_idx]) if self._pos_idx < len(probs) else 0.0
            p_neu = float(probs[self._neu_idx]) if self._neu_idx < len(probs) else 0.0

            predictions.append(
                ModelPrediction(
                    entity_aspect_id=int(item["entity_aspect_id"]),
                    ticker=str(item.get("ticker", "")),
                    aspect=str(item.get("aspect", "")),
                    sentiment_label=label,
                    sentiment_score=float(p_pos - p_neg),
                    confidence=float(max(probs)),
                    prob_negative=p_neg,
                    prob_neutral=p_neu,
                    prob_positive=p_pos,
                    raw_response={},
                )
            )

        return predictions
