"""HTTP client for calling fine-tuned sentiment FastAPI model."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ModelPrediction:
    entity_aspect_id: int
    ticker: str
    aspect: str
    sentiment_label: str
    sentiment_score: float
    confidence: float
    prob_negative: float
    prob_neutral: float
    prob_positive: float
    raw_response: dict[str, Any]


class SentimentModelClient:
    """Client for sentiment FastAPI service."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("SENTIMENT_MODEL_API_URL") or "http://localhost:8000"
        ).rstrip("/")

        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else int(os.getenv("MODEL_INFERENCE_TIMEOUT_SECONDS") or "60")
        )

    def predict_batch(self, items: list[dict[str, Any]]) -> list[ModelPrediction]:
        """Call model /predict_batch endpoint."""
        if not items:
            return []

        payload = {"items": items}

        response = requests.post(
            f"{self.base_url}/predict_batch",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        data = response.json()
        results = data.get("results", data)

        predictions: list[ModelPrediction] = []

        for result in results:
            probs = result.get("probabilities") or {}

            predictions.append(
                ModelPrediction(
                    entity_aspect_id=int(result["entity_aspect_id"]),
                    ticker=str(result["ticker"]),
                    aspect=str(result["aspect"]),
                    sentiment_label=str(result["sentiment_label"]),
                    sentiment_score=float(result["sentiment_score"]),
                    confidence=float(result.get("confidence", 0.0)),
                    prob_negative=float(
                        probs.get("negative", result.get("prob_negative", 0.0))
                    ),
                    prob_neutral=float(
                        probs.get("neutral", result.get("prob_neutral", 0.0))
                    ),
                    prob_positive=float(
                        probs.get("positive", result.get("prob_positive", 0.0))
                    ),
                    raw_response=result,
                )
            )

        return predictions