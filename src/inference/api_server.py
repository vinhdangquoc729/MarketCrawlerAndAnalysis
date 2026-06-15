"""FastAPI service for serving the local finetuned sentiment model.

Run:
    uvicorn src.inference.api_server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.local_model import LocalSentimentModel

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(title="Market Sentiment Model API", version="1.0.0")
_model: LocalSentimentModel | None = None


class PredictItem(BaseModel):
    entity_aspect_id: int
    ticker: str = ""
    aspect: str = ""
    model_input: str


class PredictBatchRequest(BaseModel):
    items: list[PredictItem] = Field(default_factory=list)


def _model_dump(item: BaseModel) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return item.dict()


def get_model() -> LocalSentimentModel:
    """Return the singleton model instance, loading it once per API process."""
    global _model
    if _model is None:
        model_path = os.getenv("SENTIMENT_MODEL_PATH") or ""
        if not model_path:
            raise RuntimeError("SENTIMENT_MODEL_PATH is required for the model API.")
        logger.info("Loading sentiment model for API from %s", model_path)
        _model = LocalSentimentModel(model_path=model_path)
    return _model


@app.on_event("startup")
def startup() -> None:
    get_model()


@app.get("/health")
def health() -> dict[str, Any]:
    model = get_model()
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": model.model_path,
        "device": str(model.device),
    }


@app.post("/predict_batch")
def predict_batch(payload: PredictBatchRequest) -> dict[str, Any]:
    if not payload.items:
        return {"results": []}

    try:
        items = [_model_dump(item) for item in payload.items]
        predictions = get_model().predict_batch(items)
    except Exception as exc:
        logger.exception("predict_batch failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results = []
    for pred in predictions:
        results.append(
            {
                "entity_aspect_id": pred.entity_aspect_id,
                "ticker": pred.ticker,
                "aspect": pred.aspect,
                "sentiment_label": pred.sentiment_label,
                "sentiment_score": pred.sentiment_score,
                "confidence": pred.confidence,
                "probabilities": {
                    "negative": pred.prob_negative,
                    "neutral": pred.prob_neutral,
                    "positive": pred.prob_positive,
                },
                "prob_negative": pred.prob_negative,
                "prob_neutral": pred.prob_neutral,
                "prob_positive": pred.prob_positive,
            }
        )

    return {"results": results}
