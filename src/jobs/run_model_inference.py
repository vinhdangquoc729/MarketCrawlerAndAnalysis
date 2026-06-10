"""Run sentiment model inference for entity_aspects."""
from __future__ import annotations

import argparse
import logging
import os
import time

import requests
from dotenv import load_dotenv

from src.inference.model_client import SentimentModelClient
from src.inference.local_model import LocalSentimentModel
from src.storage.db import (
    fetch_entity_aspects_for_inference,
    upsert_entity_sentiments,
)
from src.utils.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)


LABEL_NORMALIZE_MAP = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "tiêu cực": "negative",
    "tieu cuc": "negative",
    "trung tính": "neutral",
    "trung tinh": "neutral",
    "tích cực": "positive",
    "tich cuc": "positive",
}


def normalize_sentiment_label(label: str | None) -> str:
    """Normalize API sentiment label to negative/neutral/positive."""
    if not label:
        return "neutral"

    cleaned = str(label).strip().lower()
    return LABEL_NORMALIZE_MAP.get(cleaned, cleaned)


def validate_prediction_ids(
    expected_ids: set[int],
    returned_ids: set[int],
) -> None:
    """Warn when API response does not match requested IDs."""
    missing_ids = expected_ids - returned_ids
    extra_ids = returned_ids - expected_ids

    if missing_ids:
        logger.warning(
            "Model API did not return predictions for entity_aspect_ids=%s",
            sorted(missing_ids)[:20],
        )

    if extra_ids:
        logger.warning(
            "Model API returned unexpected entity_aspect_ids=%s",
            sorted(extra_ids)[:20],
        )


def run_inference(
    batch_size: int = 64,
    max_batches: int | None = None,
    model_version: str = "finetuned-v1",
    retry_attempts: int = 3,
    retry_sleep_seconds: float = 2.0,
    local_model_path: str | None = None,
) -> int:
    """Run batch inference and save predictions."""
    if local_model_path:
        client = LocalSentimentModel(model_path=local_model_path)
    else:
        api_url = os.getenv("SENTIMENT_MODEL_API_URL") or ""
        if not api_url or not api_url.startswith("http"):
            raise RuntimeError(
                "No inference backend configured. "
                "Set SENTIMENT_MODEL_PATH in .env to use the local GPU model, "
                "or set SENTIMENT_MODEL_API_URL to point at a running API server."
            )
        client = SentimentModelClient()

    total_saved = 0
    batch_idx = 0

    while True:
        if max_batches is not None and batch_idx >= max_batches:
            break

        rows = fetch_entity_aspects_for_inference(
            limit=batch_size,
            model_version=model_version,
        )

        if not rows:
            logger.info("No more entity_aspects for inference.")
            break

        request_items = [
            {
                "entity_aspect_id": row["entity_aspect_id"],
                "ticker": row["ticker"],
                "aspect": row["aspect"],
                "model_input": row["model_input"],
            }
            for row in rows
        ]

        logger.info(
            "Calling sentiment model batch=%s size=%s model_version=%s",
            batch_idx + 1,
            len(request_items),
            model_version,
        )

        predictions = None
        last_error: Exception | None = None

        for attempt in range(1, retry_attempts + 1):
            try:
                predictions = client.predict_batch(request_items)
                break

            except requests.exceptions.RequestException as exc:
                last_error = exc
                logger.warning(
                    "Model API request failed batch=%s attempt=%s/%s error=%s",
                    batch_idx + 1,
                    attempt,
                    retry_attempts,
                    exc,
                )
                time.sleep(retry_sleep_seconds * attempt)

            except Exception as exc:
                last_error = exc
                logger.exception(
                    "Unexpected inference error batch=%s attempt=%s/%s",
                    batch_idx + 1,
                    attempt,
                    retry_attempts,
                )
                time.sleep(retry_sleep_seconds * attempt)

        if predictions is None:
            raise RuntimeError(
                f"Model inference failed after {retry_attempts} attempts: {last_error}"
            )

        row_by_id = {
            int(row["entity_aspect_id"]): row
            for row in rows
        }

        expected_ids = set(row_by_id.keys())
        returned_ids = {int(pred.entity_aspect_id) for pred in predictions}

        validate_prediction_ids(expected_ids, returned_ids)

        save_rows: list[dict] = []

        for pred in predictions:
            entity_aspect_id = int(pred.entity_aspect_id)

            if entity_aspect_id not in row_by_id:
                continue

            source_row = row_by_id[entity_aspect_id]
            normalized_label = normalize_sentiment_label(pred.sentiment_label)

            save_rows.append(
                {
                    "entity_aspect_id": entity_aspect_id,
                    "entity_id": source_row["entity_id"],
                    "article_id": source_row["article_id"],
                    "ticker": source_row["ticker"],
                    "aspect": source_row["aspect"],
                    "sentiment_label": normalized_label,
                    "sentiment_score": pred.sentiment_score,
                    "confidence": pred.confidence,
                    "prob_negative": pred.prob_negative,
                    "prob_neutral": pred.prob_neutral,
                    "prob_positive": pred.prob_positive,
                    "inference_status": "success",
                    "error_message": None,
                }
            )

        if not save_rows:
            logger.warning(
                "No valid predictions to save for batch=%s. Stopping to avoid infinite loop.",
                batch_idx + 1,
            )
            break

        saved = upsert_entity_sentiments(
            save_rows,
            model_version=model_version,
        )

        total_saved += saved
        batch_idx += 1

        logger.info(
            "Saved predictions batch=%s saved=%s total_saved=%s",
            batch_idx,
            saved,
            total_saved,
        )

    return total_saved


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("MODEL_INFERENCE_BATCH_SIZE") or "64"),
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--model-version",
        type=str,
        default=os.getenv("SENTIMENT_MODEL_VERSION") or "finetuned-v1",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=int(os.getenv("MODEL_INFERENCE_RETRY_ATTEMPTS", "3")),
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=float(os.getenv("MODEL_INFERENCE_RETRY_SLEEP_SECONDS", "2.0")),
    )
    parser.add_argument(
        "--local-model-path",
        type=str,
        default=os.getenv("SENTIMENT_MODEL_PATH") or None,
        help=(
            "Path to a local finetuned PhoBERT model directory. "
            "When set, runs inference in-process instead of calling the API. "
            "Can also be set via SENTIMENT_MODEL_PATH in .env."
        ),
    )

    args = parser.parse_args()

    total = run_inference(
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        model_version=args.model_version,
        retry_attempts=args.retry_attempts,
        retry_sleep_seconds=args.retry_sleep_seconds,
        local_model_path=args.local_model_path,
    )

    logger.info("Inference complete total_saved=%s", total)


if __name__ == "__main__":
    main()