"""
Build GPT-4o labeled training data using this repo's NLP pipeline.

Differences from stock_news prepare_hf_content_labels.py:
  - Text unit is a CONTEXT WINDOW (sentence where company is mentioned ±1
    sentence), not the first 2 sentences of the article.
  - Multiple training samples per article (one per entity mention).
  - model_input is formatted as [STOCK] ticker [/STOCK] [TEXT] context [/TEXT]
    to match what the model will see during inference after retraining.
  - Aliases are loaded from the DB (ticker_aliases table) if available;
    falls back to ticker-only matching when the DB is unreachable.

Usage:
  python -m src.label.prepare_labels --per-company 30 --vn30-only
  python -m src.label.prepare_labels --per-company 40
  python -m src.label.prepare_labels --limit 200 --output data/labels/my_labels.csv

Env:
  OPENAI_API_KEY   required for GPT-4o calls

Output CSV columns:
  sample_id, row_id, context_idx, company, context, model_input,
  gpt4o_label, gpt4o_pos, gpt4o_neu, gpt4o_neg, gpt4o_score
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DATASET_ID = "duongnghia222/vietnam_finance_news_company_tagged"
SPLIT = "train"
VN30_TICKERS = {
    "ACB", "BID", "CTG", "DGC", "FPT", "GAS", "GVR", "HDB", "HPG", "LPB",
    "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSB", "SSI", "STB", "TCB",
    "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VPL", "VRE",
}

SPECIAL_TOKENS = ["[STOCK]", "[/STOCK]", "[TEXT]", "[/TEXT]"]

CSV_FIELDS = [
    "sample_id",
    "row_id",
    "context_idx",
    "company",
    "context",
    "model_input",
    "gpt4o_label",
    "gpt4o_pos",
    "gpt4o_neu",
    "gpt4o_neg",
    "gpt4o_score",
]

GPT_SYSTEM = """You are a Vietnamese stock market analyst labeling finance news by its likely IMPACT ON THE MENTIONED COMPANY'S STOCK PRICE and INVESTOR SENTIMENT, not by the linguistic tone.

Labels:
- POS: likely good for the company, shareholders, or stock price.
- NEG: likely bad for the company, shareholders, or stock price.
- NEU: routine, mixed, unclear, or no clear company-specific stock-price impact.

Use the company ticker/name when provided. If the article discusses a broad market trend and the company-specific impact is unclear, choose NEU.

Return ONLY valid JSON, no markdown:
{"label": "POS"|"NEU"|"NEG", "POS": float, "NEU": float, "NEG": float}
Rules: POS + NEU + NEG must sum to 1.0, and label must match the highest score."""


def build_model_input(ticker: str, context: str) -> str:
    return f"[STOCK] {ticker} [/STOCK] [TEXT] {context} [/TEXT]"


def _load_all_aliases():
    """Load all ticker aliases from the DB, or return empty list on failure."""
    from src.nlp.entity_linker import AliasRecord, make_alias_records
    from src.preprocessing.text_cleaner import normalize_for_matching
    try:
        from src.storage.db import fetch_ticker_aliases
        rows = fetch_ticker_aliases()
        return make_alias_records(rows)
    except Exception:
        return []


def _fallback_alias(company: str):
    """Minimal alias list when DB is unavailable: just the ticker string."""
    from src.nlp.entity_linker import AliasRecord
    from src.preprocessing.text_cleaner import normalize_for_matching
    return [
        AliasRecord(
            ticker=company,
            alias=company,
            alias_normalized=normalize_for_matching(company),
            alias_type="ticker",
            weight=1.0,
        )
    ]


def extract_contexts(company: str, content: str, all_aliases, window: int = 1) -> list[str]:
    """Split content into sentences, find company mentions, return context windows."""
    from src.nlp.sentence_splitter import split_sentences
    from src.nlp.entity_linker import find_entities_in_sentences
    from src.nlp.context_builder import build_entity_contexts

    sentences = split_sentences(content)
    if not sentences:
        return []

    aliases = [a for a in all_aliases if a.ticker == company] if all_aliases else []
    if not aliases:
        aliases = _fallback_alias(company)

    mentions = find_entities_in_sentences(
        sentences=sentences,
        aliases=aliases,
        restrict_tickers={company},
    )

    if not mentions:
        return []

    contexts = build_entity_contexts(
        article_id="",
        sentences=sentences,
        mentions=mentions,
        window=window,
    )
    return [ctx.context for ctx in contexts]


def row_get(row: dict[str, Any], *names: str) -> str:
    lowered = {k.lower(): k for k in row.keys()}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            value = row.get(key)
            return "" if value is None else str(value)
    return ""


def load_dataset_rows() -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. pip install datasets") from exc
    dataset = load_dataset(DATASET_ID, split=SPLIT)
    return [dict(row) for row in dataset]


def sample_per_company(
    rows: list[dict[str, Any]],
    per_company: int,
    seed: int,
    allowed_companies: set[str] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for row_id, row in enumerate(rows):
        company = row_get(row, "Company", "Ticker", "Symbol", "Label", "company").strip().upper()
        if not company:
            continue
        if allowed_companies is not None and company not in allowed_companies:
            continue
        grouped.setdefault(company, []).append((row_id, row))

    rng = random.Random(seed)
    subset: list[tuple[int, dict[str, Any]]] = []
    for company in sorted(grouped):
        items = grouped[company]
        if len(items) > per_company:
            items = rng.sample(items, per_company)
        subset.extend(items)
    return sorted(subset, key=lambda x: x[0])


def read_done_samples(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {
            row["sample_id"]
            for row in csv.DictReader(f)
            if row.get("sample_id") and row.get("gpt4o_label") not in ("ERR", "")
        }


def open_writer(path: Path):
    exists = path.exists() and path.stat().st_size > 0
    f = path.open("a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if not exists:
        writer.writeheader()
        f.flush()
    return f, writer


def normalize_label(data: dict[str, Any]) -> dict[str, Any]:
    scores = {
        "POS": float(data.get("POS", 0.0)),
        "NEU": float(data.get("NEU", 0.0)),
        "NEG": float(data.get("NEG", 0.0)),
    }
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    label = str(data.get("label") or max(scores, key=scores.get)).upper()
    if label not in scores:
        label = max(scores, key=scores.get)
    return {
        "label": label,
        "pos": round(scores["POS"], 4),
        "neu": round(scores["NEU"], 4),
        "neg": round(scores["NEG"], 4),
        "score": round(scores["POS"] - scores["NEG"], 4),
    }


def gpt4o_label(client, context: str, company: str, retries: int = 3) -> dict[str, Any]:
    from openai import OpenAI
    user = f"Company: {company or 'unknown'}\nNews excerpt:\n{context}"
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": GPT_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=80,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            return normalize_label(json.loads(raw))
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"GPT-4o failed: {exc}", file=sys.stderr)
            return {"label": "ERR", "pos": 0.0, "neu": 0.0, "neg": 0.0, "score": 0.0}


def run(
    output_path: str,
    start: int = 0,
    end: int | None = None,
    limit: int | None = None,
    per_company: int | None = None,
    vn30_only: bool = False,
    window: int = 1,
    seed: int = 42,
    delay: float = 0.5,
) -> None:
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("OPENAI_API_KEY is required.")

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    rows = load_dataset_rows()
    if per_company is not None:
        allowed = VN30_TICKERS if vn30_only else None
        subset = sample_per_company(rows, per_company, seed, allowed)
    else:
        if limit is not None:
            end = start + limit
        subset = list(enumerate(rows[start:end], start=start))

    print(f"Loaded {len(rows)} rows from {DATASET_ID}/{SPLIT}")
    print(f"Sampled {len(subset)} articles to process")

    print("Loading ticker aliases from DB (falls back to ticker-only if unavailable)...")
    all_aliases = _load_all_aliases()
    print(f"  aliases loaded: {len(all_aliases)}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = read_done_samples(output)
    out_file, writer = open_writer(output)

    print(f"Skipping {len(done)} already-labeled samples")

    written = 0
    try:
        for row_id, row in subset:
            company = row_get(row, "Company", "Ticker", "Symbol", "Label", "company").strip().upper()
            if not company:
                continue
            content = row_get(row, "Content")
            if not content:
                continue

            contexts = extract_contexts(company, content, all_aliases, window=window)

            if not contexts:
                print(f"  row={row_id} company={company}: no entity mentions found, skipping")
                continue

            # Deduplicate overlapping context windows by text
            seen_contexts: set[str] = set()
            unique_contexts = []
            for ctx in contexts:
                if ctx not in seen_contexts:
                    seen_contexts.add(ctx)
                    unique_contexts.append(ctx)
            contexts = unique_contexts

            for ctx_idx, context in enumerate(contexts):
                sample_id = f"{row_id}_{ctx_idx}"
                if sample_id in done:
                    continue

                model_input = build_model_input(company, context)
                gp = gpt4o_label(client, context, company)

                writer.writerow({
                    "sample_id": sample_id,
                    "row_id": row_id,
                    "context_idx": ctx_idx,
                    "company": company,
                    "context": context,
                    "model_input": model_input,
                    "gpt4o_label": gp["label"],
                    "gpt4o_pos": gp["pos"],
                    "gpt4o_neu": gp["neu"],
                    "gpt4o_neg": gp["neg"],
                    "gpt4o_score": gp["score"],
                })
                out_file.flush()
                written += 1
                print(f"[{written}] row={row_id} ctx={ctx_idx} company={company} label={gp['label']}")
                time.sleep(delay)
    finally:
        out_file.close()

    print(f"\nSaved {written} new samples to {output}")
    print(f"Total samples in file: {len(done) + written}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build GPT-4o labeled training data using context windows")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int)
    group.add_argument("--end", type=int)
    group.add_argument("--per-company", type=int, help="Sample up to N articles per company")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", default="data/labels/context_labels.csv")
    parser.add_argument("--vn30-only", action="store_true")
    parser.add_argument("--window", type=int, default=1, help="Sentence window around each mention (default 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    run(
        output_path=args.output,
        start=args.start,
        end=args.end,
        limit=args.limit,
        per_company=args.per_company,
        vn30_only=args.vn30_only,
        window=args.window,
        seed=args.seed,
        delay=args.delay,
    )
