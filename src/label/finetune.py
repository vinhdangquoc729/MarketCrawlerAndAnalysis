"""
Finetune PhoBERT's classifier head plus the last transformer encoder layer
on labeled context-window data from prepare_labels.py.

Key difference from stock_news finetune_last_layer.py:
  - Adds [STOCK], [/STOCK], [TEXT], [/TEXT] as special tokens to the tokenizer
    so they get dedicated embeddings instead of being split into subwords.
  - Reads model_input column (structured format) instead of text/title.

Usage:
  python -m src.label.finetune --csv data/labels/context_labels.csv
  python -m src.label.finetune --csv data/labels/context_labels.csv --epochs 15 --lr 1e-5
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"
LABEL2ID = {"NEG": 0, "POS": 1, "NEU": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

SPECIAL_TOKENS = ["[STOCK]", "[/STOCK]", "[TEXT]", "[/TEXT]"]


class TextDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_len: int = 256):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(LABEL2ID[rec["label"]], dtype=torch.long),
        }


def load_csv(path: str) -> list[dict]:
    """Load labeled samples from prepare_labels.py output CSV."""
    records = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("gpt4o_label") or "").strip().upper()
            if label not in LABEL2ID:
                continue
            # prefer structured format; fall back to plain text columns for
            # compatibility with stock_news-style CSVs (text / title columns)
            text = (
                row.get("model_input") or row.get("context")
                or row.get("text") or row.get("title") or ""
            ).strip()
            if not text:
                continue
            records.append({"text": text, "label": label})
    return records


def split(records: list[dict], val_ratio: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    data = records[:]
    rng.shuffle(data)
    cut = int(len(data) * (1 - val_ratio))
    return data[:cut], data[cut:]


def find_last_encoder_layer(model):
    roberta = getattr(model, "roberta", None)
    if roberta is None or not hasattr(roberta, "encoder"):
        raise RuntimeError("Could not find model.roberta.encoder.layer")
    layers = getattr(roberta.encoder, "layer", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("Could not find any encoder layers")
    return layers[-1]


def freeze_except_last_layer_and_head(model) -> tuple[int, int, list[str]]:
    for param in model.parameters():
        param.requires_grad = False

    trainable_parts = []
    last_layer = find_last_encoder_layer(model)
    for param in last_layer.parameters():
        param.requires_grad = True
    trainable_parts.append("roberta.encoder.layer[-1]")

    for param in model.classifier.parameters():
        param.requires_grad = True
    trainable_parts.append("classifier")

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return frozen, trainable, trainable_parts


def make_optimizer(model, lr: float, head_lr: float | None):
    if head_lr is None:
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr,
        )
    last_layer_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if name.startswith("classifier") else last_layer_params).append(param)
    return torch.optim.AdamW([
        {"params": last_layer_params, "lr": lr},
        {"params": head_params, "lr": head_lr},
    ])


def train(
    csv_path: str,
    epochs: int = 15,
    lr: float = 1e-5,
    head_lr: float | None = 1e-4,
    batch_size: int = 8,
    val_ratio: float = 0.2,
    max_len: int = 256,
    seed: int = 42,
    save_dir: str = "models/finetuned_context",
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    records = load_csv(csv_path)
    if not records:
        raise SystemExit(f"No valid labeled rows found in {csv_path}")
    print(f"Loaded {len(records)} labeled samples from {csv_path}")
    print(f"Label distribution: {dict(Counter(r['label'] for r in records))}")

    train_recs, val_recs = split(records, val_ratio, seed)
    print(f"Train: {len(train_recs)}  Val: {len(val_recs)}")

    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Add special tokens so they get dedicated embeddings
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    print(f"Added {added} special tokens: {SPECIAL_TOKENS}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
    )
    # Resize embedding matrix to cover the new tokens
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    frozen, trainable, trainable_parts = freeze_except_last_layer_and_head(model)
    total = frozen + trainable
    print(f"Frozen params   : {frozen:,}")
    print(f"Trainable params: {trainable:,}  ({trainable / total:.2%} of total)")
    print(f"Trainable parts : {', '.join(trainable_parts)}")

    train_ds = TextDataset(train_recs, tokenizer, max_len=max_len)
    val_ds = TextDataset(val_recs, tokenizer, max_len=max_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    label_counts = Counter(r["label"] for r in train_recs)
    n_total = len(train_recs)
    n_classes = len(LABEL2ID)
    weights = torch.zeros(n_classes, dtype=torch.float32)
    for label, idx in LABEL2ID.items():
        count = label_counts.get(label, 1)
        weights[idx] = n_total / (n_classes * count ** 0.5)
    weights = weights.to(device)
    print("\nClass weights: " + "  ".join(f"{k}={weights[v]:.3f}" for k, v in LABEL2ID.items()))

    optimizer = make_optimizer(model, lr, head_lr)
    criterion = nn.CrossEntropyLoss(weight=weights)

    lr_msg = f"last-layer lr={lr}"
    if head_lr is not None:
        lr_msg += f", head lr={head_lr}"
    print(f"\nTraining for {epochs} epochs  ({lr_msg}, batch={batch_size})\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val Acc':>7}  [per-class POS/NEU/NEG]")
    print("-" * 70)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids=ids, attention_mask=mask).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
            )
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_dl)

        model.eval()
        val_loss, correct, total_seen = 0.0, 0, 0
        per_class_correct: Counter = Counter()
        per_class_total: Counter = Counter()
        with torch.no_grad():
            for batch in val_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                val_loss += criterion(logits, labels).item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total_seen += labels.size(0)
                for p, l in zip(preds.tolist(), labels.tolist()):
                    per_class_total[ID2LABEL[l]] += 1
                    if p == l:
                        per_class_correct[ID2LABEL[l]] += 1

        val_loss /= len(val_dl)
        val_acc = correct / total_seen if total_seen else 0.0
        per_class = "  ".join(
            f"{lbl}={per_class_correct[lbl]}/{per_class_total[lbl]}"
            for lbl in ["POS", "NEU", "NEG"]
        )
        marker = "  <-- best" if val_acc > best_val_acc else ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>8.4f}  {val_acc:>7.1%}  [{per_class}]{marker}")

    print(f"\nBest val accuracy : {best_val_acc:.1%}")
    print(f"Model saved to    : {save_dir}/")

    print("\nSanity check (first 5 val examples):")
    model.eval()
    for rec in val_recs[:5]:
        enc = tokenizer(rec["text"], return_tensors="pt", truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = ID2LABEL[logits.argmax().item()]
        print(
            f"  gold={rec['label']}  pred={pred}  "
            f"NEG={probs[0]:.2f} POS={probs[1]:.2f} NEU={probs[2]:.2f}"
        )
        sys.stdout.buffer.write(f"  {rec['text'][:100]}\n".encode("utf-8", errors="replace"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune PhoBERT on context-window labels")
    parser.add_argument("--csv", required=True, help="CSV from prepare_labels.py")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--same-lr", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="models/finetuned_context")
    args = parser.parse_args()

    train(
        csv_path=args.csv,
        epochs=args.epochs,
        lr=args.lr,
        head_lr=None if args.same_lr else args.head_lr,
        batch_size=args.batch,
        max_len=args.max_len,
        seed=args.seed,
        save_dir=args.save,
    )
