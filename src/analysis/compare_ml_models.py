"""
Compare ML models for predicting next-day stock return direction.

For each row t, the target is whether log_return at t+horizon is positive.
Features are rolling summaries over the previous N trading days, excluding the
target day. Supports windows such as 10 days, 1 month (21), 3 months (63).

Each window is evaluated with and without sentiment features to measure the
information added by sentiment.

Usage:
  python -m src.analysis.compare_ml_models
  python -m src.analysis.compare_ml_models --news-days-only --output data/results/ml_news_days.csv
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis._panel_loader import load_panel

# "volatility" is aliased from "volatility_5d" at load time via _panel_loader
BASE_FEATURES = [
    "log_return",
    "volume_growth",
    "volatility",
    "clv",
    "sentiment_score",
    "news_count",
]
SENTIMENT_FEATURE_PREFIXES = ("sentiment_score", "news_count")

_REQUIRED = ["ticker", "date", "log_return"] + BASE_FEATURES
_NUMERIC = BASE_FEATURES

_DEFAULT_PANEL = os.path.join(
    os.getenv("MARKET_VALIDATION_OUTPUT_DIR", "data/processed/validation"),
    "daily_panel.csv",
)


def rolling_feature_frame(group: pd.DataFrame, window: int, horizon: int) -> pd.DataFrame:
    group = group.sort_values("date").reset_index(drop=True).copy()
    shifted = group[BASE_FEATURES].shift(1)

    features = {
        "ticker": group["ticker"],
        "date": group["date"],
        "target_return": group["log_return"].shift(-horizon),
    }
    for col in BASE_FEATURES:
        roll = shifted[col].rolling(window, min_periods=max(3, window // 3))
        features[f"{col}_mean_{window}"] = roll.mean()
        features[f"{col}_std_{window}"] = roll.std()
        features[f"{col}_last"] = shifted[col]

    out = pd.DataFrame(features)
    out["target_up"] = (out["target_return"] > 0).astype(int)
    return out


def build_dataset(panel: pd.DataFrame, window: int, horizon: int) -> pd.DataFrame:
    frames = [rolling_feature_frame(group, window, horizon) for _, group in panel.groupby("ticker")]
    data = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in data.columns if c not in ["ticker", "date", "target_return", "target_up"]]
    return data.dropna(subset=feature_cols + ["target_return"]).reset_index(drop=True)


def get_models(random_state: int):
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        ),
        "ridge": make_pipeline(StandardScaler(), RidgeClassifier(class_weight="balanced")),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=8,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=2, random_state=random_state,
        ),
    }


def predict_scores(model, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    pred = model.predict(x_test)
    score = None
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(x_test)
    return pred, score


def evaluate(y_true: np.ndarray, pred: np.ndarray, score: np.ndarray | None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": np.nan,
        "positive_rate": float(np.mean(y_true)),
        "pred_positive_rate": float(np.mean(pred)),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        out["auc"] = roc_auc_score(y_true, score)
    return out


def time_split(data: pd.DataFrame, train_ratio: float, horizon: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(data["date"].dropna().unique()))
    if len(dates) < 5:
        return data.iloc[:0], data.iloc[:0]
    cut_idx = int(len(dates) * train_ratio)
    cut = dates[cut_idx]
    buffer_idx = max(0, cut_idx - horizon)
    buffer_cut = dates[buffer_idx]
    return data[data["date"] < buffer_cut], data[data["date"] >= cut]


def run_scope(
    scope: str,
    feature_set: str,
    window: int,
    feature_cols: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    random_state: int,
    all_results: list[dict],
    all_predictions: list[dict],
) -> None:
    x_train = train[feature_cols]
    y_train = train["target_up"].to_numpy()
    x_test = test[feature_cols]
    y_test = test["target_up"].to_numpy()

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return

    for model_name, model in get_models(random_state).items():
        model.fit(x_train, y_train)
        pred, score = predict_scores(model, x_test)
        metrics = evaluate(y_test, pred, score)
        all_results.append({
            "scope": scope, "feature_set": feature_set, "window": window, "model": model_name,
            "n_features": len(feature_cols), "n_train": len(train), "n_test": len(test),
            "date_train_min": train["date"].min().date(), "date_train_max": train["date"].max().date(),
            "date_test_min": test["date"].min().date(), "date_test_max": test["date"].max().date(),
            **metrics,
        })
        for row, row_pred, row_score in zip(
            test.itertuples(index=False), pred,
            score if score is not None else [math.nan] * len(pred),
        ):
            all_predictions.append({
                "scope": scope, "feature_set": feature_set, "window": window, "model": model_name,
                "ticker": row.ticker, "date": row.date,
                "target_return": row.target_return, "target_up": row.target_up,
                "pred_up": int(row_pred), "score": row_score,
            })


def build_sentiment_delta(results: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["scope", "window", "model"]
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc", "pred_positive_rate"]
    with_s = results[results["feature_set"] == "with_sentiment"].copy()
    no_s = results[results["feature_set"] == "no_sentiment"].copy()
    merged = with_s.merge(no_s, on=key_cols, suffixes=("_with_sentiment", "_no_sentiment"))
    for metric in metric_cols:
        merged[f"{metric}_delta"] = merged[f"{metric}_with_sentiment"] - merged[f"{metric}_no_sentiment"]
    keep = key_cols + ["n_train_with_sentiment", "n_test_with_sentiment"]
    for metric in metric_cols:
        keep.extend([f"{metric}_with_sentiment", f"{metric}_no_sentiment", f"{metric}_delta"])
    return merged[keep].rename(columns={
        "n_train_with_sentiment": "n_train",
        "n_test_with_sentiment": "n_test",
    })


def run(
    panel_path: str,
    output_path: str,
    predictions_output: str,
    windows: list[int],
    horizon: int,
    train_ratio: float,
    min_train: int,
    min_test: int,
    fill_missing_sentiment: float | None,
    random_state: int,
    news_days_only: bool,
) -> None:
    panel = load_panel(
        panel_path,
        required_cols=_REQUIRED,
        numeric_cols=_NUMERIC,
        fill_missing_sentiment=fill_missing_sentiment,
        news_days_only=news_days_only,
    )
    all_results: list[dict] = []
    all_predictions: list[dict] = []

    for window in windows:
        data = build_dataset(panel, window, horizon)
        feature_cols_all = [c for c in data.columns if c not in ["ticker", "date", "target_return", "target_up"]]
        feature_sets = {
            "with_sentiment": feature_cols_all,
            "no_sentiment": [c for c in feature_cols_all if not c.startswith(SENTIMENT_FEATURE_PREFIXES)],
        }
        train_all, test_all = time_split(data, train_ratio, horizon)
        if len(train_all) >= min_train and len(test_all) >= min_test:
            for fs_name, fs_cols in feature_sets.items():
                run_scope("ALL", fs_name, window, fs_cols, train_all, test_all, random_state,
                          all_results, all_predictions)

        for ticker, ticker_data in data.groupby("ticker"):
            train, test = time_split(ticker_data, train_ratio, horizon)
            if len(train) < min_train or len(test) < min_test:
                continue
            for fs_name, fs_cols in feature_sets.items():
                run_scope(ticker, fs_name, window, fs_cols, train, test, random_state,
                          all_results, all_predictions)

    results = pd.DataFrame(all_results).sort_values(["scope", "window", "f1"], ascending=[True, True, False])
    predictions = pd.DataFrame(all_predictions)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_output, index=False, encoding="utf-8-sig")

    print(f"Saved metrics to {output_path}")
    print(f"Saved predictions to {predictions_output}")
    print("\nTop overall rows by F1:")
    cols = ["scope", "feature_set", "window", "model", "n_train", "n_test", "accuracy", "f1", "auc", "positive_rate"]
    print(results.sort_values("f1", ascending=False).head(20)[cols].to_string(index=False))

    comparison = build_sentiment_delta(results)
    delta_output = output_path.replace(".csv", "_sentiment_delta.csv")
    comparison.to_csv(delta_output, index=False, encoding="utf-8-sig")
    print(f"Saved sentiment deltas to {delta_output}")
    print("\nTop sentiment improvements by F1 delta:")
    delta_cols = ["scope", "window", "model", "f1_delta", "accuracy_delta", "auc_delta",
                  "f1_with_sentiment", "f1_no_sentiment"]
    print(comparison.sort_values("f1_delta", ascending=False).head(20)[delta_cols].to_string(index=False))


def _parse_fill(value: str) -> float | None:
    if value.lower() == "nan":
        return None
    return float(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ML models for next-day return direction")
    parser.add_argument("--panel", default=_DEFAULT_PANEL)
    parser.add_argument("--output", default="data/results/ml_model_comparison.csv")
    parser.add_argument("--predictions-output", default="data/results/ml_model_predictions.csv")
    parser.add_argument("--windows", type=int, nargs="+", default=[10, 21, 63])
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--min-train", type=int, default=100)
    parser.add_argument("--min-test", type=int, default=40)
    parser.add_argument("--fill-missing-sentiment", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--news-days-only", action="store_true")
    args = parser.parse_args()

    run(
        panel_path=args.panel,
        output_path=args.output,
        predictions_output=args.predictions_output,
        windows=args.windows,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        min_train=args.min_train,
        min_test=args.min_test,
        fill_missing_sentiment=_parse_fill(args.fill_missing_sentiment),
        random_state=args.seed,
        news_days_only=args.news_days_only,
    )
