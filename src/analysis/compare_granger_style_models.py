"""
Compare restricted vs sentiment-augmented predictive models.

Generalises the Granger idea:
  restricted features   = past return + past volume_growth
  unrestricted features = restricted + past sentiment_score

Evaluates both:
  - regression target: next-day log_return
  - classification target: next-day target_up

Models: Linear/Ridge/Lasso regression, Logistic/Ridge classification,
        Random Forest, Gradient Boosting, XGBoost, LightGBM (if installed).

Usage:
  python -m src.analysis.compare_granger_style_models
  python -m src.analysis.compare_granger_style_models --news-days-only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis._panel_loader import load_panel

CONTROL_COLS = ["log_return", "volume_growth"]
SENTIMENT_COLS = ["sentiment_score"]

_REQUIRED = ["ticker", "date", "log_return", "volume_growth", "sentiment_score"]
_NUMERIC = ["log_return", "volume_growth", "sentiment_score"]

_DEFAULT_PANEL = os.path.join(
    os.getenv("MARKET_VALIDATION_OUTPUT_DIR", "data/processed/validation"),
    "daily_panel.csv",
)


def build_lagged_dataset(panel: pd.DataFrame, window: int, horizon: int) -> pd.DataFrame:
    frames = []
    source_cols = CONTROL_COLS + SENTIMENT_COLS
    for ticker, group in panel.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True).copy()
        data = {"ticker": group["ticker"], "date": group["date"],
                "target_return": group["log_return"].shift(-horizon)}
        for col in source_cols:
            for lag in range(1, window + 1):
                data[f"{col}_lag{lag}"] = group[col].shift(lag)
        out = pd.DataFrame(data)
        out["target_up"] = (out["target_return"] > 0).astype(int)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def time_split(data: pd.DataFrame, train_ratio: float, horizon: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(data["date"].dropna().unique()))
    if len(dates) < 5:
        return data.iloc[:0], data.iloc[:0]
    cut_idx = int(len(dates) * train_ratio)
    cut_date = dates[cut_idx]
    buffer_idx = max(0, cut_idx - horizon)
    buffer_date = dates[buffer_idx]
    return data[data["date"] < buffer_date], data[data["date"] >= cut_date]


def regression_models(seed: int, fast: bool = False):
    n_rf = 50 if fast else 300
    n_gb = 30 if fast else 150
    n_boost = 50 if fast else 200
    lasso_iter = 2000 if fast else 10000
    models = {
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.0001, max_iter=lasso_iter)),
        "random_forest": RandomForestRegressor(
            n_estimators=n_rf, max_depth=5, min_samples_leaf=8, random_state=seed, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=n_gb, learning_rate=0.05, max_depth=2, random_state=seed,
        ),
    }
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(
            n_estimators=n_boost, max_depth=3, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, objective="reg:squarederror", random_state=seed, n_jobs=-1,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["lightgbm"] = LGBMRegressor(
            n_estimators=n_boost, max_depth=3, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, random_state=seed, verbose=-1,
        )
    except Exception:
        pass
    return models


def classification_models(seed: int, fast: bool = False):
    n_rf = 50 if fast else 300
    n_gb = 30 if fast else 150
    n_boost = 50 if fast else 200
    lr_iter = 300 if fast else 2000
    models = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=lr_iter, class_weight="balanced", random_state=seed),
        ),
        "ridge_classifier": make_pipeline(StandardScaler(), RidgeClassifier(class_weight="balanced")),
        "random_forest": RandomForestClassifier(
            n_estimators=n_rf, max_depth=5, min_samples_leaf=8,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=n_gb, learning_rate=0.05, max_depth=2, random_state=seed,
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=n_boost, max_depth=3, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, eval_metric="logloss", random_state=seed, n_jobs=-1,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=n_boost, max_depth=3, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, random_state=seed, verbose=-1,
        )
    except Exception:
        pass
    return models


def get_score(model, x: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return None


def eval_regression(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "rmse": mean_squared_error(y_true, pred) ** 0.5,
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred),
    }


def eval_classification(y_true: np.ndarray, pred: np.ndarray, score: np.ndarray | None) -> dict:
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


def run_one_scope(
    scope: str, data: pd.DataFrame, window: int,
    train_ratio: float, min_train: int, min_test: int, seed: int, horizon: int = 1,
    fast: bool = False,
) -> list[dict]:
    train, test = time_split(data, train_ratio, horizon)
    if len(train) < min_train or len(test) < min_test:
        return []

    control_features = [c for c in data.columns if any(c.startswith(f"{b}_lag") for b in CONTROL_COLS)]
    full_features = control_features + [c for c in data.columns if any(c.startswith(f"{b}_lag") for b in SENTIMENT_COLS)]
    feature_sets = {"restricted": control_features, "with_sentiment": full_features}

    rows = []
    for feature_set, cols in feature_sets.items():
        usable_train = train.dropna(subset=cols + ["target_return"])
        usable_test = test.dropna(subset=cols + ["target_return"])
        if len(usable_train) < min_train or len(usable_test) < min_test:
            continue

        x_train, x_test = usable_train[cols], usable_test[cols]
        base = {
            "scope": scope, "window": window, "feature_set": feature_set,
            "n_features": len(cols), "n_train": len(usable_train), "n_test": len(usable_test),
            "date_train_min": usable_train["date"].min().date(), "date_train_max": usable_train["date"].max().date(),
            "date_test_min": usable_test["date"].min().date(), "date_test_max": usable_test["date"].max().date(),
        }

        y_reg = usable_train["target_return"].to_numpy()
        y_reg_test = usable_test["target_return"].to_numpy()
        for model_name, model in regression_models(seed, fast=fast).items():
            model.fit(x_train, y_reg)
            rows.append({"task": "regression", "model": model_name, **base,
                         **eval_regression(y_reg_test, model.predict(x_test))})

        y_cls = usable_train["target_up"].to_numpy()
        y_cls_test = usable_test["target_up"].to_numpy()
        if len(np.unique(y_cls)) < 2 or len(np.unique(y_cls_test)) < 2:
            continue
        for model_name, model in classification_models(seed, fast=fast).items():
            model.fit(x_train, y_cls)
            pred = model.predict(x_test)
            rows.append({"task": "classification", "model": model_name, **base,
                         **eval_classification(y_cls_test, pred, get_score(model, x_test))})
    return rows


def build_delta(results: pd.DataFrame) -> pd.DataFrame:
    key = ["scope", "task", "window", "model"]
    restricted = results[results["feature_set"] == "restricted"]
    with_sent = results[results["feature_set"] == "with_sentiment"]
    merged = with_sent.merge(restricted, on=key, suffixes=("_with_sentiment", "_restricted"))
    for metric in ["rmse", "mae", "r2", "accuracy", "f1", "auc"]:
        left, right = f"{metric}_with_sentiment", f"{metric}_restricted"
        if left in merged.columns and right in merged.columns:
            merged[f"{metric}_delta"] = merged[left] - merged[right]
    if "rmse_delta" in merged.columns:
        merged["rmse_improvement"] = -merged["rmse_delta"]
    if "mae_delta" in merged.columns:
        merged["mae_improvement"] = -merged["mae_delta"]
    return merged


def run(
    panel_path: str, output_path: str, windows: list[int], horizon: int,
    train_ratio: float, min_train: int, min_test: int,
    fill_missing_sentiment: float | None, seed: int, news_days_only: bool,
) -> None:
    panel = load_panel(
        panel_path, required_cols=_REQUIRED, numeric_cols=_NUMERIC,
        fill_missing_sentiment=fill_missing_sentiment, news_days_only=news_days_only,
    )
    rows = []
    for window in windows:
        data = build_lagged_dataset(panel, window, horizon).dropna(subset=["target_return"])
        rows.extend(run_one_scope("ALL", data, window, train_ratio, min_train, min_test, seed, horizon))
        for ticker, ticker_data in data.groupby("ticker"):
            rows.extend(run_one_scope(ticker, ticker_data, window, train_ratio, min_train, min_test, seed, horizon))

    results = pd.DataFrame(rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False, encoding="utf-8-sig")

    delta = build_delta(results)
    delta_path = output_path.replace(".csv", "_sentiment_delta.csv")
    delta.to_csv(delta_path, index=False, encoding="utf-8-sig")

    print(f"Saved model comparison to {output_path}")
    print(f"Saved sentiment deltas to {delta_path}")
    if not delta.empty:
        print("\nBest classification F1 improvements:")
        cls = delta[delta["task"] == "classification"].copy()
        if not cls.empty:
            cols = ["scope", "window", "model", "f1_delta", "accuracy_delta", "auc_delta",
                    "f1_with_sentiment", "f1_restricted"]
            print(cls.sort_values("f1_delta", ascending=False).head(15)[cols].to_string(index=False))
        print("\nBest regression RMSE improvements:")
        reg = delta[delta["task"] == "regression"].copy()
        if not reg.empty:
            cols = ["scope", "window", "model", "rmse_improvement", "r2_delta",
                    "rmse_with_sentiment", "rmse_restricted"]
            print(reg.sort_values("rmse_improvement", ascending=False).head(15)[cols].to_string(index=False))


def _parse_fill(value: str) -> float | None:
    if value.lower() == "nan":
        return None
    return float(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Granger-style restricted vs sentiment models")
    parser.add_argument("--panel", default=_DEFAULT_PANEL)
    parser.add_argument("--output", default="data/results/granger_style_model_comparison.csv")
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
        windows=args.windows,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        min_train=args.min_train,
        min_test=args.min_test,
        fill_missing_sentiment=_parse_fill(args.fill_missing_sentiment),
        seed=args.seed,
        news_days_only=args.news_days_only,
    )
