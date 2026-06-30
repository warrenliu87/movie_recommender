"""
Evaluation metrics for the recommenders.

Two complementary views:
  - Rating-prediction accuracy: RMSE, MAE   (how close are predicted scores?)
  - Top-N ranking quality:       Precision@K, Recall@K, MAP@K, coverage
    (does the recommended list contain things the user actually liked?)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rating-prediction metrics
# ---------------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_rating_prediction(model, test: pd.DataFrame) -> dict:
    """For models exposing .predict(user_id, movie_id) -> float."""
    preds = np.array(
        [model.predict(int(r.userId), int(r.movieId)) for r in test.itertuples()]
    )
    truth = test["rating"].to_numpy()
    return {"rmse": rmse(truth, preds), "mae": mae(truth, preds), "n": len(test)}


# ---------------------------------------------------------------------------
# Top-N ranking metrics
# ---------------------------------------------------------------------------
def precision_recall_at_k(
    recommended: list[int], relevant: set[int], k: int
) -> tuple[float, float]:
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0, 0.0
    hits = sum(1 for m in rec_k if m in relevant)
    precision = hits / len(rec_k)
    recall = hits / len(relevant) if relevant else 0.0
    return precision, recall


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    score, hits = 0.0, 0
    for i, m in enumerate(recommended[:k], start=1):
        if m in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if relevant else 0.0


def evaluate_top_n(
    recommend_fn,
    test: pd.DataFrame,
    k: int = 10,
    like_threshold: float = 4.0,
    n_users: int | None = None,
    seed: int = 42,
    n_catalog: int | None = None,
) -> dict:
    """Evaluate a ranking model.

    recommend_fn(user_id, n) -> DataFrame with a 'movieId' column.
    A test item is 'relevant' for a user if they rated it >= like_threshold.
    """
    rng = np.random.default_rng(seed)
    users = test["userId"].unique()
    if n_users is not None and n_users < len(users):
        users = rng.choice(users, size=n_users, replace=False)

    precisions, recalls, aps = [], [], []
    recommended_items: set[int] = set()
    evaluated = 0

    for u in users:
        u_test = test[test["userId"] == u]
        relevant = set(u_test[u_test["rating"] >= like_threshold]["movieId"])
        if not relevant:
            continue
        recs = recommend_fn(int(u), k)
        rec_list = recs["movieId"].tolist() if len(recs) else []
        recommended_items.update(rec_list)
        p, r = precision_recall_at_k(rec_list, relevant, k)
        precisions.append(p)
        recalls.append(r)
        aps.append(average_precision_at_k(rec_list, relevant, k))
        evaluated += 1

    coverage = (len(recommended_items) / n_catalog) if n_catalog else float("nan")
    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"map@{k}": float(np.mean(aps)) if aps else 0.0,
        "catalog_coverage": coverage,
        "users_evaluated": evaluated,
    }
