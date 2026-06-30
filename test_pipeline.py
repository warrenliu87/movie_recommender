"""Smoke + sanity test for the recommender pipeline on synthetic data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from src.data import make_synthetic, filter_sparse, train_test_split_ratings
from src.recommenders import (
    PopularityRecommender, ItemBasedCF, UserBasedCF,
    ContentBasedRecommender, MatrixFactorization,
)
from src.evaluation import evaluate_rating_prediction, evaluate_top_n

print("=" * 60)
print("Building synthetic MovieLens-shaped dataset...")
ml = make_synthetic(n_users=80, n_movies=120, density=0.18, seed=1)
print(f"  users={ml.n_users}  movies={ml.n_movies}  ratings={ml.n_ratings}  "
      f"sparsity={ml.sparsity:.3f}")

ratings = filter_sparse(ml.ratings, 5, 5)
train, test = train_test_split_ratings(ratings, test_frac=0.2, seed=7)
print(f"  after filter: {len(ratings)} ratings | train={len(train)} test={len(test)}")
n_catalog = ratings["movieId"].nunique()

print("\n[1] PopularityRecommender")
pop = PopularityRecommender(min_votes_quantile=0.5).fit(train, ml.movies)
top = pop.recommend(5)
print(top[["movieId", "rating_count", "rating_mean", "weighted_rating"]].to_string(index=False))
assert top["weighted_rating"].is_monotonic_decreasing

print("\n[2] ItemBasedCF")
icf = ItemBasedCF(k=30).fit(train)
uid = train["userId"].iloc[0]
recs = icf.recommend(uid, 5)
print(f"  recs for user {uid}: {recs['movieId'].tolist()}")
assert len(recs) == 5
sim = icf.similar_items(int(train['movieId'].iloc[0]), 3)
print(f"  similar items: {sim['movieId'].tolist()}")

print("\n[3] UserBasedCF")
ucf = UserBasedCF(k=20).fit(train)
recs = ucf.recommend(uid, 5)
print(f"  recs for user {uid}: {recs['movieId'].tolist()}")
assert len(recs) == 5

print("\n[4] ContentBasedRecommender")
cb = ContentBasedRecommender().fit(ml.movies, ml.tags)
sim = cb.similar_items(int(ml.movies['movieId'].iloc[0]), 5)
print(f"  content-similar to movie 1: {sim['movieId'].tolist()}")
ur = train[train["userId"] == uid]
cbrec = cb.recommend_for_user(ur, 5, like_threshold=3.5)
print(f"  content recs for user {uid}: {cbrec['movieId'].tolist()}")

print("\n[5] MatrixFactorization (SGD)")
mf = MatrixFactorization(n_factors=20, n_epochs=20, lr=0.01, reg=0.05, seed=3)
mf.fit(train, verbose=False)
print(f"  train RMSE curve: {mf.history_[0]:.3f} -> {mf.history_[-1]:.3f}")
assert mf.history_[-1] < mf.history_[0], "MF should reduce training error"
recs = mf.recommend(uid, 5, exclude=set(train[train.userId == uid].movieId))
print(f"  MF recs for user {uid}: {recs['movieId'].tolist()}")

print("\n[6] Evaluation")
mf_metrics = evaluate_rating_prediction(mf, test)
print(f"  MF rating prediction: RMSE={mf_metrics['rmse']:.3f} MAE={mf_metrics['mae']:.3f}")

# Global-mean baseline for comparison
class MeanBaseline:
    def __init__(self, mu): self.mu = mu
    def predict(self, u, m): return self.mu
baseline = MeanBaseline(train["rating"].mean())
bl_metrics = evaluate_rating_prediction(baseline, test)
print(f"  baseline (global mean): RMSE={bl_metrics['rmse']:.3f}")
assert mf_metrics["rmse"] < bl_metrics["rmse"], "MF should beat the mean baseline"

topn = evaluate_top_n(lambda u, k: mf.recommend(u, k), test, k=10,
                      like_threshold=3.5, n_catalog=n_catalog)
print(f"  MF top-10: precision={topn['precision@10']:.3f} "
      f"recall={topn['recall@10']:.3f} map={topn['map@10']:.3f} "
      f"coverage={topn['catalog_coverage']:.3f}")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED ✔")
