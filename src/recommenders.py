"""
Recommender algorithms.

Four families, matching the project workflow:
  1. Non-personalised   -> PopularityRecommender
  2. Collaborative       -> ItemBasedCF, UserBasedCF
  3. Content-based       -> ContentBasedRecommender
  4. Matrix factorisation-> MatrixFactorization (SGD) + helper for TruncatedSVD

All recommenders expose a consistent surface where it makes sense:
  - .fit(...)
  - .recommend(...) -> DataFrame of movieId + score
so the Streamlit pages and evaluation code can treat them uniformly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===========================================================================
# 1. Non-personalised: popularity / weighted rating
# ===========================================================================
class PopularityRecommender:
    """Recommends globally popular movies. No personalisation.

    Uses the IMDB-style weighted rating to balance average score against the
    number of votes, so a single 5-star movie doesn't outrank a beloved classic:

        WR = (v / (v + m)) * R  +  (m / (v + m)) * C

    where v = #votes for the movie, R = its mean rating,
          C = global mean rating, m = minimum votes required (a quantile).
    """

    def __init__(self, min_votes_quantile: float = 0.90):
        self.min_votes_quantile = min_votes_quantile
        self.scores_: pd.DataFrame | None = None
        self.C_: float = 0.0
        self.m_: float = 0.0

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> "PopularityRecommender":
        agg = (
            ratings.groupby("movieId")["rating"]
            .agg(rating_count="count", rating_mean="mean")
            .reset_index()
        )
        self.C_ = ratings["rating"].mean()
        self.m_ = agg["rating_count"].quantile(self.min_votes_quantile)

        v, R = agg["rating_count"], agg["rating_mean"]
        agg["weighted_rating"] = (v / (v + self.m_)) * R + (self.m_ / (v + self.m_)) * self.C_
        self.scores_ = agg.merge(
            movies[["movieId", "clean_title", "genres", "year"]], on="movieId", how="left"
        )
        return self

    def recommend(self, n: int = 10, by: str = "weighted_rating") -> pd.DataFrame:
        """by: 'weighted_rating' (quality), 'rating_count' (most-rated),
        or 'rating_mean' (highest average, subject to the vote floor)."""
        df = self.scores_
        if by in ("weighted_rating", "rating_mean"):
            df = df[df["rating_count"] >= self.m_]
        return df.sort_values(by, ascending=False).head(n).reset_index(drop=True)


# ===========================================================================
# 2a. Item-based collaborative filtering
# ===========================================================================
class ItemBasedCF:
    """Item-item CF using cosine similarity on mean-centred rating vectors.

    Predicted rating for (user u, item i) is the similarity-weighted average
    of u's ratings on items similar to i.
    """

    def __init__(self, k: int = 30, shrinkage: float = 0.0):
        self.k = k
        self.shrinkage = shrinkage
        self.sim_: pd.DataFrame | None = None
        self.ui_: pd.DataFrame | None = None        # user x item, NaN where unrated
        self.user_means_: pd.Series | None = None

    def fit(self, ratings: pd.DataFrame) -> "ItemBasedCF":
        ui = ratings.pivot_table(index="userId", columns="movieId", values="rating")
        self.ui_ = ui
        self.user_means_ = ui.mean(axis=1)
        centred = ui.sub(self.user_means_, axis=0).fillna(0.0)
        sim = cosine_similarity(centred.T.values)       # items x items
        np.fill_diagonal(sim, 0.0)
        self.sim_ = pd.DataFrame(sim, index=ui.columns, columns=ui.columns)
        return self

    def _predict_user(self, user_id: int) -> pd.Series:
        if user_id not in self.ui_.index:
            return pd.Series(dtype=float)
        rated = self.ui_.loc[user_id].dropna()
        mean = self.user_means_[user_id]
        dev = rated - mean
        # weighted sum over rated items, for every candidate item at once
        sim_block = self.sim_.loc[rated.index]          # rated x all_items
        num = dev.values @ sim_block.values
        den = np.abs(sim_block.values).sum(axis=0) + 1e-9
        preds = pd.Series(mean + num / den, index=self.sim_.columns)
        return preds.drop(index=rated.index, errors="ignore")

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        preds = self._predict_user(user_id).sort_values(ascending=False).head(n)
        return pd.DataFrame({"movieId": preds.index, "score": preds.values})

    def similar_items(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        if movie_id not in self.sim_.columns:
            return pd.DataFrame(columns=["movieId", "score"])
        s = self.sim_[movie_id].sort_values(ascending=False).head(n)
        return pd.DataFrame({"movieId": s.index, "score": s.values})


# ===========================================================================
# 2b. User-based collaborative filtering
# ===========================================================================
class UserBasedCF:
    """User-user CF. Predicts from the k most similar users (neighbours)."""

    def __init__(self, k: int = 30):
        self.k = k
        self.sim_: pd.DataFrame | None = None
        self.ui_: pd.DataFrame | None = None
        self.user_means_: pd.Series | None = None

    def fit(self, ratings: pd.DataFrame) -> "UserBasedCF":
        ui = ratings.pivot_table(index="userId", columns="movieId", values="rating")
        self.ui_ = ui
        self.user_means_ = ui.mean(axis=1)
        centred = ui.sub(self.user_means_, axis=0).fillna(0.0)
        sim = cosine_similarity(centred.values)         # users x users
        np.fill_diagonal(sim, 0.0)
        self.sim_ = pd.DataFrame(sim, index=ui.index, columns=ui.index)
        return self

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        if user_id not in self.ui_.index:
            return pd.DataFrame(columns=["movieId", "score"])
        neighbours = self.sim_[user_id].sort_values(ascending=False).head(self.k)
        neighbours = neighbours[neighbours > 0]
        if neighbours.empty:
            return pd.DataFrame(columns=["movieId", "score"])

        mean = self.user_means_[user_id]
        nb_centred = self.ui_.loc[neighbours.index].sub(
            self.user_means_[neighbours.index], axis=0
        )
        weights = neighbours.values[:, None]
        num = np.nansum(nb_centred.values * weights, axis=0)
        den = np.nansum(~np.isnan(nb_centred.values) * np.abs(weights), axis=0) + 1e-9
        preds = pd.Series(mean + num / den, index=self.ui_.columns)
        already = self.ui_.loc[user_id].dropna().index
        preds = preds.drop(index=already, errors="ignore").sort_values(ascending=False)
        top = preds.head(n)
        return pd.DataFrame({"movieId": top.index, "score": top.values})


# ===========================================================================
# 3. Content-based filtering (genres + tags via TF-IDF)
# ===========================================================================
class ContentBasedRecommender:
    """Builds a TF-IDF profile per movie from genres (and tags if available),
    then recommends by cosine similarity in that content space.

    Can recommend either items similar to a given movie, or items matching a
    user's taste profile (the rating-weighted average of movies they liked).
    """

    def __init__(self):
        self.tfidf_: TfidfVectorizer | None = None
        self.matrix_ = None                 # sparse: movies x features
        self.movie_ids_: np.ndarray | None = None
        self.index_of_: dict[int, int] = {}

    def fit(self, movies: pd.DataFrame, tags: pd.DataFrame | None = None) -> "ContentBasedRecommender":
        soup = movies.set_index("movieId")["genre_list"].apply(lambda g: " ".join(g))
        if tags is not None and not tags.empty:
            tag_str = (
                tags.groupby("movieId")["tag"]
                .apply(lambda s: " ".join(map(str, s)))
            )
            soup = soup.add(" " + tag_str.reindex(soup.index).fillna(""), fill_value="")

        self.movie_ids_ = soup.index.to_numpy()
        self.index_of_ = {mid: i for i, mid in enumerate(self.movie_ids_)}
        self.tfidf_ = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=True)
        self.matrix_ = self.tfidf_.fit_transform(soup.values)
        return self

    def similar_items(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        if movie_id not in self.index_of_:
            return pd.DataFrame(columns=["movieId", "score"])
        idx = self.index_of_[movie_id]
        sims = cosine_similarity(self.matrix_[idx], self.matrix_).ravel()
        sims[idx] = -1.0
        top = np.argsort(sims)[::-1][:n]
        return pd.DataFrame({"movieId": self.movie_ids_[top], "score": sims[top]})

    def recommend_for_user(
        self, user_ratings: pd.DataFrame, n: int = 10, like_threshold: float = 4.0
    ) -> pd.DataFrame:
        """user_ratings: DataFrame with movieId + rating for one user."""
        liked = user_ratings[user_ratings["rating"] >= like_threshold]
        liked = liked[liked["movieId"].isin(self.index_of_)]
        if liked.empty:
            return pd.DataFrame(columns=["movieId", "score"])
        rows = [self.index_of_[m] for m in liked["movieId"]]
        weights = liked["rating"].values[:, None]
        profile = np.asarray(self.matrix_[rows].multiply(weights).sum(axis=0))
        profile = profile / (weights.sum() + 1e-9)
        sims = cosine_similarity(profile, self.matrix_).ravel()
        seen = {self.index_of_[m] for m in user_ratings["movieId"] if m in self.index_of_}
        for i in seen:
            sims[i] = -1.0
        top = np.argsort(sims)[::-1][:n]
        return pd.DataFrame({"movieId": self.movie_ids_[top], "score": sims[top]})


# ===========================================================================
# 4. Matrix factorisation (SGD, à la Funk-SVD)
# ===========================================================================
class MatrixFactorization:
    """Latent-factor model trained with SGD on observed ratings.

        r_hat[u,i] = mu + b_u + b_i + p_u . q_i

    Learns user/item biases and k-dim latent factors. This is the workhorse
    behind the Netflix-Prize-era recommenders and predicts well on sparse data.
    """

    def __init__(self, n_factors: int = 50, n_epochs: int = 30, lr: float = 0.005,
                 reg: float = 0.02, seed: int = 42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed
        self.history_: list[float] = []

    def fit(self, ratings: pd.DataFrame, verbose: bool = False) -> "MatrixFactorization":
        rng = np.random.default_rng(self.seed)
        self.user_ids_ = ratings["userId"].unique()
        self.movie_ids_ = ratings["movieId"].unique()
        self.u_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.i_index_ = {m: i for i, m in enumerate(self.movie_ids_)}
        n_u, n_i = len(self.user_ids_), len(self.movie_ids_)

        self.mu_ = ratings["rating"].mean()
        self.b_u_ = np.zeros(n_u)
        self.b_i_ = np.zeros(n_i)
        self.P_ = rng.normal(0, 0.1, (n_u, self.n_factors))
        self.Q_ = rng.normal(0, 0.1, (n_i, self.n_factors))

        u_arr = ratings["userId"].map(self.u_index_).to_numpy()
        i_arr = ratings["movieId"].map(self.i_index_).to_numpy()
        r_arr = ratings["rating"].to_numpy(dtype=float)
        order = np.arange(len(r_arr))

        for epoch in range(self.n_epochs):
            rng.shuffle(order)
            sq_err = 0.0
            for idx in order:
                u, i, r = u_arr[idx], i_arr[idx], r_arr[idx]
                pred = self.mu_ + self.b_u_[u] + self.b_i_[i] + self.P_[u] @ self.Q_[i]
                e = r - pred
                sq_err += e * e
                self.b_u_[u] += self.lr * (e - self.reg * self.b_u_[u])
                self.b_i_[i] += self.lr * (e - self.reg * self.b_i_[i])
                pu = self.P_[u].copy()
                self.P_[u] += self.lr * (e * self.Q_[i] - self.reg * self.P_[u])
                self.Q_[i] += self.lr * (e * pu - self.reg * self.Q_[i])
            rmse = np.sqrt(sq_err / len(r_arr))
            self.history_.append(rmse)
            if verbose:
                print(f"epoch {epoch + 1:2d}  train RMSE={rmse:.4f}")
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        pred = self.mu_
        if user_id in self.u_index_:
            u = self.u_index_[user_id]
            pred += self.b_u_[u]
            if movie_id in self.i_index_:
                i = self.i_index_[movie_id]
                pred += self.b_i_[i] + self.P_[u] @ self.Q_[i]
        elif movie_id in self.i_index_:
            pred += self.b_i_[self.i_index_[movie_id]]
        return float(np.clip(pred, 0.5, 5.0))

    def recommend(self, user_id: int, n: int = 10, exclude: set[int] | None = None) -> pd.DataFrame:
        if user_id not in self.u_index_:
            return pd.DataFrame(columns=["movieId", "score"])
        u = self.u_index_[user_id]
        scores = self.mu_ + self.b_u_[u] + self.b_i_ + self.Q_ @ self.P_[u]
        s = pd.Series(scores, index=self.movie_ids_)
        if exclude:
            s = s.drop(index=[m for m in exclude if m in s.index], errors="ignore")
        top = s.sort_values(ascending=False).head(n)
        return pd.DataFrame({"movieId": top.index, "score": top.values})
