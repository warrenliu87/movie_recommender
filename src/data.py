"""
Data loading, download, and preprocessing for the MovieLens recommender.

Default dataset: ml-latest-small (~100k ratings, 9k movies, 600 users).
It's small enough for fast prototyping in Streamlit. To scale up, point
DATASET_URL at ml-1m or ml-25m and adjust the file paths.
"""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXTRACT_DIR = DATA_DIR / "ml-latest-small"


# ---------------------------------------------------------------------------
# Download / load
# ---------------------------------------------------------------------------
def download_dataset(force: bool = False) -> Path:
    """Download and extract the MovieLens dataset if it isn't present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if EXTRACT_DIR.exists() and not force:
        return EXTRACT_DIR

    import urllib.request

    print(f"Downloading MovieLens dataset from {DATASET_URL} ...")
    with urllib.request.urlopen(DATASET_URL) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to {EXTRACT_DIR}")
    return EXTRACT_DIR


@dataclass
class MovieLens:
    """Container for the raw + lightly cleaned MovieLens tables."""

    ratings: pd.DataFrame   # userId, movieId, rating, timestamp
    movies: pd.DataFrame    # movieId, title, genres, year, genre_list
    tags: pd.DataFrame      # userId, movieId, tag, timestamp
    links: pd.DataFrame     # movieId, imdbId, tmdbId

    @property
    def n_users(self) -> int:
        return self.ratings["userId"].nunique()

    @property
    def n_movies(self) -> int:
        return self.ratings["movieId"].nunique()

    @property
    def n_ratings(self) -> int:
        return len(self.ratings)

    @property
    def sparsity(self) -> float:
        return 1.0 - self.n_ratings / (self.n_users * self.n_movies)


def load_movielens(data_dir: Path | str | None = None) -> MovieLens:
    """Load and lightly clean the MovieLens tables into a MovieLens object."""
    base = Path(data_dir) if data_dir else download_dataset()

    ratings = pd.read_csv(base / "ratings.csv")
    movies = pd.read_csv(base / "movies.csv")
    links = pd.read_csv(base / "links.csv")
    try:
        tags = pd.read_csv(base / "tags.csv")
    except FileNotFoundError:
        tags = pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])

    movies = _clean_movies(movies)
    return MovieLens(ratings=ratings, movies=movies, tags=tags, links=links)


def _clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Extract release year and split the pipe-separated genre string."""
    movies = movies.copy()
    # Year is the trailing "(YYYY)" in the title for most rows.
    movies["year"] = (
        movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("Float64")
    )
    movies["clean_title"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)\s*$", "", regex=True
    )
    movies["genre_list"] = movies["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    return movies


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def filter_sparse(
    ratings: pd.DataFrame, min_user_ratings: int = 5, min_movie_ratings: int = 5
) -> pd.DataFrame:
    """Drop users / movies with too few ratings (reduces noise + matrix size).

    Applied iteratively because removing a user can drop a movie below the
    threshold and vice-versa.
    """
    out = ratings.copy()
    while True:
        n_before = len(out)
        uc = out["userId"].value_counts()
        out = out[out["userId"].isin(uc[uc >= min_user_ratings].index)]
        mc = out["movieId"].value_counts()
        out = out[out["movieId"].isin(mc[mc >= min_movie_ratings].index)]
        if len(out) == n_before:
            break
    return out.reset_index(drop=True)


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Return a dense user x movie ratings matrix (NaN where unrated)."""
    return ratings.pivot_table(index="userId", columns="movieId", values="rating")


def train_test_split_ratings(
    ratings: pd.DataFrame, test_frac: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user temporal-ish split: hold out a random fraction of each user's
    ratings for testing. Keeps every user represented in both sets.
    """
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, grp in ratings.groupby("userId"):
        n_test = max(1, int(round(len(grp) * test_frac)))
        test_idx.extend(rng.choice(grp.index.values, size=n_test, replace=False))
    test_mask = ratings.index.isin(test_idx)
    return ratings[~test_mask].reset_index(drop=True), ratings[test_mask].reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Synthetic data (for offline tests / demos without the download)
# ---------------------------------------------------------------------------
def make_synthetic(
    n_users: int = 80, n_movies: int = 120, density: float = 0.15, seed: int = 0
) -> MovieLens:
    """Generate a small MovieLens-shaped dataset with latent-factor structure,
    so collaborative methods have real signal to find. Used by the test suite.
    """
    rng = np.random.default_rng(seed)
    k = 3
    user_f = rng.normal(size=(n_users, k))
    movie_f = rng.normal(size=(n_movies, k))
    raw = user_f @ movie_f.T
    raw = (raw - raw.mean()) / raw.std()              # standardise
    user_bias = rng.normal(0, 0.4, size=(n_users, 1))
    movie_bias = rng.normal(0, 0.4, size=(1, n_movies))
    base = 3.2 + 1.1 * raw + user_bias + movie_bias    # spread ratings out

    rows = []
    for u in range(n_users):
        for m in range(n_movies):
            if rng.random() < density:
                r = float(np.clip(round((base[u, m] + rng.normal(0, 0.25)) * 2) / 2, 0.5, 5))
                rows.append((u + 1, m + 1, r, 1_000_000_000 + len(rows)))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])

    genres_pool = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller"]
    mrows = []
    for m in range(n_movies):
        gl = rng.choice(genres_pool, size=rng.integers(1, 4), replace=False).tolist()
        mrows.append((m + 1, f"Movie {m + 1} ({1990 + m % 30})", "|".join(gl)))
    movies = _clean_movies(pd.DataFrame(mrows, columns=["movieId", "title", "genres"]))

    tags = pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
    links = pd.DataFrame({"movieId": movies["movieId"], "imdbId": 0, "tmdbId": 0})
    return MovieLens(ratings=ratings, movies=movies, tags=tags, links=links)
