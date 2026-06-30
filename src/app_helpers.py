"""
Shared Streamlit helpers: cached data loading and cached model fitting.

st.cache_data  -> for dataframes (serialisable results)
st.cache_resource -> for fitted model objects (kept in memory, not copied)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.recommenders import (  # noqa: E402
    PopularityRecommender, ItemBasedCF, UserBasedCF,
    ContentBasedRecommender, MatrixFactorization,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading MovieLens data…")
def get_data():
    """Load and cache the MovieLens dataset."""
    return _bundle(load_movielens(), source="ml-latest-small")

def _bundle(ml, source: str) -> dict:
    return {
        "ratings": ml.ratings,
        "movies": ml.movies,
        "tags": ml.tags,
        "links": ml.links,
        "n_users": ml.n_users,
        "n_movies": ml.n_movies,
        "n_ratings": ml.n_ratings,
        "sparsity": ml.sparsity,
        "source": source,
    }


@st.cache_data(show_spinner="Filtering + splitting…")
def get_splits(ratings: pd.DataFrame, min_u: int, min_m: int, test_frac: float):
    filtered = filter_sparse(ratings, min_u, min_m)
    train, test = train_test_split_ratings(filtered, test_frac=test_frac)
    return filtered, train, test


def title_map(movies: pd.DataFrame) -> dict[int, str]:
    return dict(zip(movies["movieId"], movies["clean_title"]))


def attach_titles(recs: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Join movieId -> title/genres/year for display."""
    if recs.empty:
        return recs
    return recs.merge(
        movies[["movieId", "clean_title", "genres", "year"]], on="movieId", how="left"
    )


# ---------------------------------------------------------------------------
# Models (cached so they fit once per parameter set)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Fitting item-based CF…")
def fit_item_cf(train: pd.DataFrame, k: int):
    return ItemBasedCF(k=k).fit(train)


@st.cache_resource(show_spinner="Fitting user-based CF…")
def fit_user_cf(train: pd.DataFrame, k: int):
    return UserBasedCF(k=k).fit(train)


@st.cache_resource(show_spinner="Building content model…")
def fit_content(movies: pd.DataFrame, tags: pd.DataFrame):
    return ContentBasedRecommender().fit(movies, tags)


@st.cache_resource(show_spinner="Training matrix factorisation…")
def fit_mf(train: pd.DataFrame, n_factors: int, n_epochs: int, lr: float, reg: float):
    return MatrixFactorization(
        n_factors=n_factors, n_epochs=n_epochs, lr=lr, reg=reg
    ).fit(train)


@st.cache_resource(show_spinner="Computing popularity…")
def fit_popularity(train: pd.DataFrame, movies: pd.DataFrame, q: float):
    return PopularityRecommender(min_votes_quantile=q).fit(train, movies)


# ---------------------------------------------------------------------------
# Sidebar config shared across pages
# ---------------------------------------------------------------------------
def sidebar_config() -> dict:
    st.sidebar.header("⚙️ Configuration")
    min_u = st.sidebar.slider("Min ratings / user", 1, 50, 5)
    min_m = st.sidebar.slider("Min ratings / movie", 1, 50, 5)
    test_frac = st.sidebar.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
    return {"min_u": min_u, "min_m": min_m, "test_frac": test_frac}
