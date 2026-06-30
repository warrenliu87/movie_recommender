"""Stage 1 — Dataset, EDA and preprocessing."""
import altair as alt
import pandas as pd
import streamlit as st

from src.app_helpers import get_data, get_splits, sidebar_config

st.set_page_config(page_title="Dataset & EDA", page_icon="📊", layout="wide")
st.title("📊 Dataset, EDA & Preprocessing")

cfg = st.session_state.get("cfg") or sidebar_config()
data = get_data(use_synthetic=cfg["use_synthetic"])
ratings, movies, tags = data["ratings"], data["movies"], data["tags"]

# ---- Overview -------------------------------------------------------------
st.subheader("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Users", f"{data['n_users']:,}")
c2.metric("Movies", f"{data['n_movies']:,}")
c3.metric("Ratings", f"{data['n_ratings']:,}")
c4.metric("Mean rating", f"{ratings['rating'].mean():.2f}")

with st.expander("Peek at the raw tables"):
    st.write("**ratings**"); st.dataframe(ratings.head(), use_container_width=True)
    st.write("**movies**"); st.dataframe(movies.head(), use_container_width=True)

# ---- Distributions --------------------------------------------------------
st.subheader("Rating distribution")
rdist = ratings["rating"].value_counts().sort_index().reset_index()
rdist.columns = ["rating", "count"]
st.altair_chart(
    alt.Chart(rdist).mark_bar().encode(
        x=alt.X("rating:O", title="Rating"),
        y=alt.Y("count:Q", title="Count"),
        tooltip=["rating", "count"],
    ).properties(height=260),
    use_container_width=True,
)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Ratings per user")
    upc = ratings.groupby("userId").size().reset_index(name="n")
    st.altair_chart(
        alt.Chart(upc).mark_bar().encode(
            x=alt.X("n:Q", bin=alt.Bin(maxbins=40), title="# ratings"),
            y=alt.Y("count():Q", title="# users"),
        ).properties(height=240),
        use_container_width=True,
    )
    st.caption(f"Median user has {int(upc['n'].median())} ratings; "
               f"max {int(upc['n'].max())}.")
with col_b:
    st.subheader("Ratings per movie")
    mpc = ratings.groupby("movieId").size().reset_index(name="n")
    st.altair_chart(
        alt.Chart(mpc).mark_bar().encode(
            x=alt.X("n:Q", bin=alt.Bin(maxbins=40), title="# ratings"),
            y=alt.Y("count():Q", title="# movies"),
        ).properties(height=240),
        use_container_width=True,
    )
    st.caption("The long tail: most movies have very few ratings.")

# ---- Genres ---------------------------------------------------------------
st.subheader("Most common genres")
genre_counts = (
    movies.explode("genre_list")
    .dropna(subset=["genre_list"])
    .groupby("genre_list").size().reset_index(name="count")
    .sort_values("count", ascending=False)
)
st.altair_chart(
    alt.Chart(genre_counts).mark_bar().encode(
        x=alt.X("count:Q", title="# movies"),
        y=alt.Y("genre_list:N", sort="-x", title="Genre"),
        tooltip=["genre_list", "count"],
    ).properties(height=380),
    use_container_width=True,
)

# ---- Sparsity / preprocessing --------------------------------------------
st.subheader("Preprocessing: filtering sparse users & movies")
st.markdown(
    "Collaborative methods struggle when users/movies have only one or two "
    "ratings. We iteratively drop them, which shrinks the user–item matrix and "
    "reduces noise. Adjust the thresholds in the sidebar."
)
filtered, train, test = get_splits(ratings, cfg["min_u"], cfg["min_m"], cfg["test_frac"])
c1, c2, c3 = st.columns(3)
c1.metric("Ratings after filter", f"{len(filtered):,}",
          delta=f"{len(filtered) - len(ratings):,}")
c2.metric("Train ratings", f"{len(train):,}")
c3.metric("Test ratings", f"{len(test):,}")

dense_users = filtered["userId"].nunique()
dense_movies = filtered["movieId"].nunique()
new_sparsity = 1 - len(filtered) / (dense_users * dense_movies)
st.caption(f"After filtering: {dense_users:,} users × {dense_movies:,} movies, "
           f"sparsity {new_sparsity:.2%}. These splits feed every model + the evaluation page.")
