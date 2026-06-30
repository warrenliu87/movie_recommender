"""Stage 3 — Collaborative filtering (user-based & item-based)."""
import streamlit as st

from src.app_helpers import (
    get_data, get_splits, fit_item_cf, fit_user_cf,
    attach_titles, title_map, sidebar_config,
)

st.set_page_config(page_title="Collaborative Filtering", page_icon="👥", layout="wide")
st.title("👥 Collaborative Filtering")
st.caption("Recommendations from the ratings matrix alone — no movie metadata used.")

cfg = st.session_state.get("cfg") or sidebar_config()
data = get_data(use_synthetic=cfg["use_synthetic"])
movies = data["movies"]
_, train, _ = get_splits(data["ratings"], cfg["min_u"], cfg["min_m"], cfg["test_frac"])
tmap = title_map(movies)

tab_user, tab_item = st.tabs(["User-based", "Item-based"])

# ---------------------------------------------------------------- user-based
with tab_user:
    st.markdown("**Idea:** find users with similar taste, recommend what *they* liked.")
    c1, c2, c3 = st.columns(3)
    user_ids = sorted(train["userId"].unique())
    uid = c1.selectbox("User", user_ids, key="ucf_user")
    k = c2.slider("Neighbours (k)", 5, 80, 30, key="ucf_k")
    n = c3.slider("# recommendations", 5, 25, 10, key="ucf_n")

    ucf = fit_user_cf(train, k)
    recs = attach_titles(ucf.recommend(int(uid), n), movies)

    st.write(f"**What user {uid} has rated highly:**")
    hist = train[train.userId == uid].sort_values("rating", ascending=False).head(5)
    hist = hist.assign(title=hist.movieId.map(tmap))
    st.dataframe(hist[["title", "rating"]], hide_index=True, use_container_width=True)

    st.write("**Recommendations:**")
    if recs.empty:
        st.warning("No recommendations (user may have no similar neighbours).")
    else:
        st.dataframe(
            recs[["clean_title", "year", "genres", "score"]]
            .rename(columns={"clean_title": "Title", "year": "Year",
                             "genres": "Genres", "score": "Pred. rating"})
            .style.format({"Pred. rating": "{:.2f}", "Year": "{:.0f}"}),
            hide_index=True, use_container_width=True,
        )

# ---------------------------------------------------------------- item-based
with tab_item:
    st.markdown("**Idea:** *people who liked X also liked…* — recommend items similar "
                "to ones the user already rated highly.")
    mode = st.radio("Mode", ["Recommend for a user", "Find similar movies"],
                    horizontal=True, key="icf_mode")
    k = st.slider("Neighbourhood size (k)", 5, 80, 30, key="icf_k")
    icf = fit_item_cf(train, k)

    if mode == "Recommend for a user":
        c1, c2 = st.columns(2)
        uid = c1.selectbox("User", sorted(train["userId"].unique()), key="icf_user")
        n = c2.slider("# recommendations", 5, 25, 10, key="icf_n")
        recs = attach_titles(icf.recommend(int(uid), n), movies)
        st.dataframe(
            recs[["clean_title", "year", "genres", "score"]]
            .rename(columns={"clean_title": "Title", "year": "Year",
                             "genres": "Genres", "score": "Pred. rating"})
            .style.format({"Pred. rating": "{:.2f}", "Year": "{:.0f}"}),
            hide_index=True, use_container_width=True,
        )
    else:
        movie_options = sorted(train["movieId"].unique(), key=lambda m: tmap.get(m, ""))
        mid = st.selectbox("Movie", movie_options,
                           format_func=lambda m: tmap.get(m, m), key="icf_movie")
        n = st.slider("# similar", 5, 25, 10, key="icf_simn")
        sim = attach_titles(icf.similar_items(int(mid), n), movies)
        st.dataframe(
            sim[["clean_title", "year", "genres", "score"]]
            .rename(columns={"clean_title": "Title", "year": "Year",
                             "genres": "Genres", "score": "Similarity"})
            .style.format({"Similarity": "{:.3f}", "Year": "{:.0f}"}),
            hide_index=True, use_container_width=True,
        )
