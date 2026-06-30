"""Stage 4 — Content-based filtering (TF-IDF over genres + tags)."""
import streamlit as st

from src.app_helpers import (
    get_data, get_splits, fit_content, attach_titles, title_map, sidebar_config,
)

st.set_page_config(page_title="Content-Based", page_icon="🎭", layout="wide")
st.title("🎭 Content-Based Filtering")
st.caption("Uses movie attributes (genres + tags) — works even for movies with no ratings.")

cfg = st.session_state.get("cfg") or sidebar_config()
data = get_data(use_synthetic=cfg["use_synthetic"])
movies, tags = data["movies"], data["tags"]
_, train, _ = get_splits(data["ratings"], cfg["min_u"], cfg["min_m"], cfg["test_frac"])
tmap = title_map(movies)

cb = fit_content(movies, tags)

st.markdown(
    "Each movie becomes a **TF-IDF vector** over its genres (and tags, if present). "
    "Similarity is cosine distance in that space."
)

mode = st.radio("Mode", ["More like this", "Recommend for a user"], horizontal=True)

if mode == "More like this":
    movie_options = sorted(movies["movieId"].tolist(), key=lambda m: tmap.get(m, ""))
    mid = st.selectbox("Pick a movie", movie_options,
                       format_func=lambda m: tmap.get(m, m))
    n = st.slider("# similar movies", 5, 25, 10)
    src = movies[movies.movieId == mid].iloc[0]
    st.caption(f"Genres: {src['genres']}")
    sim = attach_titles(cb.similar_items(int(mid), n), movies)
    st.dataframe(
        sim[["clean_title", "year", "genres", "score"]]
        .rename(columns={"clean_title": "Title", "year": "Year",
                         "genres": "Genres", "score": "Similarity"})
        .style.format({"Similarity": "{:.3f}", "Year": "{:.0f}"}),
        hide_index=True, use_container_width=True,
    )
else:
    c1, c2, c3 = st.columns(3)
    uid = c1.selectbox("User", sorted(train["userId"].unique()))
    thr = c2.slider("'Liked' threshold", 3.0, 5.0, 4.0, 0.5)
    n = c3.slider("# recommendations", 5, 25, 10)
    user_ratings = train[train.userId == uid]

    st.write("**Built from these highly-rated movies:**")
    liked = user_ratings[user_ratings.rating >= thr].sort_values("rating", ascending=False)
    liked = liked.assign(title=liked.movieId.map(tmap))
    st.dataframe(liked[["title", "rating"]].head(8), hide_index=True, use_container_width=True)

    recs = attach_titles(cb.recommend_for_user(user_ratings, n, like_threshold=thr), movies)
    st.write("**Recommendations matching their taste profile:**")
    if recs.empty:
        st.warning("No liked movies above the threshold — lower it to build a profile.")
    else:
        st.dataframe(
            recs[["clean_title", "year", "genres", "score"]]
            .rename(columns={"clean_title": "Title", "year": "Year",
                             "genres": "Genres", "score": "Match"})
            .style.format({"Match": "{:.3f}", "Year": "{:.0f}"}),
            hide_index=True, use_container_width=True,
        )

st.info("Content-based filtering can't surprise you much — it stays within genres you "
        "already like. That's why it's often combined with collaborative methods (hybrid).")
