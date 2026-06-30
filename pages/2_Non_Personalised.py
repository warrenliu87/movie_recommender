"""Stage 2 — Non-personalised (popularity) recommendations."""
import streamlit as st

from src.app_helpers import (
    get_data, get_splits, fit_popularity, attach_titles, sidebar_config,
)

st.set_page_config(page_title="Non-Personalised", page_icon="🏆", layout="wide")
st.title("🏆 Non-Personalised Recommendations")
st.caption("Same list for everyone — a strong cold-start fallback and evaluation baseline.")

cfg = st.session_state.get("cfg") or sidebar_config()
data = get_data(use_synthetic=cfg["use_synthetic"])
_, train, _ = get_splits(data["ratings"], cfg["min_u"], cfg["min_m"], cfg["test_frac"])

col1, col2 = st.columns([1, 2])
with col1:
    method = st.radio(
        "Ranking method",
        ["weighted_rating", "rating_count", "rating_mean"],
        format_func={
            "weighted_rating": "Weighted rating (IMDB-style)",
            "rating_count": "Most rated (popularity)",
            "rating_mean": "Highest average",
        }.get,
    )
    q = st.slider("Vote-floor quantile (m)", 0.5, 0.99, 0.90, 0.01,
                  help="Minimum #votes to qualify = this quantile of the vote counts.")
    n = st.slider("How many to show", 5, 30, 10)

pop = fit_popularity(train, data["movies"], q)

st.markdown(
    r"""
**Weighted rating** balances quality against vote count:

$$WR = \frac{v}{v+m}\,R + \frac{m}{v+m}\,C$$

where $v$ = #votes, $R$ = movie mean, $C$ = global mean, $m$ = vote floor.
"""
)

recs = pop.recommend(n=n, by=method)
show = recs[["clean_title", "year", "genres", "rating_count", "rating_mean", "weighted_rating"]]
show = show.rename(columns={
    "clean_title": "Title", "year": "Year", "genres": "Genres",
    "rating_count": "# ratings", "rating_mean": "Avg", "weighted_rating": "Weighted",
})
st.dataframe(
    show.style.format({"Avg": "{:.2f}", "Weighted": "{:.2f}", "Year": "{:.0f}"}),
    use_container_width=True, hide_index=True,
)

st.info(f"Global mean rating C = **{pop.C_:.2f}**, vote floor m = **{pop.m_:.0f}**. "
        "Notice how 'highest average' alone surfaces obscure movies, while the "
        "weighted rating favours broadly-loved ones.")
