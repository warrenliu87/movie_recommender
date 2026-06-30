"""Stage 5 — Matrix factorisation (Funk-SVD trained with SGD)."""
import altair as alt
import pandas as pd
import streamlit as st

from src.app_helpers import (
    get_data, get_splits, fit_mf, attach_titles, title_map, sidebar_config,
)

st.set_page_config(page_title="Matrix Factorisation", page_icon="🧮", layout="wide")
st.title("🧮 Matrix Factorisation (SVD)")
st.caption("Learns latent factors for users and movies — usually the strongest single model.")

cfg = st.session_state.get("cfg") or sidebar_config()
data = get_data(use_synthetic=cfg["use_synthetic"])
movies = data["movies"]
_, train, _ = get_splits(data["ratings"], cfg["min_u"], cfg["min_m"], cfg["test_frac"])
tmap = title_map(movies)

st.markdown(
    r"""
Predicts a rating as $\hat r_{ui} = \mu + b_u + b_i + \mathbf{p}_u \cdot \mathbf{q}_i$,
where $\mathbf{p}_u, \mathbf{q}_i$ are learned $k$-dimensional latent vectors and
$b_u, b_i$ are user/item biases. Trained by stochastic gradient descent.
"""
)

with st.sidebar:
    st.header("MF hyperparameters")
    n_factors = st.slider("Latent factors (k)", 5, 100, 50, 5)
    n_epochs = st.slider("Epochs", 5, 60, 30, 5)
    lr = st.select_slider("Learning rate", [0.001, 0.005, 0.01, 0.02], value=0.005)
    reg = st.select_slider("Regularisation", [0.005, 0.01, 0.02, 0.05, 0.1], value=0.02)

mf = fit_mf(train, n_factors, n_epochs, lr, reg)

# ---- training curve -------------------------------------------------------
st.subheader("Training convergence")
curve = pd.DataFrame({"epoch": range(1, len(mf.history_) + 1), "train_rmse": mf.history_})
st.altair_chart(
    alt.Chart(curve).mark_line(point=True).encode(
        x="epoch:Q", y=alt.Y("train_rmse:Q", title="Train RMSE"),
        tooltip=["epoch", "train_rmse"],
    ).properties(height=260),
    use_container_width=True,
)
st.caption(f"Train RMSE: {mf.history_[0]:.3f} → {mf.history_[-1]:.3f}")

# ---- recommendations ------------------------------------------------------
st.subheader("Recommendations")
c1, c2 = st.columns(2)
uid = c1.selectbox("User", sorted(train["userId"].unique()))
n = c2.slider("# recommendations", 5, 25, 10)
seen = set(train[train.userId == uid].movieId)
recs = attach_titles(mf.recommend(int(uid), n, exclude=seen), movies)
st.dataframe(
    recs[["clean_title", "year", "genres", "score"]]
    .rename(columns={"clean_title": "Title", "year": "Year",
                     "genres": "Genres", "score": "Pred. rating"})
    .style.format({"Pred. rating": "{:.2f}", "Year": "{:.0f}"}),
    hide_index=True, use_container_width=True,
)

# ---- single prediction ----------------------------------------------------
with st.expander("Predict a specific (user, movie) rating"):
    mid = st.selectbox("Movie", sorted(train["movieId"].unique()),
                       format_func=lambda m: tmap.get(m, m))
    pred = mf.predict(int(uid), int(mid))
    st.metric(f"Predicted rating for user {uid} on '{tmap.get(mid, mid)}'", f"{pred:.2f} ★")
