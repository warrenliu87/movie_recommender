"""Stage 6 — Evaluation across all models."""
import altair as alt
import pandas as pd
import streamlit as st

from src.app_helpers import (
    get_data, get_splits, fit_popularity, fit_item_cf, fit_user_cf, fit_mf,
    sidebar_config,
)
from src.evaluation import evaluate_rating_prediction, evaluate_top_n

st.set_page_config(page_title="Evaluation", page_icon="📐", layout="wide")
st.title("📐 Evaluation")
st.caption("Compare models on rating-prediction accuracy and top-N ranking quality.")

cfg = st.session_state.get("cfg") or sidebar_config()
data = get_data(use_synthetic=cfg["use_synthetic"])
movies = data["movies"]
filtered, train, test = get_splits(
    data["ratings"], cfg["min_u"], cfg["min_m"], cfg["test_frac"]
)
n_catalog = filtered["movieId"].nunique()

c1, c2, c3 = st.columns(3)
k = c1.slider("Top-N cutoff (K)", 5, 20, 10)
thr = c2.slider("Relevance threshold", 3.0, 5.0, 4.0, 0.5)
sample_users = c3.slider("Users sampled for ranking eval", 50, 500, 150, 50)

run = st.button("▶ Run evaluation", type="primary")

st.markdown(
    """
- **RMSE / MAE** — rating-prediction error (lower is better). Only defined for models
  that predict a score (MF, and a global-mean baseline).
- **Precision/Recall/MAP@K** — top-N ranking quality (higher is better): does the
  recommended list contain movies the user actually liked in the held-out test set?
"""
)

if run:
    class MeanBaseline:
        def __init__(self, mu): self.mu = mu
        def predict(self, u, m): return self.mu

    # ---- fit models -------------------------------------------------------
    with st.status("Fitting models…", expanded=True) as status:
        st.write("Popularity…")
        pop = fit_popularity(train, movies, 0.9)
        st.write("Item-based CF…")
        icf = fit_item_cf(train, 30)
        st.write("User-based CF…")
        ucf = fit_user_cf(train, 30)
        st.write("Matrix factorisation…")
        mf = fit_mf(train, 50, 30, 0.005, 0.02)
        baseline = MeanBaseline(train["rating"].mean())
        status.update(label="Models fitted", state="complete")

    # ---- rating prediction ------------------------------------------------
    st.subheader("Rating-prediction accuracy")
    rp_rows = [
        {"model": "Global mean", **evaluate_rating_prediction(baseline, test)},
        {"model": "Matrix Factorisation", **evaluate_rating_prediction(mf, test)},
    ]
    rp = pd.DataFrame(rp_rows)[["model", "rmse", "mae", "n"]]
    st.dataframe(rp.style.format({"rmse": "{:.4f}", "mae": "{:.4f}"}),
                 hide_index=True, use_container_width=True)

    # ---- top-N ------------------------------------------------------------
    st.subheader(f"Top-{k} ranking quality")

    pop_list = pop.recommend(n=200, by="weighted_rating")["movieId"].tolist()

    def pop_recommend(u, n):
        return pd.DataFrame({"movieId": pop_list[:n]})

    models = {
        "Popularity": pop_recommend,
        "Item-based CF": lambda u, n: icf.recommend(u, n),
        "User-based CF": lambda u, n: ucf.recommend(u, n),
        "Matrix Factorisation": lambda u, n: mf.recommend(u, n),
    }

    rows = []
    prog = st.progress(0.0)
    for j, (name, fn) in enumerate(models.items(), start=1):
        m = evaluate_top_n(fn, test, k=k, like_threshold=thr,
                           n_users=sample_users, n_catalog=n_catalog)
        rows.append({"model": name, **m})
        prog.progress(j / len(models))
    prog.empty()

    topn = pd.DataFrame(rows)
    st.dataframe(
        topn.style.format({
            f"precision@{k}": "{:.4f}", f"recall@{k}": "{:.4f}",
            f"map@{k}": "{:.4f}", "catalog_coverage": "{:.3f}",
        }),
        hide_index=True, use_container_width=True,
    )

    melt = topn.melt(
        id_vars="model",
        value_vars=[f"precision@{k}", f"recall@{k}", f"map@{k}"],
        var_name="metric", value_name="value",
    )
    st.altair_chart(
        alt.Chart(melt).mark_bar().encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=-20)),
            y=alt.Y("value:Q"),
            color="model:N",
            column=alt.Column("metric:N", title=None),
            tooltip=["model", "metric", "value"],
        ).properties(height=240),
        use_container_width=False,
    )
    st.info("Typical finding: matrix factorisation wins on RMSE, while item-based CF "
            "and MF compete on top-N. Popularity is a surprisingly tough baseline for "
            "precision because popular movies are popular for a reason.")
else:
    st.info("Set the parameters above and click **Run evaluation**. "
            "Fitting all four models can take a little while on the full dataset.")
