"""
🎬 MovieLens Recommender — Home

Run with:  streamlit run Home.py
"""

import streamlit as st

from src.app_helpers import get_data, sidebar_config

st.set_page_config(page_title="MovieLens Recommender", page_icon="🎬", layout="wide")

st.title("🎬 MovieLens Recommender System")
st.caption("A prototype recommender built on the MovieLens dataset, following a "
           "non-personalised → collaborative → content-based → matrix-factorisation workflow.")

cfg = sidebar_config()
data = get_data(use_synthetic=cfg["use_synthetic"])

# Keep config available to all pages
st.session_state["cfg"] = cfg

c1, c2, c3, c4 = st.columns(4)
c1.metric("Users", f"{data['n_users']:,}")
c2.metric("Movies", f"{data['n_movies']:,}")
c3.metric("Ratings", f"{data['n_ratings']:,}")
c4.metric("Sparsity", f"{data['sparsity']:.2%}")

st.info(f"Data source: **{data['source']}**. "
        "Switch to synthetic data in the sidebar if the download is blocked.")

st.subheader("Project workflow")
st.markdown(
    """
| Stage | Page | What it does |
|---|---|---|
| **1. Dataset + EDA + preprocessing** | `Dataset and EDA` | Explore distributions, sparsity, top genres; filter sparse users/movies. |
| **2. Non-personalised** | `Non Personalised` | Popularity & IMDB-style weighted-rating recommendations. |
| **3. Collaborative filtering** | `Collaborative Filtering` | User-based & item-based neighbourhood models. |
| **4. Content-based filtering** | `Content Based` | TF-IDF over genres/tags; "more like this" + taste profiles. |
| **5. Matrix factorisation** | `Matrix Factorisation` | Funk-SVD latent-factor model trained with SGD. |
| **6. Evaluation** | `Evaluation` | RMSE/MAE and Precision/Recall/MAP@K across models. |

Use the **sidebar pages** to move through the stages. Configuration in the
left sidebar (data source, sparsity filters, test split) applies everywhere.
"""
)

with st.expander("How the recommenders differ"):
    st.markdown(
        """
- **Non-personalised** ignores who you are — everyone sees the same popular list.
  Great as a cold-start fallback and an evaluation baseline.
- **Collaborative filtering** uses the ratings matrix only: *"users like you also
  liked…"* (user-based) or *"people who liked X also liked…"* (item-based).
- **Content-based** uses movie *attributes* (genres, tags). It can recommend brand-new
  movies with no ratings, but tends to stay inside a user's existing tastes.
- **Matrix factorisation** learns latent factors (e.g. a hidden "quirky-indie" axis)
  for users and movies. Usually the strongest single model on accuracy.
"""
    )
