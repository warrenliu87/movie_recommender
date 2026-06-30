# 🎬 MovieLens Recommender System

A prototype movie recommender built on the **MovieLens** dataset with a **Streamlit**
multipage UI. It walks through a full recommender workflow, one stage per page:

1. **Dataset + EDA + preprocessing**
2. **Non-personalised** (popularity / weighted rating)
3. **Collaborative filtering** (user-based & item-based)
4. **Content-based filtering** (TF-IDF over genres + tags)
5. **Matrix factorisation** (Funk-SVD via SGD)
6. **Evaluation** (RMSE/MAE + Precision/Recall/MAP@K)

## Quick start

```bash
cd movie-recommender
python -m venv .venv && source .venv/bin/activate    # optional but recommended
pip install -r requirements.txt
streamlit run Home.py
```

The app opens in your browser. On first run it downloads `ml-latest-small`
(~1 MB, ~100k ratings) from grouplens.org into `./data/`. If the download is
blocked, flip **"Use synthetic data"** in the sidebar — the app generates a small
MovieLens-shaped dataset so everything still works offline.

### Scaling to a bigger dataset
Edit `DATASET_URL` in `src/data.py` to point at `ml-1m` or `ml-25m`. Note that
the user-based/item-based CF build dense similarity matrices, so for ml-25m you'll
want to raise the sparsity filters (sidebar) or switch those pages to a sparse /
approximate-nearest-neighbour implementation.

## Project structure

```
movie-recommender/
├── Home.py                       # Streamlit entry point
├── pages/                        # one page per workflow stage
│   ├── 1_Dataset_and_EDA.py
│   ├── 2_Non_Personalised.py
│   ├── 3_Collaborative_Filtering.py
│   ├── 4_Content_Based.py
│   ├── 5_Matrix_Factorisation.py
│   └── 6_Evaluation.py
├── src/
│   ├── data.py                   # download, load, clean, split, synthetic data
│   ├── recommenders.py           # all 5 recommender models
│   ├── evaluation.py             # RMSE/MAE + ranking metrics
│   └── app_helpers.py            # cached loaders shared across pages
├── test_pipeline.py              # offline sanity test (no Streamlit needed)
├── requirements.txt
└── README.md
```

## The models

| Model | File / class | Uses | Strengths | Weaknesses |
|---|---|---|---|---|
| Popularity | `PopularityRecommender` | rating counts/means | great cold-start baseline | identical for everyone |
| User-based CF | `UserBasedCF` | ratings matrix | intuitive, personalised | cold-start, scaling |
| Item-based CF | `ItemBasedCF` | ratings matrix | stable, "more like this" | needs rating overlap |
| Content-based | `ContentBasedRecommender` | genres + tags | handles unrated movies | stays in your bubble |
| Matrix factorisation | `MatrixFactorization` | ratings matrix | best accuracy, compact | latent factors opaque |

Each exposes a consistent `.fit()` / `.recommend()` surface so the UI and the
evaluation page treat them uniformly.

## Evaluation

`evaluate_rating_prediction` reports **RMSE/MAE** for score-predicting models
(matrix factorisation + a global-mean baseline). `evaluate_top_n` reports
**Precision@K, Recall@K, MAP@K** and catalog coverage for the ranked top-N lists,
using a per-user hold-out split where a test item counts as *relevant* if the user
rated it at or above the threshold.

## Sanity check without the UI

```bash
python test_pipeline.py
```

Builds a synthetic dataset, fits all five models, and asserts that matrix
factorisation beats the global-mean baseline and that every recommender returns
sensible output.

## Suggested next steps
- **Hybrid model**: blend content-based + matrix-factorisation scores for cold-start users.
- **Implicit feedback**: treat "rated" as a signal and try ALS / BPR.
- **Better MF**: swap the hand-rolled SGD for `scikit-surprise` (SVD/SVD++) or `implicit`.
- **Poster art**: use the `links.csv` TMDB IDs to pull poster thumbnails into the cards.
```
