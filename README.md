# What Makes a Song Popular on Spotify?
**CIS 2450 — Data Science | University of Pennsylvania | Spring 2026**
**Solo Project | Thakia Waksanga**

---

## Overview

This project investigates whether a song's audio features can predict its Spotify popularity score. Using a dataset of 114,000 tracks, I built and compared six regression models — finding that a tuned HistGradientBoosting model explains ~39% of popularity variance from audio features alone. The most important finding: popularity is driven more by what a song *isn't* (instrumental, acoustic) than what it is.

---

## Results

| Model | RMSE | R² |
|---|---|---|
| HistGradientBoosting (tuned) ★ | 15.91 | 0.394 |
| LinearRegression (baseline) | 16.83 | 0.322 |
| RandomForest (tuned) | 17.39 | 0.276 |
| Lasso | 18.46 | 0.185 |
| ElasticNet | 19.47 | 0.092 |

---

## Difficulty Concepts

1. **Permutation Feature Importance** — robust importance ranking using mean R² decrease
2. **Hyperparameter Tuning** — RandomizedSearchCV with 5-fold cross-validation
3. **Feature Engineering** — `energy_dance`, `log_duration`, `acoustic_instrumental`

---

## Data Sources

- **Kaggle** — [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (114k tracks, 21 columns)
- **Spotify Web API** — authenticated via `spotipy` (OAuth 2.0 client credentials)

> The dataset CSV is not included in this repo due to file size. Download it from the Kaggle link above and place it in the root directory as `dataset.csv`.

---

## How to Run the Dashboard

```bash
pip install streamlit plotly pandas numpy
streamlit run dashboard.py
```

Then open **http://localhost:8501** in your browser.

The dashboard has five pages: Overview, EDA, Model Results, Feature Importance, and an interactive Popularity Predictor.

---

## Files

```
├── cs2450_spotify_v3.ipynb   # Main analysis notebook
├── dashboard.py              # Streamlit dashboard
└── README.md
and videos
```
