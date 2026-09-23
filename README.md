````markdown
# Chess Position Evaluator

A from-scratch data science project exploring machine learning on chess positions: predicting game outcomes from a board position, and predicting Stockfish's own evaluation of that position. Built as a learning project to work through the full ML pipeline — data engineering, feature engineering, model comparison, evaluation, and diagnostic ablation studies — using real games rather than a toy dataset.

## Overview

Using ~1M Lichess game positions (with Stockfish evaluations, player Elo, and game metadata), this project answers two questions:

1. **Classification** — given a board position, does white or black win the game?
2. **Regression** — given a board position, what evaluation (in centipawns) would Stockfish assign it?

Rather than feeding raw board state into a black-box model, every feature is hand-engineered from the FEN string using `python-chess` — material balance, mobility, king safety, pawn structure, pins, and more — so the project doubles as a study of which classical chess-evaluation concepts actually carry predictive signal.

## Key results

- **Classification (board features only):** ~66.1% test accuracy / 0.70 F1 with gradient boosting, up from a 64.9% logistic regression baseline — improvement driven almost entirely by richer features (pawn structure, king safety, pins), not by model choice or hyperparameter tuning.
- **Elo ablation:** adding player Elo as a feature lifts accuracy to ~72.5% — a much bigger jump than any board feature — which says more about player skill than board understanding. Reported separately from the board-only model rather than blended into one headline number.
- **Regression:** ~0.70 R² predicting Stockfish's evaluation with tree-based models, versus ~0.05 for plain linear regression — the eval landscape is highly nonlinear. Mate-position outliers (forced-mate scores pushed far outside the normal centipawn range) actively hurt the fit unless excluded from training.
- Model comparisons cover logistic/linear regression, decision trees, random forest, `HistGradientBoostingClassifier`/`Regressor`, and XGBoost.

## Project structure

```
chess-position-evaluator/
├── data/
│   ├── raw/          # downloaded Kaggle dataset (gitignored)
│   └── processed/    # train/test splits with engineered features (gitignored)
├── notebooks/         # numbered, run in order — see below
├── src/
│   ├── download_data.py   # pulls the dataset via kagglehub
│   ├── data.py             # score/mate unification, outcome labels, game-level train/test split
│   └── features.py         # FEN -> feature vector (three progressively richer versions)
├── requirements.txt
└── .gitignore
```

## Dataset

[`nikitricky/chess-positions`](https://www.kaggle.com/datasets/nikitricky/chess-positions) on Kaggle — ~1.09M positions sampled from real Lichess games, each with FEN, the move played, Stockfish score/mate/depth, game metadata (players, Elo, opening, time control, termination), and the game's final result.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
```

`download_data.py` uses `kagglehub`, which can fetch this (public) dataset without credentials. For private datasets, Kaggle auth goes in `~/.kaggle/access_token` (see Kaggle account settings → API).

## Notebooks

Run in order — each one builds on artifacts saved by the last:

| Notebook | What it does |
|---|---|
| `01_eda.ipynb` / `01_eda_practice.ipynb` | Initial exploration of the raw dataset: score/mate structure, draw encoding, Elo data quality, positions-per-game |
| `02_labels_and_split.ipynb` | Unifies `score`/`mate` into one continuous eval (`eval_cp`), derives the binary outcome label, and splits train/test by `game_id` (not by row) to avoid leaking correlated positions across the split |
| `03_feature_engineering.ipynb` | First feature set (24 features): material, mobility, castling rights, center control |
| `04_baseline_logreg.ipynb` | Baseline logistic regression, multicollinearity diagnosis, L1/L2 regularization experiments, class-imbalance investigation |
| `05_model_comparison.ipynb` | Decision tree / random forest / gradient boosting comparison against the logistic regression baseline |
| `06_rich_feature_engineering.ipynb` | Second feature set (+10 features): pawn structure, king safety, piece activity, hanging pieces |
| `07_model_comparison_v2.ipynb` | Model comparison on the richer feature set; feature-pruning and split-criterion experiments |
| `08_targeted_features.ipynb` | Third feature set (+5 features): king-attack pressure, pins, game-phase (endgame) detection |
| `09_model_comparison_v3.ipynb` | Model comparison incl. XGBoost; the Elo ablation experiment (board-only vs. board+Elo) |
| `10_regression_eval.ipynb` | Pivots to regression: predicting `eval_cp` directly, including the mate-position outlier experiment |

## Notes on methodology

- **Leakage-safe splitting**: all train/test splits are done at the `game_id` level, since positions from the same game are highly correlated.
- **Feature versions are additive and separately importable** (`extract_features`, `extract_features_v2`, `extract_features_v3` in `src/features.py`), so any notebook can be re-run against a specific feature-set generation.
- **Ablation over assumption**: several results in this project (multicollinearity's effect on accuracy, whether L1 regularization actually zeroes out redundant features, whether mate positions hurt regression, how much Elo alone explains) were tested directly rather than assumed — see the relevant notebooks for the experiment and result side by side.

## Possible next steps

- Multi-class move prediction (predict the next move given a position) — a natural extension noted early on, likely better suited to a CNN/board-tensor representation than hand-engineered features
- From-scratch gradient descent implementation (batch/SGD/mini-batch) alongside sklearn's optimizers, for the optimization-technique learning goal that motivated this project
````
