"""Reusable data-prep helpers: score unification, outcome labels, game-level splitting.

These live here (rather than only in a notebook) so the same logic can be
imported by later notebooks/scripts without copy-pasting cells.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MATE_SCORE = 10000
SCORE_CAP = 2000


def unify_score(df: pd.DataFrame, mate_score: int = MATE_SCORE, cap: int = SCORE_CAP) -> pd.Series:
    """Combine `score` (centipawns) and `mate` (moves to forced mate) into one
    continuous, signed evaluation column.

    - Non-mate positions: `score` clipped to +/- `cap`.
    - Mate positions: sign(mate) * (mate_score - |mate|), so any forced mate
      is always more extreme than any non-mate eval, and shorter mates are
      more extreme than longer ones.
    """
    clipped_score = df["score"].clip(-cap, cap)
    mate_value = np.sign(df["mate"]) * (mate_score - df["mate"].abs())
    return clipped_score.where(df["score"].notna(), mate_value)


def add_outcome_label(df: pd.DataFrame, drop_draws: bool = True) -> pd.DataFrame:
    """Derive a binary `white_win` label (1 = white won, 0 = black won) from
    `white_result` (values are the strings '1', '0', '1/2').

    By default draws are dropped, matching the binary classifier we're
    starting with. Set drop_draws=False to keep them (they'll need their own
    handling downstream — e.g. a 3-class target).
    """
    df = df.copy()
    is_draw = df["white_result"] == "1/2"

    if drop_draws:
        df = df[~is_draw]
    else:
        df = df[df["white_result"].isin(["0", "1", "1/2"])]

    df["white_win"] = np.where(
        df["white_result"] == "1", 1,
        np.where(df["white_result"] == "0", 0, np.nan),
    )
    return df


def game_level_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_on_outcome: bool = True,
):
    """Split by game_id (not by row) so every position from a given game ends
    up entirely in train or entirely in test — avoids leaking correlated
    positions from the same game across the split.

    stratify_on_outcome=True keeps the win/loss balance similar between train
    and test by stratifying on each game's final white_win label.
    """
    games = df.drop_duplicates("game_id")[["game_id", "white_win"]]

    stratify = games["white_win"] if stratify_on_outcome else None
    train_ids, test_ids = train_test_split(
        games["game_id"], test_size=test_size, random_state=random_state, stratify=stratify
    )

    train_df = df[df["game_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["game_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df
