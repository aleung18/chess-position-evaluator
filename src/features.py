"""Turn a FEN string into a numeric feature vector for classical ML models.

Feature set is intentionally simple/interpretable for a first pass:
- material counts per piece type, per side, and total material_diff
- side to move, castling rights
- mobility (legal move count) for each side, and the diff
- center-square control, and the diff
- whether the side to move is in check
- move number
"""

import chess
import pandas as pd

PIECE_TYPES = [
    (chess.PAWN, "pawns"),
    (chess.KNIGHT, "knights"),
    (chess.BISHOP, "bishops"),
    (chess.ROOK, "rooks"),
    (chess.QUEEN, "queens"),
]
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
CENTER_SQUARES = [chess.D4, chess.D5, chess.E4, chess.E5]


def _mobility(board: chess.Board) -> tuple[int, int]:
    """Legal move count for white and for black, regardless of whose turn it
    actually is (uses a null move to flip sides without making a real move)."""
    if board.turn == chess.WHITE:
        white_mobility = board.legal_moves.count()
        board.push(chess.Move.null())
        black_mobility = board.legal_moves.count()
        board.pop()
    else:
        black_mobility = board.legal_moves.count()
        board.push(chess.Move.null())
        white_mobility = board.legal_moves.count()
        board.pop()
    return white_mobility, black_mobility


def extract_features(fen: str) -> dict:
    board = chess.Board(fen)
    features = {}

    for color, prefix in [(chess.WHITE, "white"), (chess.BLACK, "black")]:
        for piece_type, name in PIECE_TYPES:
            features[f"{prefix}_{name}"] = len(board.pieces(piece_type, color))

    features["material_diff"] = sum(
        PIECE_VALUES[pt] * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
        for pt in PIECE_VALUES
    )

    features["side_to_move_white"] = int(board.turn == chess.WHITE)
    features["white_kingside_castle"] = int(board.has_kingside_castling_rights(chess.WHITE))
    features["white_queenside_castle"] = int(board.has_queenside_castling_rights(chess.WHITE))
    features["black_kingside_castle"] = int(board.has_kingside_castling_rights(chess.BLACK))
    features["black_queenside_castle"] = int(board.has_queenside_castling_rights(chess.BLACK))

    white_mobility, black_mobility = _mobility(board)
    features["mobility_white"] = white_mobility
    features["mobility_black"] = black_mobility
    features["mobility_diff"] = white_mobility - black_mobility

    white_center = sum(board.is_attacked_by(chess.WHITE, sq) for sq in CENTER_SQUARES)
    black_center = sum(board.is_attacked_by(chess.BLACK, sq) for sq in CENTER_SQUARES)
    features["white_center_control"] = white_center
    features["black_center_control"] = black_center
    features["center_control_diff"] = white_center - black_center

    features["in_check"] = int(board.is_check())
    features["fullmove_number"] = board.fullmove_number

    return features


def add_features(df: pd.DataFrame, fen_col: str = "fen") -> pd.DataFrame:
    """Return a copy of df with one new column per feature, extracted from fen_col."""
    feature_rows = df[fen_col].apply(extract_features)
    feature_df = pd.DataFrame(list(feature_rows), index=df.index)
    return pd.concat([df, feature_df], axis=1)  
