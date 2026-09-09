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


# ---------------------------------------------------------------------------
# Richer, second-pass features: pawn structure, king safety, development,
# and tactical exposure. Kept separate from extract_features() so the
# original feature set (and every notebook/result built on it) stays
# reproducible; extract_features_v2() builds on top of it.
# ---------------------------------------------------------------------------

STARTING_MINOR_SQUARES = {
    chess.WHITE: [chess.B1, chess.C1, chess.F1, chess.G1],
    chess.BLACK: [chess.B8, chess.C8, chess.F8, chess.G8],
}


def _pawn_files(board: chess.Board, color: bool) -> list[int]:
    return [chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)]


def _doubled_pawns(board: chess.Board, color: bool) -> int:
    """Extra pawns stacked on the same file, beyond the first on that file."""
    files = _pawn_files(board, color)
    counts = {f: files.count(f) for f in set(files)}
    return sum(c - 1 for c in counts.values() if c > 1)


def _isolated_pawns(board: chess.Board, color: bool) -> int:
    """Pawns with no friendly pawn on an adjacent file to support them."""
    files = _pawn_files(board, color)
    file_set = set(files)
    return sum(1 for f in files if (f - 1) not in file_set and (f + 1) not in file_set)


def _passed_pawns(board: chess.Board, color: bool) -> int:
    """Pawns with no enemy pawn on the same or an adjacent file ahead of them
    (i.e. nothing standing between them and promotion)."""
    own_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = list(board.pieces(chess.PAWN, not color))
    count = 0
    for sq in own_pawns:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        blocked = any(
            abs(chess.square_file(e) - f) <= 1
            and (chess.square_rank(e) > r if color == chess.WHITE else chess.square_rank(e) < r)
            for e in enemy_pawns
        )
        if not blocked:
            count += 1
    return count


def _king_danger(board: chess.Board, color: bool) -> int:
    """How many squares immediately around the king are attacked by the enemy."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    neighbors = [sq for sq in chess.SQUARES if chess.square_distance(sq, king_sq) == 1]
    return sum(board.is_attacked_by(not color, sq) for sq in neighbors)


def _pawn_shield(board: chess.Board, color: bool) -> int:
    """Own pawns on the rank directly in front of the king, in the 3 files
    spanning the king's position — a rough castled-safety proxy."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    f, r = chess.square_file(king_sq), chess.square_rank(king_sq)
    shield_rank = r + 1 if color == chess.WHITE else r - 1
    if not (0 <= shield_rank <= 7):
        return 0
    count = 0
    for df in (-1, 0, 1):
        ff = f + df
        if 0 <= ff <= 7:
            piece = board.piece_at(chess.square(ff, shield_rank))
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                count += 1
    return count


def _rooks_on_open_files(board: chess.Board, color: bool) -> int:
    """Rooks on files with no pawn of their own color (open or semi-open file)."""
    own_pawn_files = set(_pawn_files(board, color))
    return sum(1 for sq in board.pieces(chess.ROOK, color) if chess.square_file(sq) not in own_pawn_files)


def _undeveloped_minors(board: chess.Board, color: bool) -> int:
    """Knights/bishops still sitting on their starting squares."""
    count = 0
    for sq in STARTING_MINOR_SQUARES[color]:
        piece = board.piece_at(sq)
        if piece and piece.color == color and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            count += 1
    return count


def _hanging_pieces(board: chess.Board, color: bool) -> int:
    """Own non-king pieces that are attacked and not defended by any own piece."""
    count = 0
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type != chess.KING:
            if board.is_attacked_by(not color, sq) and not board.is_attacked_by(color, sq):
                count += 1
    return count


def _advanced_pawns(board: chess.Board, color: bool) -> int:
    """Pawns that have crossed the board's midline (a simple space measure)."""
    pawns = board.pieces(chess.PAWN, color)
    if color == chess.WHITE:
        return sum(1 for sq in pawns if chess.square_rank(sq) >= 4)
    return sum(1 for sq in pawns if chess.square_rank(sq) <= 3)


def extract_features_v2(fen: str) -> dict:
    """extract_features() plus pawn structure, king safety, development, and
    tactical-exposure features. All new features are expressed as
    white-minus-black diffs, oriented so positive = better for white,
    matching the sign convention of material_diff/mobility_diff/etc."""
    board = chess.Board(fen)
    features = extract_features(fen)

    white_bishop_pair = int(len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2)
    black_bishop_pair = int(len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2)
    features["bishop_pair_diff"] = white_bishop_pair - black_bishop_pair

    features["doubled_pawns_diff"] = _doubled_pawns(board, chess.BLACK) - _doubled_pawns(board, chess.WHITE)
    features["isolated_pawns_diff"] = _isolated_pawns(board, chess.BLACK) - _isolated_pawns(board, chess.WHITE)
    features["passed_pawns_diff"] = _passed_pawns(board, chess.WHITE) - _passed_pawns(board, chess.BLACK)

    # king_danger counts attacked squares around the king, so *lower* is safer —
    # flip the subtraction order so positive still means "better for white".
    features["king_safety_diff"] = _king_danger(board, chess.BLACK) - _king_danger(board, chess.WHITE)
    features["pawn_shield_diff"] = _pawn_shield(board, chess.WHITE) - _pawn_shield(board, chess.BLACK)
    features["rook_open_files_diff"] = _rooks_on_open_files(board, chess.WHITE) - _rooks_on_open_files(board, chess.BLACK)
    features["undeveloped_minors_diff"] = _undeveloped_minors(board, chess.BLACK) - _undeveloped_minors(board, chess.WHITE)
    features["hanging_pieces_diff"] = _hanging_pieces(board, chess.BLACK) - _hanging_pieces(board, chess.WHITE)
    features["advanced_pawns_diff"] = _advanced_pawns(board, chess.WHITE) - _advanced_pawns(board, chess.BLACK)

    return features


def add_features_v2(df: pd.DataFrame, fen_col: str = "fen") -> pd.DataFrame:
    """Same as add_features(), using the richer extract_features_v2()."""
    feature_rows = df[fen_col].apply(extract_features_v2)
    feature_df = pd.DataFrame(list(feature_rows), index=df.index)
    return pd.concat([df, feature_df], axis=1)


# ---------------------------------------------------------------------------
# Third pass: targeted king-attack pressure, pins, and game-phase features.
# Built on extract_features_v2() the same way v2 was built on v1.
# ---------------------------------------------------------------------------

NON_PAWN_VALUES = {pt: v for pt, v in PIECE_VALUES.items() if pt != chess.PAWN}
STARTING_TOTAL_NON_PAWN_MATERIAL = 2 * sum(NON_PAWN_VALUES.values())  # both sides, full set

# standard chess-programming convention: a side is in the endgame once its own
# non-pawn material drops to roughly a queen+rook or queen+minor (~13 points)
# or less. This must be checked *per side*, not on the combined total — a
# combined threshold would misfire when one side is stripped down but the
# other still has nearly everything.
PER_SIDE_ENDGAME_THRESHOLD = 13


def _king_zone(board: chess.Board, king_color: bool) -> list[int]:
    king_sq = board.king(king_color)
    if king_sq is None:
        return []
    return [sq for sq in chess.SQUARES if chess.square_distance(sq, king_sq) == 1]


def _attackers_near_king(board: chess.Board, king_color: bool, attacker_color: bool) -> int:
    """Count of *unique attacking pieces* (not attacked squares) of attacker_color
    bearing on the 8 squares surrounding king_color's king. Complements
    king_safety_diff (which counts attacked squares — a single queen can cover
    several) by measuring how much attacking force is actually present."""
    attacker_squares: set[int] = set()
    for sq in _king_zone(board, king_color):
        attacker_squares |= set(board.attackers(attacker_color, sq))
    return len(attacker_squares)


def _pinned_count(board: chess.Board, color: bool) -> int:
    """Own pieces that are pinned to their king (can't move without exposing it)."""
    return sum(
        1 for sq, piece in board.piece_map().items()
        if piece.color == color and board.is_pinned(color, sq)
    )


def _non_pawn_material(board: chess.Board, color: bool) -> int:
    return sum(NON_PAWN_VALUES[pt] * len(board.pieces(pt, color)) for pt in NON_PAWN_VALUES)


def extract_features_v3(fen: str) -> dict:
    """extract_features_v2() plus targeted king-attack pressure, pins, and
    game-phase features."""
    board = chess.Board(fen)
    features = extract_features_v2(fen)

    white_attackers_on_black_king = _attackers_near_king(board, chess.BLACK, chess.WHITE)
    black_attackers_on_white_king = _attackers_near_king(board, chess.WHITE, chess.BLACK)
    features["attackers_near_king_diff"] = white_attackers_on_black_king - black_attackers_on_white_king

    # more of *your own* pieces pinned is bad for you, so black_pins - white_pins
    # keeps positive = better for white.
    features["pins_diff"] = _pinned_count(board, chess.BLACK) - _pinned_count(board, chess.WHITE)

    white_non_pawn = _non_pawn_material(board, chess.WHITE)
    black_non_pawn = _non_pawn_material(board, chess.BLACK)
    features["total_non_pawn_material"] = white_non_pawn + black_non_pawn
    features["is_endgame"] = int(
        white_non_pawn <= PER_SIDE_ENDGAME_THRESHOLD and black_non_pawn <= PER_SIDE_ENDGAME_THRESHOLD
    )

    # explicit interaction term: mainly useful for logistic regression, which
    # can't discover "material matters more in the endgame" on its own the
    # way a tree-based model can via sequential splits.
    features["material_diff_x_endgame"] = features["material_diff"] * features["is_endgame"]

    return features


def add_features_v3(df: pd.DataFrame, fen_col: str = "fen") -> pd.DataFrame:
    """Same as add_features(), using the further-enriched extract_features_v3()."""
    feature_rows = df[fen_col].apply(extract_features_v3)
    feature_df = pd.DataFrame(list(feature_rows), index=df.index)
    return pd.concat([df, feature_df], axis=1)
