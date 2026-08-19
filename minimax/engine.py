"""
Depth-limited minimax chess engine with alpha-beta pruning.

Features:
  - Negamax with alpha-beta pruning
  - Iterative deepening
  - Move ordering (MVV-LVA, promotions, TT-ish killer heuristic)
  - Quiescence search to avoid the horizon effect
  - Piece-square table evaluation

Self-contained: only depends on `python-chess`.
"""
import chess
import time

# ---------------------------------------------------------------------------
# Piece values and piece-square tables (from white's perspective, index 0 = a8)
# ---------------------------------------------------------------------------

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Classic PSTs (white's perspective, row 0 = rank 8).
_PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
   50, 50, 50, 50, 50, 50, 50, 50,
   10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0,
]

_KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

_BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

_ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

_QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

_KING_MIDDLE_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

_KING_END_TABLE = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

PST = {
    chess.PAWN: _PAWN_TABLE,
    chess.KNIGHT: _KNIGHT_TABLE,
    chess.BISHOP: _BISHOP_TABLE,
    chess.ROOK: _ROOK_TABLE,
    chess.QUEEN: _QUEEN_TABLE,
    chess.KING: _KING_MIDDLE_TABLE,
}

# Mate / stalemate scores.
MATE = 100000
MATE_THRESHOLD = MATE - 1000


def _pst_index(square, color):
    """PST index from white's perspective (a8 = 0)."""
    if color == chess.WHITE:
        return square
    return square ^ 56  # mirror rank


def evaluate(board):
    """Static evaluation in centipawns from white's perspective."""
    if board.is_checkmate():
        return -MATE + board.ply()
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue
            idx = _pst_index(sq, color)
            table = PST[piece.piece_type]
            if piece.piece_type == chess.KING:
                # Blend middle and endgame king tables based on material left.
                non_pawn = sum(
                    1 for s in chess.SQUARES
                    if board.piece_at(s) and board.piece_at(s).piece_type in
                    (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
                )
                table = _blend_king_tables(table, non_pawn)
            score += sign * (PIECE_VALUE[piece.piece_type] + table[idx])

    return score


def _blend_king_tables(mid_table, non_pawn_material):
    if non_pawn_material <= 1:
        return _KING_END_TABLE
    return mid_table


# ---------------------------------------------------------------------------
# Move ordering
# ---------------------------------------------------------------------------

# MVV-LVA: most valuable victim, least valuable attacker.
_VICTIM_SCORE = {
    chess.PAWN: 100,
    chess.KNIGHT: 200,
    chess.BISHOP: 300,
    chess.ROOK: 400,
    chess.QUEEN: 500,
    chess.KING: 600,
}


def move_score(board, move, killers):
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            score += 10 * _VICTIM_SCORE[victim.piece_type] - _VICTIM_SCORE[attacker.piece_type]
        else:
            score += _VICTIM_SCORE[chess.PAWN]
    if move.promotion:
        score += PIECE_VALUE[chess.QUEEN]
    if move in killers:
        score += 3000
    if board.is_castling(move):
        score += 1000
    return score


def order_moves(board, moves, killers):
    scored = [(move_score(board, m, killers), m) for m in moves]
    scored.sort(key=lambda x: -x[0])
    return [m for _, m in scored]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class SearchResult:
    __slots__ = ("move", "score", "nodes", "depth", "time", "pv")

    def __init__(self, move, score, nodes, depth, time, pv):
        self.move = move
        self.score = score
        self.nodes = nodes
        self.depth = depth
        self.time = time
        self.pv = pv

    def __repr__(self):
        return (f"best={self.move.uci() if self.move else None} score={self.score} "
                f"depth={self.depth} nodes={self.nodes} time={self.time:.2f}s")


class Engine:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.nodes = 0
        self.killers = []
        self.depth_limit = 0
        self.deadline = None
        self.pv = []

    # -- entry point -------------------------------------------------------

    def best_move(self, board, depth=None, movetime=None, pv=False):
        """Find the best move. `depth` overrides max_depth; `movetime` caps search time."""
        self.nodes = 0
        self.killers = []
        self.depth_limit = depth or self.max_depth
        self.deadline = time.time() + movetime if movetime else None
        self.pv = []

        best = None
        best_score = -MATE
        start = time.time()
        best_pv = []
        for d in range(1, self.depth_limit + 1):
            if self.deadline and time.time() > self.deadline:
                break
            score, move, pv = self._search(board, d, -MATE, MATE, 1)
            if move is not None:
                best = move
                best_score = score
                best_pv = pv
        return SearchResult(best, best_score, self.nodes, self.depth_limit,
                            time.time() - start, best_pv)

    # -- core negamax -------------------------------------------------------

    def _search(self, board, depth, alpha, beta, ply):
        self.nodes += 1

        if board.is_checkmate():
            return -MATE + ply, None, []
        if board.is_stalemate() or board.is_insufficient_material():
            return 0, None, []

        # Quiescence search at the horizon.
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply), None, []

        legal = list(board.legal_moves)
        if not legal:
            return 0, None, []

        ordered = order_moves(board, legal, self.killers[:2])
        best_move = None
        best_score = -MATE
        best_pv = []
        for move in ordered:
            board.push(move)
            score, _, child_pv = self._search(board, depth - 1, -beta, -alpha, ply + 1)
            score = -score
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
                best_pv = [move] + child_pv
            if score > alpha:
                alpha = score
            if alpha >= beta:
                # Beta cutoff -> save killer move.
                if not board.is_capture(move) and not move.promotion:
                    if move not in self.killers:
                        self.killers.insert(0, move)
                        self.killers = self.killers[:2]
                break

        if best_pv:
            self.pv = best_pv
        return best_score, best_move, best_pv

    # -- quiescence search --------------------------------------------------

    def _quiescence(self, board, alpha, beta, ply):
        """Search only captures and promotions to avoid the horizon effect."""
        self.nodes += 1
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        for move in order_moves(board, (m for m in board.legal_moves
                                        if board.is_capture(m) or m.promotion), []):
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha


# ---------------------------------------------------------------------------
# CLI / play
# ---------------------------------------------------------------------------

def play_game(depth=4, time_limit=None):
    """Play a game against the engine from the terminal."""
    board = chess.Board()
    engine = Engine(max_depth=depth)
    print("Minimax + alpha-beta engine. You are White. Enter moves as UCI (e.g. e2e4).")
    print(board)
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move_uci = input("Your move: ").strip().lower()
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                print("Invalid UCI. Try again.")
                continue
            if move not in board.legal_moves:
                print("Illegal move. Try again.")
                continue
        else:
            result = engine.best_move(board, movetime=time_limit)
            move = result.move
            print(f"Engine: {move.uci()} (score {result.score / 100:.2f}, "
                  f"{result.nodes:,} nodes, depth {result.depth})")
        board.push(move)
        print(board)
        print()
    print("Game over:", board.result())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--play":
        depth = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        play_game(depth=depth)
    else:
        # Self-test: play a short game engine vs engine at depth 2.
        board = chess.Board()
        e = Engine(max_depth=2)
        moves = []
        while not board.is_game_over() and len(moves) < 40:
            r = e.best_move(board)
            moves.append(r.move)
            board.push(r.move)
        print(f"Self-test: {len(moves)} moves, result {board.result()}")
        print("Moves:", " ".join(m.uci() for m in moves))
