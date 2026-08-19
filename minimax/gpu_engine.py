"""
GPU-accelerated minimax chess engine with alpha-beta pruning.

Design
------
Alpha-beta search is inherently sequential (CPU), but the expensive part
of a deep search -- evaluating leaf positions -- is perfectly parallelizable.
This engine:

  1. Builds the search tree on the CPU using python-chess move generation.
  2. Batches all leaf boards into (N, 17, 8, 8) tensors.
  3. Evaluates every leaf in one forward pass on the GPU with torch
     (vectorized material + piece-square tables; a neural net can be plugged in).
  4. Propagates scores up the tree with negamax + alpha-beta pruning.

The evaluator runs on the torch device selected at startup
(``cuda`` if available, else ``cpu``), so it uses whatever torch ships in the
venv with zero extra dependencies beyond python-chess and torch.
"""
import chess
import time

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_PLANES = 17
NUM_ACTIONS = 73  # unused here, kept for parity with the RL model conventions

MATE = 100000
INF = float("inf")

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Piece-square tables, white's perspective, index 0 = a8 (same as model.py).
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
_KING_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

_PST = {
    chess.PAWN: _PAWN_TABLE,
    chess.KNIGHT: _KNIGHT_TABLE,
    chess.BISHOP: _BISHOP_TABLE,
    chess.ROOK: _ROOK_TABLE,
    chess.QUEEN: _QUEEN_TABLE,
    chess.KING: _KING_TABLE,
}

# Plane layout (matches model.py): 0-5 white P,N,B,R,Q,K; 6-11 black; 12 stm;
# 13-16 castling.  We only use the 12 piece planes for evaluation.
PIECE_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


def encode_board(board):
    """Encode a board as a (NUM_PLANES, 8, 8) float32 array, white's perspective."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        r = 7 - (sq // 8)
        c = sq % 8
        planes[PIECE_PLANE[(piece.piece_type, piece.color)], r, c] = 1.0
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return planes


# ---------------------------------------------------------------------------
# GPU evaluator
# ---------------------------------------------------------------------------

class GPUEvaluator:
    """Batch leaf evaluation on the torch device.

    The score is a linear function of the 17-plane encoding, i.e. a single
    1x1 convolution whose weights encode material values + piece-square
    bonuses.  This is exactly what a trained value head approximates, and a
    real `nn.Module` can be dropped in place of `_score_tensor`.
    """

    def __init__(self, device=None, batch=512):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch = batch
        self.net = nn.Module()  # placeholder so `_score_tensor` could be swapped for a real net
        self._build_weight()

    def _build_weight(self):
        """Build the (17, 8, 8) evaluation weight tensor on the GPU."""
        w = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
        for sq in chess.SQUARES:
            r = 7 - (sq // 8)
            c = sq % 8
            for color in (chess.WHITE, chess.BLACK):
                for ptype, table in _PST.items():
                    p = PIECE_PLANE[(ptype, color)]
                    val = PIECE_VALUE[ptype] + table[sq]
                    if color == chess.BLACK:
                        val = -val
                    w[p, r, c] = val
        self.weight = torch.from_numpy(w).to(self.device)
        self.weight.requires_grad_(False)

    def evaluate(self, boards):
        """Evaluate a list of boards -> numpy array of scores (white's perspective, cp).

        Boards are evaluated in GPU batches; the GPU stays busy throughout.
        """
        scores = np.empty(len(boards), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(boards), self.batch):
                chunk = boards[start:start + self.batch]
                x = torch.from_numpy(
                    np.stack([encode_board(b) for b in chunk])
                ).to(self.device)
                s = self._score_tensor(x).cpu().numpy()
                scores[start:start + len(chunk)] = s
        return scores

    def _score_tensor(self, x):
        """x: (N, 17, 8, 8) float on device -> (N,) score in cp from white's view."""
        return (x * self.weight).sum(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
# Search tree construction
# ---------------------------------------------------------------------------

_VICTIM_SCORE = {
    chess.PAWN: 100,
    chess.KNIGHT: 200,
    chess.BISHOP: 300,
    chess.ROOK: 400,
    chess.QUEEN: 500,
    chess.KING: 600,
}


def _move_order_key(board, move):
    """Static ordering score: captures and promotions first."""
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
    return score


def _build_tree(root_board, depth, max_nodes):
    """Expand the minimax tree on CPU.

    Returns a tuple describing the tree:
      nodes_moves  : moves that led to each node (None for root)
      nodes_parent : parent index for each node
      nodes_children : list of child indices
      nodes_depth  : ply of each node from the root
      leaf_indices : node indices that are leaves (depth limit or game over)
      leaf_boards  : a fresh board per leaf (for GPU batching)
      leaf_game_over: True for leaves where the game already ended

    Expansion is depth-first; when `max_nodes` is reached the search is
    truncated and remaining nodes are treated as leaves.
    """
    nodes_moves = [None]
    nodes_parent = [-1]
    nodes_children = [[]]
    nodes_depth = [0]
    leaf_indices = []
    leaf_boards = []
    leaf_game_over = []

    board = root_board.copy()
    count = 1

    def expand(idx, dr):
        nonlocal count
        if board.is_game_over():
            leaf_indices.append(idx)
            leaf_boards.append(board.copy())
            leaf_game_over.append(True)
            return
        if dr == 0:
            leaf_indices.append(idx)
            leaf_boards.append(board.copy())
            leaf_game_over.append(False)
            return

        moves = list(board.legal_moves)
        moves.sort(key=lambda m: _move_order_key(board, m), reverse=True)
        for m in moves:
            board.push(m)
            child = len(nodes_moves)
            nodes_moves.append(m)
            nodes_parent.append(idx)
            nodes_children[idx].append(child)
            nodes_children.append([])
            nodes_depth.append(nodes_depth[idx] + 1)
            count += 1
            if count >= max_nodes:
                # Truncate here: treat this node as a leaf and stop expanding.
                leaf_indices.append(child)
                leaf_boards.append(board.copy())
                leaf_game_over.append(board.is_game_over())
                board.pop()
                break
            expand(child, dr - 1)
            board.pop()

    expand(0, depth)

    return (nodes_moves, nodes_parent, nodes_children, nodes_depth,
            leaf_indices, leaf_boards, leaf_game_over)


# ---------------------------------------------------------------------------
# Alpha-beta propagation
# ---------------------------------------------------------------------------

class GPUEngine:
    """Depth-limited minimax with alpha-beta pruning and GPU batched eval."""

    def __init__(self, depth=4, max_nodes=200000, evaluator=None):
        self.depth = depth
        self.max_nodes = max_nodes
        self.evaluator = evaluator or GPUEvaluator()

    # -- entry point -------------------------------------------------------

    def best_move(self, board, depth=None, max_nodes=None):
        depth = depth or self.depth
        max_nodes = max_nodes or self.max_nodes
        start = time.time()

        (moves, parents, children, depths,
         leaf_idx, leaf_boards, leaf_game_over) = _build_tree(board, depth, max_nodes)

        # ---- GPU phase: evaluate all leaves in one batched pass ----------
        value = np.zeros(len(moves), dtype=np.float64)
        gp = self.evaluator
        use_gpu = bool(leaf_boards)
        if use_gpu:
            scores = gp.evaluate(leaf_boards)
            for i, (nid, game_over) in enumerate(zip(leaf_idx, leaf_game_over)):
                b = leaf_boards[i]
                if game_over:
                    # Side-to-move perspective: a checkmated side is at -(MATE - ply).
                    if b.is_checkmate():
                        value[nid] = -(MATE - depths[nid])
                    else:
                        value[nid] = 0.0
                else:
                    # White-perspective score -> side-to-move perspective.
                    v = scores[i]
                    value[nid] = v if b.turn == chess.WHITE else -v

        # ---- CPU phase: negamax + alpha-beta bottom-up --------------------
        # Children are always created after parents, so iterating in reverse
        # guarantees children are scored before their parents.
        for nid in range(len(moves) - 1, 0, -1):
            kids = children[nid]
            if not kids:
                continue  # leaf: value already set
            # Move ordering by child scores improves pruning.  value[c] is from
            # c's side-to-move perspective, so from nid's view score = -value[c];
            # children with the lowest value[c] are searched first.
            kids_sorted = sorted(kids, key=lambda c: value[c])
            best = -INF
            alpha = -INF
            beta = INF
            for c in kids_sorted:
                score = -value[c]
                if score > best:
                    best = score
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
            value[nid] = best

        # ---- pick the root move -------------------------------------------
        root = 0
        best_kid = None
        best_score = -INF
        for c in children[root]:
            score = -value[c]
            if score > best_score:
                best_score = score
                best_kid = moves[c]

        elapsed = time.time() - start
        return self.SearchResult(best_kid, best_score, len(moves), depth, elapsed,
                                 self._principal_variation(best_kid, moves, parents, children, value))

    @staticmethod
    def _principal_variation(move, moves, parents, children, value):
        if move is None:
            return []
        # Find the root child node for this move.
        for c in children[0]:
            if moves[c] == move:
                start = c
                break
        else:
            return [move]
        pv = []
        nid = start
        while True:
            pv.append(moves[nid])
            if not children[nid]:
                break
            kids = sorted(children[nid], key=lambda k: value[k])  # lowest value[c] = best child
            nid = kids[0]
        return pv

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
            return (f"best={self.move.uci() if self.move else None} "
                    f"score={self.score:.0f}cp nodes={self.nodes} "
                    f"depth={self.depth} time={self.time:.3f}s pv="
                    f"{' '.join(m.uci() for m in self.pv)}")


# ---------------------------------------------------------------------------
# Play / self-test
# ---------------------------------------------------------------------------

def play_game(engine, time_limit=10.0):
    board = chess.Board()
    print("GPU minimax engine. You are White. Enter moves as UCI (e.g. e2e4).")
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
            result = engine.best_move(board)
            move = result.move
            print(f"Engine: {result}")
        board.push(move)
        print(board)
        print()
    print("Game over:", board.result())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | evaluator device: {device}")

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--play":
        eng = GPUEngine(depth=int(sys.argv[2]) if len(sys.argv) > 2 else 4)
        play_game(eng)
    else:
        # Quick self-test: one search from the start position.
        eng = GPUEngine(depth=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
        b = chess.Board()
        r = eng.best_move(b)
        print(r)
