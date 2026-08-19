"""
Chess policy+value network, board encoder, and action space.

Action space: AlphaZero canonical 4672 = 64 from_squares * 73 move planes.
Move planes: 8 queen directions * 7 distances (56), 8 knight offsets (8),
             and 9 underpromotion variants (3 directions * 3 piece types = 9).
Total: 56 + 8 + 9 = 73.

Board encoding: 17 planes (8x8), float32, on CUDA.
  0-5:  white pieces P,N,B,R,Q,K
  6-11: black pieces P,N,B,R,Q,K
  12:   side to move (1.0 = white, 0.0 = black)  [constant plane]
  13-16: castling rights K, Q, k, q

Network: small residual conv net, ~95K parameters.
"""
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_PLANES = 17
BOARD_H = 8
BOARD_W = 8
NUM_MOVE_PLANES = 73
NUM_ACTIONS = 64 * NUM_MOVE_PLANES  # 4672

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

# Move plane definitions for action encoding.
# 8 queen directions, 7 distances, total 56 planes.
# (dr, dc) direction in (row, col) space; row 0 = rank 8, white moves up (row decreases).
QUEEN_DIRS = [
    (-1,  0),  # N  (white up)
    (-1,  1),  # NE
    ( 0,  1),  # E
    ( 1,  1),  # SE
    ( 1,  0),  # S
    ( 1, -1),  # SW
    ( 0, -1),  # W
    (-1, -1),  # NW
]
# Knight offsets: 8.
KNIGHT_OFFSETS = [
    (-2, -1), (-2,  1), (-1, -2), (-1,  2),
    ( 1, -2), ( 1,  2), ( 2, -1), ( 2,  1),
]
# Underpromotion: 3 directions (N, W, E) * 3 piece types (N, B, R) = 9.
# 0..2: knight-promotions N, W, E
# 3..5: bishop-promotions
# 6..8: rook-promotions
PROMO_OFFSETS = [(-1, -1), (-1, 0), (-1, 1)]
PROMO_PIECE_TYPES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


# ---------------------------------------------------------------------------
# Move <-> action index
# ---------------------------------------------------------------------------

def get_plane_idx(from_sq, to_sq, promo=None):
    """Compute the move plane index (0..72) for a move from from_sq to to_sq."""
    fr, fc = from_sq // 8, from_sq % 8
    tr, tc = to_sq // 8, to_sq % 8
    dr = tr - fr
    dc = tc - fc

    # Underpromotions (planes 64..72)
    if promo and promo != chess.QUEEN:
        norm_dr = -dr if dr > 0 else dr
        for p_idx, (p_dr, p_dc) in enumerate(PROMO_OFFSETS):
            for pt_idx, pt in enumerate(PROMO_PIECE_TYPES):
                if pt == promo and norm_dr == p_dr and dc == p_dc:
                    return 64 + p_idx * 3 + pt_idx
        return -1

    # Knight moves (planes 56..63)
    if (dr, dc) in KNIGHT_OFFSETS:
        return 56 + KNIGHT_OFFSETS.index((dr, dc))

    # Queen moves (planes 0..55)
    for d_idx, (gdr, gdc) in enumerate(QUEEN_DIRS):
        for dist in range(1, 8):
            if dr == gdr * dist and dc == gdc * dist:
                return d_idx * 7 + (dist - 1)

    return -1


def move_to_action(board, move):
    """Given a chess.Move, return its action index (0..4671)."""
    f = move.from_square
    t = move.to_square
    plane_idx = get_plane_idx(f, t, move.promotion)
    if plane_idx == -1:
        return -1
    return f * 73 + plane_idx


def action_to_move(action_idx):
    """Given an action index (0..4671), return a chess.Move."""
    from_sq = action_idx // 73
    plane_idx = action_idx % 73
    fr, fc = from_sq // 8, from_sq % 8

    # Underpromotions
    if plane_idx >= 64:
        promo_idx = plane_idx - 64
        p_idx = promo_idx // 3
        pt_idx = promo_idx % 3
        p_dr, p_dc = PROMO_OFFSETS[p_idx]
        promo = PROMO_PIECE_TYPES[pt_idx]
        dr = p_dr if fr == 1 else -p_dr
        tr, tc = fr + dr, fc + p_dc
        to_sq = tr * 8 + tc
        return chess.Move(from_sq, to_sq, promotion=promo)

    # Knight moves
    if plane_idx >= 56:
        k_idx = plane_idx - 56
        dr, dc = KNIGHT_OFFSETS[k_idx]
        tr, tc = fr + dr, fc + dc
        to_sq = tr * 8 + tc
        return chess.Move(from_sq, to_sq)

    # Queen moves
    d_idx = plane_idx // 7
    dist = (plane_idx % 7) + 1
    gdr, gdc = QUEEN_DIRS[d_idx]
    tr, tc = fr + gdr * dist, fc + gdc * dist
    to_sq = tr * 8 + tc
    return chess.Move(from_sq, to_sq)


def legal_action_mask(board):
    """Returns a (NUM_ACTIONS,) numpy bool array: True where action is legal in the current position."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for move in board.legal_moves:
        a = move_to_action(board, move)
        if a >= 0:
            mask[a] = True
    return mask


# ---------------------------------------------------------------------------
# Board encoder
# ---------------------------------------------------------------------------

def encode_board(board):
    """Encode a chess.Board as a (NUM_PLANES, 8, 8) numpy float32 array on CPU.
    Channel layout: 12 piece planes (P,N,B,R,Q,K for W then B), side-to-move,
    4 castling rights.
    """
    planes = np.zeros((NUM_PLANES, BOARD_H, BOARD_W), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        r = 7 - (sq // 8)  # row 0 in array = rank 8 (top of screen)
        c = sq % 8
        planes[PIECE_PLANE[(piece.piece_type, piece.color)], r, c] = 1.0
    # Side to move: 1.0 if white to move, 0.0 if black.
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    # Castling rights.
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    return planes


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class ChessNet(nn.Module):
    """Small policy+value net: stem conv -> N residual blocks -> policy head + value head."""

    def __init__(self, num_blocks=2, channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        # Policy head: 1x1 conv to NUM_MOVE_PLANES channels, output is (B, 73, 8, 8) = (B, 4672)
        self.policy_conv = nn.Conv2d(channels, NUM_MOVE_PLANES, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(NUM_MOVE_PLANES)
        # Value head: 1x1 conv to 1 channel, then FC -> tanh
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """Input: (B, 17, 8, 8) float. Output: logits (B, 4672), value (B,)."""
        h = self.stem(x)
        h = self.blocks(h)
        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(h)))  # (B, 73, 8, 8)
        logits = p.reshape(p.size(0), -1)                 # (B, 4672)
        # Value
        v = F.relu(self.value_bn(self.value_conv(h)))    # (B, 1, 8, 8)
        v = v.reshape(v.size(0), -1)                      # (B, 64)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)    # (B,)
        return logits, v


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Smoke test: build net, encode a board, run a forward pass, pick a move.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    net = ChessNet().to(device)
    print(f"Params: {count_parameters(net):,}")

    board = chess.Board()
    x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)  # (1, 17, 8, 8)
    logits, value = net(x)
    print(f"Logits shape: {logits.shape}, value: {value.item():.4f}")

    mask = torch.from_numpy(legal_action_mask(board)).to(device)
    logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = F.softmax(logits, dim=-1)
    a = torch.multinomial(probs[0], 1).item()
    move = action_to_move(a)
    print(f"Sampled move: {move.uci()} (action {a}, prob {probs[0, a].item():.4f})")
    print("OK")
