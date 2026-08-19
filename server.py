import os
import random
import math
import threading

import numpy as np
import torch
import torch.nn.functional as F

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

import chess
from model import ChessNet, encode_board, legal_action_mask, action_to_move, move_to_action

# ---------------------------------------------------------------------------
# Setup and Model/Checkpoint Directories
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_lock = threading.Lock()

MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
GAMES_COUNT_FILE = os.path.join(BASE_DIR, "games_count.txt")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

net = ChessNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Persistent game counter
games_count = 0
if os.path.exists(GAMES_COUNT_FILE):
    try:
        with open(GAMES_COUNT_FILE, "r") as f:
            games_count = int(f.read().strip())
        print(f"Loaded persistent game counter: {games_count} games trained.")
    except Exception as e:
        print(f"Error reading games_count: {e}")

# Save pristine 0-games checkpoint (randomly initialized model)
checkpoint_0_path = os.path.join(CHECKPOINTS_DIR, "model_checkpoint_0.pth")
if not os.path.exists(checkpoint_0_path):
    print("Generating pristine 0-games checkpoint...")
    try:
        fresh_net = ChessNet().to(device)
        torch.save(fresh_net.state_dict(), checkpoint_0_path)
        print(f"Pristine untrained model successfully saved to: {checkpoint_0_path}")
    except Exception as e:
        print(f"Failed to save 0-games checkpoint: {e}")

if os.path.exists(MODEL_PATH):
    print(f"Loading active model weights from {MODEL_PATH}...")
    try:
        net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}. Starting with fresh weights.")
else:
    print("No saved active weights found. Starting with fresh weights.")
    torch.save(net.state_dict(), MODEL_PATH)

app = FastAPI(title="Chess RL Model Server")

# Allow CORS from any origin so the frontend can be hosted anywhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class MoveRequest(BaseModel):
    fen: str
    temperature: Optional[float] = 1.0
    checkpoint: Optional[str] = "latest"

class MinimaxMoveRequest(BaseModel):
    fen: str
    depth: int = 3
    backend: str = "gpu"  # 'gpu' (torch-batched leaf eval, default) or 'cpu'

class EvalRequest(BaseModel):
    fen: str
    checkpoint: Optional[str] = "latest"

class TrainRequest(BaseModel):
    fens: List[str]
    moves: List[str]  # UCI format (e.g., e2e4, g1f3)
    outcome: float    # 1.0 for White win, -1.0 for Black win, 0.0 for Draw

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "device": str(device),
        "model_path": MODEL_PATH,
        "games_trained": games_count,
        "has_saved_weights": os.path.exists(MODEL_PATH),
    }

@app.get("/api/checkpoints")
def list_checkpoints():
    checkpoints = ["latest"]
    if os.path.exists(CHECKPOINTS_DIR):
        try:
            files = [f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith(".pth")]
            # Sort checkpoints numerically: model_checkpoint_100.pth -> 100
            def get_num(name):
                try:
                    return int(name.split("_")[-1].split(".")[0])
                except:
                    return 0
            files.sort(key=get_num)
            checkpoints.extend(files)
        except Exception as e:
            print(f"Error listing checkpoints: {e}")
    return {"checkpoints": checkpoints}

@app.post("/api/move")
def get_move(data: MoveRequest):
    try:
        board = chess.Board(data.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {str(e)}")

    if board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    # Determine which weights to load
    target_weights = MODEL_PATH
    if data.checkpoint and data.checkpoint != "latest":
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, data.checkpoint)
        if os.path.exists(checkpoint_path):
            target_weights = checkpoint_path
        else:
            print(f"Checkpoint {data.checkpoint} not found. Defaulting to latest active model.")

    net.eval()
    with torch.no_grad():
        try:
            # Load specific weights for this move
            net.load_state_dict(torch.load(target_weights, map_location=device))

            # Encode board state
            planes = encode_board(board)
            x = torch.from_numpy(planes).unsqueeze(0).to(device)
            logits, value_tensor = net(x)

            # Mask illegal moves
            mask = legal_action_mask(board)
            mask_tensor = torch.from_numpy(mask).to(device)
            logits = logits.masked_fill(~mask_tensor.unsqueeze(0), float("-inf"))

            # Probabilities via Softmax
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            value = value_tensor.item()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Get legal action indices and their probabilities
    legal_indices = np.where(probs > 0.0)[0]
    if len(legal_indices) == 0:
        # Fallback to random legal move
        legal_moves = list(board.legal_moves)
        chosen_move = random.choice(legal_moves)
        return {
            "move": chosen_move.uci(),
            "value": 0.0,
            "fallback": True
        }

    legal_probs = probs[legal_indices]
    legal_probs = legal_probs / np.sum(legal_probs)  # Normalize

    temp = data.temperature if data.temperature is not None else 1.0

    if temp <= 0.0:
        # Greedy selection
        best_idx = legal_indices[np.argmax(legal_probs)]
    else:
        # Temperature-scaled selection
        if temp != 1.0:
            scaled_probs = np.power(legal_probs, 1.0 / temp)
            scaled_probs = scaled_probs / np.sum(scaled_probs)
        else:
            scaled_probs = legal_probs

        best_idx = np.random.choice(legal_indices, p=scaled_probs)

    chosen_move = action_to_move(best_idx)

    # Check and handle queen promotions if needed in python-chess
    if chosen_move not in board.legal_moves:
        piece = board.piece_at(chosen_move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(chosen_move.to_square)
            if to_rank == 7 or to_rank == 0:
                chosen_move.promotion = chess.QUEEN

    # Double check legality
    if chosen_move not in board.legal_moves:
        # Fallback to random legal move
        legal_moves = list(board.legal_moves)
        chosen_move = random.choice(legal_moves)
        return {
            "move": chosen_move.uci(),
            "value": 0.0,
            "fallback": True
        }

    return {
        "move": chosen_move.uci(),
        "value": value,
        "fallback": False
    }

# ---------------------------------------------------------------------------
# Minimax engine (alpha-beta) endpoints
# ---------------------------------------------------------------------------

_cpu_minimax_engine = None
_gpu_minimax_engine = None
_minimax_lock = threading.Lock()


def _get_minimax_engine(backend):
    """Lazily build a shared minimax engine. CPU backend is fast; GPU backend
    evaluates leaves in torch batches on the selected device."""
    global _cpu_minimax_engine, _gpu_minimax_engine
    if backend == "gpu":
        if _gpu_minimax_engine is None:
            from minimax.gpu_engine import GPUEngine
            _gpu_minimax_engine = GPUEngine()
        return _gpu_minimax_engine
    if _cpu_minimax_engine is None:
        from minimax.engine import Engine
        _cpu_minimax_engine = Engine()
    return _cpu_minimax_engine


@app.post("/api/minimax/move")
def minimax_move(data: MinimaxMoveRequest):
    try:
        board = chess.Board(data.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {str(e)}")

    if board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    depth = max(1, min(6, data.depth))
    backend = "gpu" if data.backend == "gpu" else "cpu"

    with _minimax_lock:
        try:
            engine = _get_minimax_engine(backend)
            result = engine.best_move(board, depth=depth)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Minimax search error: {str(e)}")

    if result.move is None:
        raise HTTPException(status_code=400, detail="No legal moves")

    return {
        "move": result.move.uci(),
        "score": result.score,
        "nodes": result.nodes,
        "depth": result.depth,
        "time": round(result.time or 0.0, 3),
        "pv": [m.uci() for m in result.pv],
        "backend": backend,
    }


@app.post("/api/minimax/eval")
def minimax_eval(data: EvalRequest):
    """Static evaluation of a position (white's perspective, in [-1, 1]) using
    the same material + piece-square tables the minimax engine searches with."""
    try:
        board = chess.Board(data.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {str(e)}")

    from minimax.engine import evaluate as _mm_eval
    cp = _mm_eval(board)
    value = math.tanh(cp / 200.0)
    return {"value": value, "centipawns": int(cp)}

@app.post("/api/eval")
def get_eval(data: EvalRequest):
    """Evaluate a position with the neural network (white's perspective, [-1, 1])."""
    try:
        board = chess.Board(data.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {str(e)}")

    target_weights = MODEL_PATH
    if data.checkpoint and data.checkpoint != "latest":
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, data.checkpoint)
        if os.path.exists(checkpoint_path):
            target_weights = checkpoint_path

    net.eval()
    with torch.no_grad():
        try:
            net.load_state_dict(torch.load(target_weights, map_location=device))
            planes = encode_board(board)
            x = torch.from_numpy(planes).unsqueeze(0).to(device)
            _, value_tensor = net(x)
            value = value_tensor.item()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return {"value": value}

@app.post("/api/train")
def train_on_game(data: TrainRequest):
    """Reinforcement-learning update on a finished game. Without Stockfish,
    the value target is the game outcome (1.0 / -1.0 / 0.0 from White's
    perspective), with reward shaping toward faster decisive games."""
    global games_count
    if len(data.fens) != len(data.moves):
        raise HTTPException(status_code=400, detail="Fens and moves list must have the same length")

    with train_lock:
        # Reload latest active weights first to ensure gradients are applied to current model state
        if os.path.exists(MODEL_PATH):
            net.load_state_dict(torch.load(MODEL_PATH, map_location=device))

        net.train()
        optimizer.zero_grad()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        count = 0

        for idx, (fen, uci) in enumerate(zip(data.fens, data.moves)):
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(uci)

                # Double check promotion flag if pawn reaches backrank
                if move not in board.legal_moves:
                    piece = board.piece_at(move.from_square)
                    if piece and piece.piece_type == chess.PAWN:
                        to_rank = chess.square_rank(move.to_square)
                        if to_rank == 7 or to_rank == 0:
                            move.promotion = chess.QUEEN

                a_t = move_to_action(board, move)
                if a_t == -1:
                    continue

                # Target evaluation from the perspective of the player to move
                raw_val = data.outcome if board.turn == chess.WHITE else -data.outcome

                # Shaping terms (draw penalty + length penalty + speed bonus).
                # These are added ONLY to the policy advantage, NOT to the value target.
                shaping = 0.0

                # Outcome blend: nudge the policy toward decisive, faster results
                if data.outcome == 0.0:
                    outcome_for_side = -0.5
                    shaping += 0.5 * (outcome_for_side - raw_val)
                else:
                    outcome_for_side = data.outcome if board.turn == chess.WHITE else -data.outcome
                    shaping += 0.5 * (outcome_for_side - raw_val)

                # Per-position length penalty: every move past 40 adds -0.015
                shaping += -0.015 * max(0, idx - 40)

                # Speed bonus for quick wins
                if data.outcome != 0.0:
                    speed_bonus = max(0, 0.4 - 0.002 * len(data.moves))
                    shaping += speed_bonus

                # Forward pass
                planes = encode_board(board)
                x = torch.from_numpy(planes).unsqueeze(0).to(device)
                logits, value_tensor = net(x)

                # Policy Loss: reinforcement learning (policy-gradient advantage)
                log_probs = F.log_softmax(logits, dim=-1)
                log_prob_action = log_probs[0, a_t]
                advantage = (raw_val + shaping) - value_tensor.detach().item()
                policy_loss = -log_prob_action * advantage

                # Value Loss (MSE against raw outcome — shaping excluded)
                raw_val_tensor = torch.tensor([raw_val], dtype=torch.float32).to(device)
                value_loss = F.mse_loss(value_tensor, raw_val_tensor)

                loss = policy_loss + value_loss
                loss.backward()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                count += 1
            except Exception as e:
                print(f"Skipping error during transition training: {e}")
                continue

        if count > 0:
            optimizer.step()
            try:
                # Save the updated active model weights atomically to avoid races with workers reading them
                temp_model_path = MODEL_PATH + ".tmp"
                torch.save(net.state_dict(), temp_model_path)
                os.replace(temp_model_path, MODEL_PATH)

                # Increment games count and save persistently
                games_count += 1
                with open(GAMES_COUNT_FILE, "w") as f:
                    f.write(str(games_count))

                # Save checkpoints exactly at multiples of 10 for quick user testing
                if games_count % 10 == 0 or games_count % 100 == 0:
                    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"model_checkpoint_{games_count}.pth")
                    torch.save(net.state_dict(), checkpoint_path)
                    print(f"Saved persistent checkpoint: {checkpoint_path}")

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to save trained weights/checkpoints: {str(e)}")

            print(f"Successfully trained on {count} moves. Avg policy loss: {total_policy_loss/count:.4f}, Avg value loss: {total_value_loss/count:.4f}. Total games: {games_count}")
            return {
                "status": "success",
                "moves_trained": count,
                "avg_policy_loss": total_policy_loss / count,
                "avg_value_loss": total_value_loss / count,
                "total_games_trained": games_count,
                "checkpoint_saved": (games_count % 10 == 0 or games_count % 100 == 0),
                "checkpoint_name": f"model_checkpoint_{games_count}.pth" if (games_count % 10 == 0 or games_count % 100 == 0) else None
            }
        else:
            return {
                "status": "ignored",
                "message": "No valid transitions were trained."
            }

# ---------------------------------------------------------------------------
# Optional static frontend: serve a built UI from ./dist (same-origin deploy)
# ---------------------------------------------------------------------------

DIST_DIR = os.path.join(BASE_DIR, "dist")

if os.path.isdir(DIST_DIR):
    assets_dir = os.path.join(DIST_DIR, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str):
        candidate = os.path.join(DIST_DIR, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(DIST_DIR, "index.html"))
else:
    @app.get("/", include_in_schema=False)
    def read_root():
        return {"status": "ok", "message": "Chess RL model server is running. API docs at /docs"}

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)