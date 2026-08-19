import os
import sys
import time
import random
import requests
import numpy as np
import torch
import torch.nn.functional as F

import chess
from model import ChessNet, encode_board, legal_action_mask, action_to_move, move_to_action

# ---------------------------------------------------------------------------
# Setup & Device Initialization
# ---------------------------------------------------------------------------

MODEL_PATH = "model.pth"
SERVER_URL = os.environ.get("TRAIN_SERVER_URL", "http://localhost:8001/api/train")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get optional worker ID from command line arguments
worker_id = "Worker #1"
if len(sys.argv) > 1:
    worker_id = f"Worker #{sys.argv[1]}"

print("================================--------------------")
print(f"DISTRIBUTED SELF-PLAY TRAINING WORKER: {worker_id}")
print(f"Device: {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"Listening to server at: {SERVER_URL}")
print("================================--------------------")

# Initialize local network to run in-memory inferences once
net = ChessNet().to(device)


# ---------------------------------------------------------------------------
# High-Performance Self-Play Worker Loop
# ---------------------------------------------------------------------------

def run_worker_loop(temperature=1.0):
    print(f"\n[{worker_id}] Starting self-play generation loop...")
    print(f"[{worker_id}] Press Ctrl+C to halt safely.\n")
    
    game_idx = 1
    
    while True:
        game_start_time = time.time()
        board = chess.Board()
        
        # 1. Reload the latest trained model weights once before the game starts
        if os.path.exists(MODEL_PATH):
            try:
                # Load the latest weights from the shared directory
                net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                net.eval()
            except Exception as e:
                print(f"[{worker_id}] Warning: Failed to reload weights ({e}). Playing with existing weights.")
        else:
            print(f"[{worker_id}] Warning: model.pth not found. Using local initialized weights.")

        states_history = []
        turns_history = []
        actions_history = []
        
        # 2. Play the game completely in memory on local CUDA (Zero network overhead!)
        move_count = 0
        while not board.is_game_over():
            planes = encode_board(board)
            states_history.append(planes)
            turns_history.append(board.turn)
            
            # Local CUDA inference
            with torch.no_grad():
                x = torch.from_numpy(planes).unsqueeze(0).to(device)
                logits, _ = net(x)
                
                # Mask illegal moves
                mask = legal_action_mask(board)
                mask_tensor = torch.from_numpy(mask).to(device)
                logits = logits.masked_fill(~mask_tensor.unsqueeze(0), float("-inf"))
                
                # Softmax
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
                
            legal_indices = np.where(probs > 0.0)[0]
            if len(legal_indices) == 0:
                # Fallback to random move
                legal_moves = list(board.legal_moves)
                move = random.choice(legal_moves)
                action_idx = move_to_action(board, move)
            else:
                legal_probs = probs[legal_indices]
                legal_probs = legal_probs / np.sum(legal_probs)  # Normalize
                
                # Sample using temperature-scaled distribution
                if temperature <= 0.0:
                    best_idx = legal_indices[np.argmax(legal_probs)]
                else:
                    if temperature != 1.0:
                        scaled_probs = np.power(legal_probs, 1.0 / temperature)
                        scaled_probs = scaled_probs / np.sum(scaled_probs)
                    else:
                        scaled_probs = legal_probs
                    best_idx = np.random.choice(legal_indices, p=scaled_probs)
                
                action_idx = best_idx
                move = action_to_move(action_idx)
                
            # Handle promotions if needed
            if move not in board.legal_moves:
                piece = board.piece_at(move.from_square)
                if piece and piece.piece_type == chess.PAWN:
                    to_rank = chess.square_rank(move.to_square)
                    if to_rank == 7 or to_rank == 0:
                        move.promotion = chess.QUEEN
                        
            # Record action index
            actions_history.append(action_idx)
            board.push(move)
            move_count += 1

        # Determine outcome
        result = board.result()
        outcome = 0.0 # Draw
        if result == "1-0":
            outcome = 1.0  # White won
        elif result == "0-1":
            outcome = -1.0 # Black won
            
        result_label = "Draw" if outcome == 0.0 else ("White wins" if outcome == 1.0 else "Black wins")
        reason = "Checkmate" if board.is_checkmate() else "Draw (stalemate/insufficient/50-move)"
        elapsed_game = time.time() - game_start_time

        # 3. Send the game data to the central server via a single POST request
        G = len(states_history)
        if G > 0:
            # Construct FENs and moves lists to submit
            # For efficiency, construct matching moves list in string format
            temp_board = chess.Board()
            fens = []
            moves = []
            for action_idx in actions_history:
                fens.append(temp_board.fen())
                move = action_to_move(action_idx)
                
                # Promotion double check
                if move not in temp_board.legal_moves:
                    piece = temp_board.piece_at(move.from_square)
                    if piece and piece.piece_type == chess.PAWN:
                        to_rank = chess.square_rank(move.to_square)
                        if to_rank == 7 or to_rank == 0:
                            move.promotion = chess.QUEEN
                
                moves.append(move.uci())
                temp_board.push(move)

            # Submit to the locked central server
            print(f"[{worker_id}] Game {game_idx} complete ({move_count} moves in {elapsed_game:.1f}s, {result_label}). Submitting to server...")
            try:
                res = requests.post(SERVER_URL, json={
                    "fens": fens,
                    "moves": moves,
                    "outcome": outcome
                })
                if res.status_code == 200:
                    data = res.json()
                    checkpoint_str = f" [💾 Checkpoint: {data['checkpoint_name']}]" if data.get("checkpoint_saved") else ""
                    print(f"[{worker_id}] Submitted successfully! Total games: {data['total_games_trained']} | Losses - Pol: {data['avg_policy_loss']:.4f}, Val: {data['avg_value_loss']:.4f}{checkpoint_str}")
                else:
                    print(f"[{worker_id}] Error submitting game: Server returned status {res.status_code} ({res.text})")
            except Exception as err:
                print(f"[{worker_id}] Failed to contact training server: {err}")
                
        game_idx += 1


if __name__ == "__main__":
    try:
        run_worker_loop(temperature=1.0)
    except KeyboardInterrupt:
        print(f"\n[{worker_id}] Worker halted cleanly by user.")
        sys.exit(0)
