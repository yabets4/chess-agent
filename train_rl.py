import os
import sys
import time
import random
import threading
import numpy as np
import torch
import torch.nn.functional as F

import chess
from model import ChessNet, encode_board, legal_action_mask, action_to_move, move_to_action

# ---------------------------------------------------------------------------
# Setup & Device Initialization
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
GAMES_COUNT_FILE = os.path.join(BASE_DIR, "games_count.txt")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("================================--------------------")
print(f"SELF-PLAY TRAINING (REINFORCEMENT LEARNING)")
print(f"Device: {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("================================--------------------")

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

# Save pristine 0-games checkpoint if not present
checkpoint_0_path = os.path.join(CHECKPOINTS_DIR, "model_checkpoint_0.pth")
if not os.path.exists(checkpoint_0_path):
    print("Saving pristine 0-games checkpoint...")
    try:
        fresh_net = ChessNet().to(device)
        torch.save(fresh_net.state_dict(), checkpoint_0_path)
        print(f"Untrained baseline saved to: {checkpoint_0_path}")
    except Exception as e:
        print(f"Failed to save 0-games checkpoint: {e}")

# Load active weights
if os.path.exists(MODEL_PATH):
    print(f"Loading active weights from {MODEL_PATH}...")
    try:
        net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}. Starting fresh.")
else:
    print("No active weights found. Starting fresh.")
    torch.save(net.state_dict(), MODEL_PATH)

# ---------------------------------------------------------------------------
# Batched Training Function
# ---------------------------------------------------------------------------

def train_on_batch(batch_states, batch_actions, batch_raw_targets, batch_shaping):
    """One big forward + backward pass over ALL accumulated game data.

    Value network learns the raw game outcome only.
    Policy advantage adds shaping (draw/length penalties) — can't be learned away.
    """
    net.train()
    optimizer.zero_grad()

    all_states = np.concatenate(batch_states, axis=0)
    all_actions = np.concatenate(batch_actions)
    all_raw = np.concatenate(batch_raw_targets)
    all_shaping = np.concatenate(batch_shaping)
    total_moves = len(all_actions)

    X = torch.from_numpy(all_states).to(device)
    actions = torch.tensor(all_actions, dtype=torch.long).to(device)
    raw_targets = torch.tensor(all_raw, dtype=torch.float32).to(device)

    logits, value = net(X)

    log_probs = F.log_softmax(logits, dim=-1)

    shaping_t = torch.tensor(all_shaping, dtype=torch.float32).to(device)
    advantages = (raw_targets + shaping_t) - value.detach()

    step_losses = -log_probs[torch.arange(total_moves), actions] * advantages
    policy_loss = torch.mean(step_losses)
    value_loss = F.mse_loss(value, raw_targets)

    loss = policy_loss + value_loss
    loss.backward()
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item()

# ---------------------------------------------------------------------------
# Self-Play RL Training Loop
# ---------------------------------------------------------------------------

def run_self_play_training(num_games=100, temperature=1.0):
    global games_count
    print(f"\nStarting training loop for {num_games} games...")
    print("Press Ctrl+C at any time to halt safely and save active weights.\n")

    start_time = time.time()
    BATCH_SIZE = 10

    # Batch accumulation buffers
    batch_states = []
    batch_actions = []
    batch_raw_targets = []
    batch_shaping = []

    for game_idx in range(1, num_games + 1):
        game_start_time = time.time()
        board = chess.Board()

        states_history = []
        turns_history = []
        actions_history = []

        MAX_MOVES = 90
        move_count = 0

        while not board.is_game_over() and move_count < MAX_MOVES:
            turn = board.turn
            planes = encode_board(board)
            states_history.append(planes)
            turns_history.append(turn)

            # Inference
            net.eval()
            with torch.no_grad():
                x = torch.from_numpy(planes).unsqueeze(0).to(device)
                logits, _ = net(x)
                mask = legal_action_mask(board)
                mask_tensor = torch.from_numpy(mask).to(device)
                logits = logits.masked_fill(~mask_tensor.unsqueeze(0), float("-inf"))
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

            # --- Move selection ---
            legal_indices = np.where(probs > 0.0)[0]
            if len(legal_indices) == 0:
                legal_moves = list(board.legal_moves)
                move = random.choice(legal_moves)
                action_idx = move_to_action(board, move)
            else:
                legal_probs = probs[legal_indices]
                legal_probs = legal_probs / np.sum(legal_probs)
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

            if move not in board.legal_moves:
                piece = board.piece_at(move.from_square)
                if piece and piece.piece_type == chess.PAWN:
                    to_rank = chess.square_rank(move.to_square)
                    if to_rank == 7 or to_rank == 0:
                        move.promotion = chess.QUEEN

            actions_history.append(action_idx)
            board.push(move)
            move_count += 1

        # Determine outcome
        timeout = (move_count >= MAX_MOVES and not board.is_game_over())
        if timeout:
            outcome = 0.0
            result_label = "Timeout"
            reason = "Timeout (90 move limit)"
        else:
            result = board.result()
            outcome = 0.0
            if result == "1-0":
                outcome = 1.0
            elif result == "0-1":
                outcome = -1.0
            result_label = "Draw" if outcome == 0.0 else ("White wins" if outcome == 1.0 else "Black wins")
            reason = "Checkmate" if board.is_checkmate() else "Draw (stalemate/insufficient/50-move)"

        # Value targets from the perspective of the side to move (white = outcome)
        G = len(states_history)
        raw_vals = [outcome if turns_history[t] == chess.WHITE else -outcome for t in range(G)]

        # Shaping terms (draw penalty + length penalty + speed bonus)
        shaping_vals = np.zeros(G, dtype=np.float32)

        if outcome == 0.0:
            for t in range(G):
                outcome_for_side = -0.5
                shaping_vals[t] += 0.5 * (outcome_for_side - raw_vals[t])
        else:
            for t in range(G):
                outcome_for_side = outcome if turns_history[t] == chess.WHITE else -outcome
                shaping_vals[t] += 0.5 * (outcome_for_side - raw_vals[t])

        for t in range(G):
            shaping_vals[t] += -0.015 * max(0, t - 40)

        if outcome != 0.0:
            speed_bonus = max(0, 0.4 - 0.002 * move_count)
            shaping_vals += speed_bonus

        # Accumulate into batch buffer
        batch_states.append(np.array(states_history))
        batch_actions.append(np.array(actions_history, dtype=np.int64))
        batch_raw_targets.append(np.array(raw_vals, dtype=np.float32))
        batch_shaping.append(shaping_vals)

        # Train when buffer is full or at the very end
        train_now = (len(batch_states) >= BATCH_SIZE or game_idx == num_games)
        loss_val = policy_val = value_val = 0.0
        games_in_batch = 0
        ckpt_str = ""

        if train_now:
            loss_val, policy_val, value_val = train_on_batch(
                batch_states, batch_actions, batch_raw_targets, batch_shaping
            )
            games_in_batch = len(batch_states)

            # Persistent saves (once per batch)
            for _ in range(games_in_batch):
                games_count += 1
                try:
                    temp_model_path = MODEL_PATH + ".tmp"
                    torch.save(net.state_dict(), temp_model_path)
                    os.replace(temp_model_path, MODEL_PATH)
                    with open(GAMES_COUNT_FILE, "w") as f:
                        f.write(str(games_count))
                except Exception as e:
                    print(f"Error saving training state: {e}")

            if games_count % 10 == 0 or games_count % 100 == 0:
                checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"model_checkpoint_{games_count}.pth")
                try:
                    torch.save(net.state_dict(), checkpoint_path)
                    ckpt_str = f" [CHECKPOINT {games_count}]"
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")

            batch_states = []
            batch_actions = []
            batch_raw_targets = []
            batch_shaping = []

        elapsed = time.time() - game_start_time
        display_count = games_count + (0 if train_now else len(batch_states))
        loss_str = f"Loss: {loss_val:.4f} (Pol: {policy_val:.4f}, Val: {value_val:.4f})" if train_now else "Accumulating..."
        print(f"Game {game_idx}/{num_games} ({display_count} total): {move_count} moves in {elapsed:.1f}s | {result_label} ({reason}) | {loss_str}{ckpt_str}")

    total_elapsed = time.time() - start_time
    games_done = num_games
    print("\n================--------------------")
    print(f"TRAINING COMPLETE!")
    print(f"Trained {games_done} games in {total_elapsed:.1f}s (Avg {total_elapsed/games_done:.1f}s per game).")
    print(f"Current persistent game count: {games_count}")
    print("================--------------------")


if __name__ == "__main__":
    num_games = 100
    if len(sys.argv) > 1:
        try:
            num_games = int(sys.argv[1])
        except ValueError:
            print("Invalid argument. Defaulting to 100 games.")

    try:
        run_self_play_training(num_games=num_games, temperature=1.0)
    except KeyboardInterrupt:
        print("\n\nTraining halted by user. Saving state...")
    finally:
        try:
            torch.save(net.state_dict(), MODEL_PATH)
        except Exception as e:
            print(f"Error saving weights: {e}")
        sys.exit(0)