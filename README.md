# Chess RL Server

A chess **AI backend**: a small neural network trained from scratch with PyTorch,
plus a fast minimax engine, both exposed as a simple REST API.

Point any client at the API and it will suggest chess moves, evaluate positions,
run deep searches, and even learn from games you feed it. It is CPU-only, ships
as a single Python service, and deploys to **Render** or **Railway** out of the box.

## What is this?

This is not the board UI — it's the brain. The project contains:

- **A self-trained chess model.** An AlphaZero-style **policy + value network**
  (~95k parameters) that predicts good moves and position strength. It was trained
  by playing chess against itself thousands of times (self-play reinforcement
  learning), no grandmaster data required. Weights are shipped in `model.pth`.
- **A minimax engine.** A classic depth-limited alpha-beta search with move ordering
  and quiescence, as a configurable alternative to the neural net.
- **A training loop.** The model keeps improving: give the API finished games and
  it updates the weights on the spot.

Current playing strength is casual/amateuristic — good for a fun opponent, not a
Stockfish killer. Everything runs on a single CPU.

## What it's built on

- **PyTorch** — the neural network (CPU or GPU)
- **python-chess** — only for chess move generation and rules (no engine binary)
- **FastAPI** — the REST layer, with auto-generated docs at `/docs`

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/move` | POST | Pick a move for a position (`fen`, `temperature`, `checkpoint`) |
| `/api/eval` | POST | Evaluate a position `[-1, 1]` (1 = winning for White) |
| `/api/minimax/move` | POST | Depth-limited minimax + alpha-beta move (`depth`, `backend`) |
| `/api/minimax/eval` | POST | Static minimax evaluation in centipawns and `[-1, 1]` |
| `/api/train` | POST | Train the model on finished games (`fens`, `moves`, `outcome`) |
| `/api/checkpoints` | GET | List available trained checkpoints |
| `/api/status` | GET | Server health and device info |

### Quick example

```bash
curl -X POST https://your-server.example.com/api/move \
  -H "Content-Type: application/json" \
  -d '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "temperature": 0.0}'
```

```json
{"move": "e2e4", "value": 0.16, "fallback": false}
```

`fallback: true` means the model had no confident move and picked a random legal one.

## Running locally

Requires Python 3.9+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
python server.py
```

The server starts on `http://localhost:8001` (override with the `PORT` env var).
Open `/docs` for interactive API testing.

## Deployment

The service is stateless over HTTP and runs anywhere Python runs.

<details>
<summary><b>Deploy to Render</b></summary>

1. Push this repo to GitHub.
2. On [render.com](https://render.com) → **New** → **Web Service** → connect the repo.
3. Use these settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free
4. Deploy and visit `https://<your-app>.onrender.com/docs`.
</details>

<details>
<summary><b>Deploy to Railway</b></summary>

1. Push this repo to GitHub.
2. On [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Railway auto-detects Python and runs `pip install -r requirements.txt` and
   `uvicorn server:app --host 0.0.0.0 --port $PORT` for you.
4. Set up a public Networking domain to get a public URL.
</details>

<details>
<summary><b>Serve a frontend from the same service</b></summary>

If you build a chess UI and place the static files in a `dist/` folder next to
`server.py`, the server automatically serves the UI too (with SPA fallback). One
single URL for everything, no CORS needed:

```bash
# from your frontend project
npm run build
# copy the build output into this repo as /dist
```

The deployment then serves both the board UI (`/`) and the model API (`/api/*`).
</details>

## Training

The included model is already trained; these scripts are for when you want to
continue or re-train it yourself.

```bash
# Self-play reinforcement learning (plays <N> games against itself every run)
python train_rl.py 100

# Distributed worker: plays games locally, posts them to a central server
python train_worker.py
```

Checkpoints are saved every 10 games into `checkpoints/`. You can select which
checkpoint the API plays with via the `checkpoint` field of `/api/move`.

## Project layout

```
server.py          FastAPI app (model + minimax endpoints)
model.py           Neural network, board encoder, action space (4672 moves)
minimax/           CPU and GPU-batched minimax engines
train_rl.py        Self-play reinforcement learning trainer
train_worker.py    Distributed training worker (posts games to the server)
model.pth          The pre-trained weights (~685 KB)
checkpoints/       Saved training checkpoints (created at runtime)
```