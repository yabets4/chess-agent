# Chess RL Server

A FastAPI backend that serves a small (~95k parameter) policy + value neural network
that plays chess. It is the model engine for the React chess UI in the sibling
[`chess-game`](../chess-game) project.

Self-contained, CPU-only, and ready to deploy to **Render** or **Railway** for free.

## What it does

- `/api/move` – picks the model's move for a FEN (temperature + checkpoint selectable)
- `/api/eval` – neural-network evaluation of a position `[-1, 1]` (White's perspective)
- `/api/minimax/move` – depth-limited minimax + alpha-beta search (CPU or batched GPU evals)
- `/api/minimax/eval` – static minimax evaluator
- `/api/train` – online RL update on finished games (outcome-based targets, no Stockfish)
- `/api/checkpoints` – list available trained checkpoints
- `/api/status` – health/status

Interactive API docs at `/docs` once running.

> Stockfish was removed. Evaluation and training targets now use the neural network
> and game outcomes only, so the server has no external engine binary or licensing concerns.

## Requirements

- Python 3.11+ (3.9–3.13 covered by the pinned CPU torch)
- Dependencies are in `requirements.txt`

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
python server.py
```

The server listens on `0.0.0.0:8001` (override with the `PORT` env var). The trained
weights are in [model.pth](model.pth); `checkpoints/` is created and populated on
startup.

## Deploy to Render

1. Push this repo to GitHub.
2. [render.com](https://render.com) → **New** → **Web Service** → connect the repo.
3. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free
4. Deploy. You get a public URL like `https://chess-rl.onrender.com`. Hit `/docs`
   to verify.

## Deploy to Railway

1. Push this repo to GitHub.
2. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Railway auto-detects Python and runs `pip install -r requirements.txt` +
   `uvicorn server:app --host 0.0.0.0 --port $PORT`. Set a `PORT` variable if needed
   (Railway sets it automatically).
4. Public URL looks like `https://chess-rl.up.railway.app`. Add a public
   **Networking** domain to it.

## Serving the chess UI from the same deploy

If you build the frontend and drop the static output into a `dist/` folder next to
`server.py`, the server will serve it too — one single URL for everything:

```bash
# from the chess-game directory
npm run build
# copy the output into this repo
xcopy dist ..\chess-rl\dist\ /E /I     # Windows
```

`server.py` automatically mounts `dist/` when present (SPA-style fallback to
`index.html` included). This is how you get one public URL — no CORS, no second host.

## ⚠️ Frontend API URL

The `chess-game` frontend currently calls `http://localhost:8001/api/...` directly.
That must point at your deployed backend. For a same-origin deploy (above), change
those calls to relative `/api/...`. For a separate frontend host (Vercel/Netlify),
point it at your Render/Railway URL instead.

If the `dist/` folder is present and committed, it will be served by the deploy —
make sure it is *not* gitignored.

## Training scripts (optional, local)

- `train_rl.py` – self-play RL loop. `python train_rl.py 100` trains 100 games.
- `train_worker.py` – distributed self-play worker that POSTs finished games to the
  server's `/api/train` (`TRAIN_SERVER_URL` env var selects the endpoint).

Checkpoints are saved every 10 games into `checkpoints/` and are selectable from the
UI via `/api/move`'s `checkpoint` field.

## Project layout

```
server.py          FastAPI app (model + minimax endpoints)
model.py           Neural network, board encoder, action space
minimax/           CPU and GPU-batched minimax engines
train_rl.py        Self-play RL training
train_worker.py    Distributed training worker
model.pth          The trained weights (~685 KB)
checkpoints/       Saved training checkpoints (created at runtime)
```