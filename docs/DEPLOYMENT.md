# Deployment Guide

The trained model (`models/heart_pipeline.joblib`) is committed, so every option
below works without retraining. Pick based on what you want to show.

| Goal | Best option | Effort |
| --- | --- | --- |
| **Live demo link for recruiters** | Streamlit Community Cloud | ⭐ easiest, free |
| **Production-style REST API** | Render / Railway + Docker | medium |
| **Self-contained / local** | Docker Compose | easy |
| **ML-community visibility** | Hugging Face Spaces | easy |

---

## Recommended: Streamlit Community Cloud (free, ~5 min)

1. Push the repo to GitHub (see `docs/GITHUB_SETUP.md`).
2. Go to <https://share.streamlit.io> → **New app**.
3. Select your repo, branch `main`, main file path:
   `app/streamlit_app.py`.
4. Click **Deploy**. Streamlit installs `requirements.txt` automatically.
5. Share the generated `https://<app>.streamlit.app` URL on your resume/LinkedIn.

> The app inserts `src/` onto `sys.path` itself, so no extra config is needed.

---

## Docker (API + UI together)

```bash
docker compose up --build
#  API  → http://localhost:8000/docs
#  App  → http://localhost:8501
```

Single API container:
```bash
docker build -t heart-api .
docker run -p 8000:8000 heart-api
curl http://localhost:8000/health
```

---

## Render (FastAPI, free tier)

1. New → **Web Service** → connect your GitHub repo.
2. Environment: **Docker** (uses the included `Dockerfile`).
3. Render auto-detects the exposed port `8000`. Deploy.
4. Health check path: `/health`.

(Equivalent steps work on **Railway** and **Fly.io**.)

---

## Hugging Face Spaces (Streamlit)

1. Create a new **Space** → SDK: **Streamlit**.
2. Push this repo to the Space (or connect GitHub).
3. Set the app file to `app/streamlit_app.py` in the Space settings, or rename a
   copy to `app.py` at the root if the Space expects that.

---

## Production checklist (talking points)
- [ ] `GET /health` returns `200` and `status: ok` (used by load balancers).
- [ ] Model artifact present or trained at build (`Dockerfile` handles both).
- [ ] Input validation rejects out-of-range values (HTTP 422).
- [ ] Logs are structured and timestamped (`heart.logging_conf`).
- [ ] CI (lint + train + test) green before deploy (`.github/workflows/ci.yml`).
- [ ] (Future) request logging + drift monitoring + rate limiting.
