# ── Base image ───────────────────────────────────────────────────
# Slim Python image keeps the final image smaller than the full
# python image, while still having pip and standard build tools.
FROM python:3.11-slim

# ── Working directory inside the container ──────────────────────
WORKDIR /app

# ── Install dependencies first (before copying all code) ────────
# This is a Docker best practice: dependencies change far less often
# than source code, so putting this step first lets Docker CACHE it.
# Future builds skip reinstalling packages unless requirements.txt
# actually changes — much faster rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code and trained model artifacts ────────────
# models/ is normally gitignored (generated locally via `python
# main.py`), but Docker builds from your local filesystem, not from
# git — so as long as models/ exists on your machine when you run
# `docker build`, it gets included in the image.
COPY . .

# ── Expose the port FastAPI/uvicorn will listen on ───────────────
EXPOSE 8000

# ── Start the API ─────────────────────────────────────────────────
# No --reload here: --reload is a development convenience and is
# never used in production/container images.
CMD ["uvicorn", "fraud_monitoring_app:app", "--host", "0.0.0.0", "--port", "8000"]