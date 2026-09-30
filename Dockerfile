# ── PCOS Care Navigator — HuggingFace Spaces Dockerfile ──────────────────
# Uses CPU-only PyTorch to keep the image under 1.5 GB.
# HF Spaces exposes port 7860 by default.
# ─────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps for sentence-transformers, Pillow, and LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 1. Install CPU-only PyTorch first (separate layer for Docker cache) ───
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── 2. Install the rest of the dependencies ───────────────────────────────
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# ── 3. Copy application code ──────────────────────────────────────────────
COPY agents/     ./agents/
COPY api/        ./api/
COPY graph/      ./graph/
COPY retrieval/  ./retrieval/
COPY models/     ./models/
COPY data/       ./data/

# ── 4. HuggingFace Spaces runs as a non-root user ─────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# ── 5. Expose HF Spaces default port ──────────────────────────────────────
EXPOSE 7860

# ── 6. Start FastAPI ───────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
