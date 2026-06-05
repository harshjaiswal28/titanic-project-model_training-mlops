# ── Base image: official Python 3.11 slim ────────────────────
FROM python:3.11-slim

# ── Set working directory inside container ────────────────────
WORKDIR /app

# ── Copy requirements first (Docker cache optimization) ───────
# If requirements.txt doesn't change, Docker reuses this layer
COPY requirements.txt .

# ── Install dependencies ──────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy app files ────────────────────────────────────────────
COPY app.py .
COPY models/ ./models/

# ── Expose port 8000 ─────────────────────────────────────────
EXPOSE 8000

# ── Start command ─────────────────────────────────────────────
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "300", "app:app"]
