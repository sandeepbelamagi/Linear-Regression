# ── Base stage: shared dependencies ──────────────────────────────────────────
FROM python:3.11-slim AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Training stage ───────────────────────────────────────────────────────────
FROM base AS training

COPY src/ src/
COPY data/ data/
COPY run_pipeline.py .

RUN mkdir -p outputs/models && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "run_pipeline.py"]
CMD []

# ── Serving stage ────────────────────────────────────────────────────────────
FROM base AS serving

COPY src/ src/

RUN mkdir -p outputs/models && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["gunicorn", "src.api.serve:app", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8000", \
    "--workers", "4", \
    "--timeout", "120", \
    "--access-logfile", "-", \
    "--error-logfile", "-"]
