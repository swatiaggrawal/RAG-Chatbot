# Backend API image. The frontend is static files and is hosted separately.
#
#   docker build -t rag-chatbot .
#   docker run --env-file .env -p 8000:8000 rag-chatbot
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes don't invalidate this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY data/ data/

# Don't run as root.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", 8000)}/health')"

# Hosting platforms (Render, Fly, Cloud Run) inject $PORT.
CMD ["sh", "-c", "uvicorn api:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}"]
