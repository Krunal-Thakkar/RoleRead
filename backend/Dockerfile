# Build context must be the repo root (not backend/), since main.py serves ../frontend
# as static files. Build with:  docker build -f backend/Dockerfile -t roleread .
FROM python:3.12-slim

WORKDIR /app/backend

# Install dependencies first so this layer is cached across code-only changes.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY frontend /app/frontend

# Chroma persists here; mount a volume at this path to survive container restarts.
ENV CHROMA_PERSIST_DIR=/data/chroma_data
RUN mkdir -p /data/chroma_data

EXPOSE 8420

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8420"]
