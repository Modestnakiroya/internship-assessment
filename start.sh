#!/usr/bin/env bash
# Run FastAPI + Streamlit (Docker / Hugging Face Spaces).

set -euo pipefail

export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8000}"

uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
UVICORN_PID=$!

cleanup() {
  if kill -0 "${UVICORN_PID}" 2>/dev/null; then
    kill "${UVICORN_PID}" 2>/dev/null || true
    wait "${UVICORN_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

python - <<'PY'
import os
import time
import urllib.error
import urllib.request

base = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
url = base + "/health"
for _ in range(60):
    try:
        urllib.request.urlopen(url, timeout=2)
        break
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        time.sleep(1)
else:
    raise SystemExit("FastAPI backend did not become ready at " + url)
PY

# Streamlit must call FastAPI inside this container. If BACKEND_URL was set to the
# public Hugging Face Space URL (https://*.hf.space), POST /pipeline hits Streamlit
# and returns HTTP 405. Force the internal API base for this process.
export BACKEND_URL="http://127.0.0.1:8000"

streamlit run app.py \
  --server.port "${STREAMLIT_SERVER_PORT:-7860}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableXsrfProtection false \
  --server.maxUploadSize 500
