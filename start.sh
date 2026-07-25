#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ ! -d venv ]; then
    echo "[sapphire] creating venv..."
    python3 -m venv venv
fi

source venv/bin/activate

if ! pip show -q fastapi 2>/dev/null; then
    echo "[sapphire] installing dependencies..."
    pip install -q -r requirements.txt
fi

echo "[sapphire] starting on 127.0.0.1:${SAPPHIRE_PORT:-3123}..."
echo "[sapphire] KRYSTAL_GENERIC=${KRYSTAL_GENERIC_URL:-http://127.0.0.1:3124}"
echo "[sapphire] KRYSTAL_SEMANTIC=${KRYSTAL_SEMANTIC_URL:-http://127.0.0.1:3125}"
exec python server.py
