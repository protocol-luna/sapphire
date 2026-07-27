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

echo "[sapphire] using config.yml"
exec python server.py
