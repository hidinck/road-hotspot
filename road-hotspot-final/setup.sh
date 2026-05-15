#!/usr/bin/env bash
# setup.sh — One-command environment setup for TrafficSense Dashboard
set -e

echo "============================================"
echo "  TrafficSense — Road Hotspot Dashboard"
echo "  Environment Setup"
echo "============================================"

# 1. Check Python version
PYTHON=$(command -v python3 || command -v python)
PY_VER=$($PYTHON --version 2>&1)
echo "[1/4] Python found: $PY_VER"

# 2. Create virtual environment
if [ ! -d "venv" ]; then
    echo "[2/4] Creating virtual environment..."
    $PYTHON -m venv venv
else
    echo "[2/4] Virtual environment already exists, skipping."
fi

# 3. Activate and install deps
echo "[3/4] Installing dependencies..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
pip install --upgrade pip -q
pip install -r requirements.txt

# 4. Done
echo "[4/4] Setup complete!"
echo ""
echo "  To start the server, run:"
echo "    ./start.sh"
echo "  Or manually:"
echo "    source venv/bin/activate && python main.py"
echo ""
