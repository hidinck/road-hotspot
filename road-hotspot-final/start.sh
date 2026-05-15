#!/usr/bin/env bash
# start.sh — Start the TrafficSense Dashboard server
set -e

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
fi

echo "============================================"
echo "  TrafficSense — Road Hotspot Dashboard"
echo "  Starting server on http://localhost:8000"
echo "  Press Ctrl+C to stop."
echo "============================================"

python main.py
