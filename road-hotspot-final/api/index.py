# api/index.py — Vercel serverless entry point
# Vercel looks for an `app` (ASGI) object in this file.
# We just re-export the FastAPI app from main.py.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app  # noqa: F401  (Vercel detects `app` automatically)
