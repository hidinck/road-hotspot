@echo off
REM start.bat — Start the TrafficSense Dashboard server (Windows)

echo ============================================
echo   TrafficSense - Road Hotspot Dashboard
echo   Starting server on http://localhost:8000
echo   Press Ctrl+C to stop.
echo ============================================

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

python main.py
