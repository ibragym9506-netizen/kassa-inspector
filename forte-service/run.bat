@echo off
cd /d "%~dp0"
if "%API_KEY%"=="" set API_KEY=change-me
set ML_ENGINE=ollama
if not exist venv (
  python -m venv venv
  venv\Scripts\pip install -r requirements.txt
)
venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
