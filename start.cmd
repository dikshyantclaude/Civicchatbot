@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Install Python and follow the setup steps in README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
pause
