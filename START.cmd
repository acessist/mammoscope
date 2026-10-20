@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Python environment missing. Run SETUP.ps1 first.
  pause
  exit /b 1
)
start "MammoScope" ".venv\Scripts\pythonw.exe" app.py
