@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" sem_workbench.py
) else (
  python sem_workbench.py
)
if errorlevel 1 pause
