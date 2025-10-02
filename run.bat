@echo off
REM Portfolio Dashboard Launcher
REM This script sets the proper encoding and launches the dashboard

echo Setting up environment...
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo Starting Portfolio Dashboard...
python main.py

pause
