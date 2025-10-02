@echo off
title Portfolio Dashboard
color 0A

echo.
echo ========================================
echo    🚀 Portfolio Dashboard Launcher
echo ========================================
echo.

cd /d "%~dp0"

echo 🔧 Setting up environment...
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONLEGACYWINDOWSSTDIO=0

echo 🐍 Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    echo 💡 Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found!

echo.
echo 🎯 Starting Portfolio Dashboard...
echo 🌐 Dashboard will be available at: http://localhost:8000
echo 📊 Press Ctrl+C to stop the server
echo ========================================
echo.

python main.py

if errorlevel 1 (
    echo.
    echo ❌ Error starting dashboard!
    echo 💡 Check the console output above for details
    echo 🔧 Try running: scripts\setup-unicode.ps1
    echo.
    pause
)
