# Portfolio Dashboard PowerShell Launcher
# This script sets the proper encoding and launches the dashboard

Write-Host "Setting up environment..." -ForegroundColor Green
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Write-Host "Starting Portfolio Dashboard..." -ForegroundColor Cyan
python main.py
