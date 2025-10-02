# Portfolio Dashboard Launcher with Unicode Support
# This script launches the dashboard with proper unicode handling

Write-Host "🚀 Starting Portfolio Dashboard..." -ForegroundColor Green

# Set up unicode environment
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python first." -ForegroundColor Red
    exit 1
}

# Check if main.py exists
if (-not (Test-Path "main.py")) {
    Write-Host "❌ main.py not found. Please run this script from the project root directory." -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
if (Test-Path "venv") {
    Write-Host "🐍 Activating virtual environment..." -ForegroundColor Cyan
    & "venv\Scripts\Activate.ps1"
} elseif (Test-Path ".venv") {
    Write-Host "🐍 Activating virtual environment..." -ForegroundColor Cyan
    & ".venv\Scripts\Activate.ps1"
} else {
    Write-Host "⚠️  No virtual environment found. Using system Python." -ForegroundColor Yellow
}

# Install/update dependencies
Write-Host "📦 Checking dependencies..." -ForegroundColor Cyan
try {
    pip install -r requirements.txt --quiet
    Write-Host "✅ Dependencies up to date" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Some dependencies may need manual installation" -ForegroundColor Yellow
}

# Start the dashboard
Write-Host "`n🎯 Launching Portfolio Dashboard..." -ForegroundColor Green
Write-Host "🌐 Dashboard will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "📊 Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "=" * 50 -ForegroundColor Gray

try {
    python main.py
} catch {
    Write-Host "`n❌ Error starting dashboard: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 Check the console output above for more details" -ForegroundColor Yellow
    exit 1
}
