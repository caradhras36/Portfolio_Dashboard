# Install Dependencies Script
# This script installs all required dependencies for the Portfolio Dashboard

Write-Host "📦 Installing Portfolio Dashboard Dependencies" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Gray

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
    Write-Host "💡 Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check if pip is available
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✅ pip found: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ pip not found. Please install pip first." -ForegroundColor Red
    exit 1
}

# Upgrade pip first
Write-Host "`n🔄 Upgrading pip..." -ForegroundColor Cyan
try {
    python -m pip install --upgrade pip
    Write-Host "✅ pip upgraded successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not upgrade pip: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Check if requirements.txt exists
if (-not (Test-Path "requirements.txt")) {
    Write-Host "❌ requirements.txt not found. Please run this script from the project root directory." -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host "`n📥 Installing dependencies from requirements.txt..." -ForegroundColor Cyan
try {
    pip install -r requirements.txt
    Write-Host "✅ All dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Error installing dependencies: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 Try running: pip install --upgrade pip" -ForegroundColor Yellow
    exit 1
}

# Install additional development dependencies
Write-Host "`n🛠️  Installing development dependencies..." -ForegroundColor Cyan
$devDependencies = @(
    "black",           # Code formatter
    "flake8",          # Linter
    "pytest",          # Testing framework
    "pytest-cov",      # Coverage
    "mypy"             # Type checker
)

foreach ($dep in $devDependencies) {
    try {
        Write-Host "Installing $dep..." -NoNewline
        pip install $dep --quiet
        Write-Host " ✅" -ForegroundColor Green
    } catch {
        Write-Host " ❌" -ForegroundColor Red
        Write-Host "⚠️  Could not install $dep" -ForegroundColor Yellow
    }
}

# Verify installation
Write-Host "`n🧪 Verifying installation..." -ForegroundColor Cyan
try {
    python -c "import fastapi, uvicorn, pandas, numpy, requests; print('✅ Core dependencies verified')"
    Write-Host "✅ Core dependencies are working correctly" -ForegroundColor Green
} catch {
    Write-Host "❌ Some core dependencies may not be working correctly" -ForegroundColor Red
    Write-Host "💡 Try reinstalling: pip install -r requirements.txt --force-reinstall" -ForegroundColor Yellow
}

Write-Host "`n🎉 Installation complete!" -ForegroundColor Green
Write-Host "💡 You can now run the dashboard with: .\run.ps1" -ForegroundColor Cyan
Write-Host "🔧 Or test unicode support with: .\scripts\test-unicode.ps1" -ForegroundColor Cyan
