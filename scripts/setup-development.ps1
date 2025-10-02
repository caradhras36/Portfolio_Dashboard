# Development Environment Setup Script
# This script sets up a complete development environment for the Portfolio Dashboard

Write-Host "🛠️  Setting up Portfolio Dashboard Development Environment" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Gray

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
    Write-Host "❌ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    Write-Host "💡 Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Create virtual environment
Write-Host "`n🐍 Creating virtual environment..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "⚠️  Virtual environment already exists. Removing old one..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

try {
    python -m venv venv
    Write-Host "✅ Virtual environment created successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Error creating virtual environment: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "`n🔌 Activating virtual environment..." -ForegroundColor Cyan
try {
    & "venv\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "❌ Error activating virtual environment: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "`n🔄 Upgrading pip..." -ForegroundColor Cyan
try {
    python -m pip install --upgrade pip
    Write-Host "✅ pip upgraded successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not upgrade pip: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Install dependencies
Write-Host "`n📦 Installing project dependencies..." -ForegroundColor Cyan
if (Test-Path "requirements.txt") {
    try {
        pip install -r requirements.txt
        Write-Host "✅ Project dependencies installed" -ForegroundColor Green
    } catch {
        Write-Host "❌ Error installing project dependencies: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "⚠️  requirements.txt not found. Installing basic dependencies..." -ForegroundColor Yellow
    $basicDeps = @("fastapi", "uvicorn", "pandas", "numpy", "requests", "python-dotenv")
    foreach ($dep in $basicDeps) {
        try {
            pip install $dep
            Write-Host "✅ Installed $dep" -ForegroundColor Green
        } catch {
            Write-Host "❌ Failed to install $dep" -ForegroundColor Red
        }
    }
}

# Install development tools
Write-Host "`n🔧 Installing development tools..." -ForegroundColor Cyan
$devTools = @(
    "black",           # Code formatter
    "flake8",          # Linter
    "pytest",          # Testing framework
    "pytest-cov",      # Coverage
    "mypy",            # Type checker
    "pre-commit",      # Git hooks
    "jupyter",         # Notebooks
    "ipython"          # Enhanced REPL
)

foreach ($tool in $devTools) {
    try {
        Write-Host "Installing $tool..." -NoNewline
        pip install $tool --quiet
        Write-Host " ✅" -ForegroundColor Green
    } catch {
        Write-Host " ❌" -ForegroundColor Red
        Write-Host "⚠️  Could not install $tool" -ForegroundColor Yellow
    }
}

# Create .gitignore if it doesn't exist
Write-Host "`n📝 Setting up .gitignore..." -ForegroundColor Cyan
if (-not (Test-Path ".gitignore")) {
    $gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
.venv/
env/
.env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
logs/
*.log
data/*.csv
data/*.json
results/
charts/
__pycache__/
*.pyc

# Environment variables
.env
.env.local
.env.production

# Database
*.db
*.sqlite3

# Cache
.cache/
*.cache
"@
    $gitignoreContent | Out-File -FilePath ".gitignore" -Encoding UTF8
    Write-Host "✅ .gitignore created" -ForegroundColor Green
} else {
    Write-Host "✅ .gitignore already exists" -ForegroundColor Green
}

# Create pre-commit configuration
Write-Host "`n🔗 Setting up pre-commit hooks..." -ForegroundColor Cyan
if (-not (Test-Path ".pre-commit-config.yaml")) {
    $precommitConfig = @"
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.2.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
"@
    $precommitConfig | Out-File -FilePath ".pre-commit-config.yaml" -Encoding UTF8
    Write-Host "✅ Pre-commit configuration created" -ForegroundColor Green
} else {
    Write-Host "✅ Pre-commit configuration already exists" -ForegroundColor Green
}

# Install pre-commit hooks
try {
    pre-commit install
    Write-Host "✅ Pre-commit hooks installed" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not install pre-commit hooks" -ForegroundColor Yellow
}

# Create development scripts
Write-Host "`n📜 Creating development scripts..." -ForegroundColor Cyan

# Format code script
$formatScript = @"
# Format Code Script
Write-Host "🎨 Formatting code with Black..." -ForegroundColor Cyan
black .
Write-Host "✅ Code formatting complete!" -ForegroundColor Green
"@
$formatScript | Out-File -FilePath "scripts\format-code.ps1" -Encoding UTF8

# Lint code script
$lintScript = @"
# Lint Code Script
Write-Host "🔍 Linting code with Flake8..." -ForegroundColor Cyan
flake8 . --max-line-length=88 --extend-ignore=E203
Write-Host "✅ Code linting complete!" -ForegroundColor Green
"@
$lintScript | Out-File -FilePath "scripts\lint-code.ps1" -Encoding UTF8

# Run tests script
$testScript = @"
# Run Tests Script
Write-Host "🧪 Running tests..." -ForegroundColor Cyan
python -m pytest tests/ -v --cov=portfolio_management --cov-report=html
Write-Host "✅ Tests complete!" -ForegroundColor Green
"@
$testScript | Out-File -FilePath "scripts\run-tests.ps1" -Encoding UTF8

Write-Host "✅ Development scripts created" -ForegroundColor Green

# Verify installation
Write-Host "`n🧪 Verifying development environment..." -ForegroundColor Cyan
try {
    python -c "import fastapi, uvicorn, pandas, numpy, requests, black, flake8, pytest; print('✅ All tools verified')"
    Write-Host "✅ Development environment is ready!" -ForegroundColor Green
} catch {
    Write-Host "❌ Some tools may not be working correctly" -ForegroundColor Red
    Write-Host "💡 Try running: pip install -r requirements.txt --force-reinstall" -ForegroundColor Yellow
}

Write-Host "`n🎉 Development environment setup complete!" -ForegroundColor Green
Write-Host "`n📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Activate virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. Run the dashboard: .\run.ps1" -ForegroundColor White
Write-Host "3. Format code: .\scripts\format-code.ps1" -ForegroundColor White
Write-Host "4. Lint code: .\scripts\lint-code.ps1" -ForegroundColor White
Write-Host "5. Run tests: .\scripts\run-tests.ps1" -ForegroundColor White
Write-Host "6. Test unicode: .\scripts\test-unicode.ps1" -ForegroundColor White
