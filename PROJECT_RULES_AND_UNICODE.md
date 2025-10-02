# Project Rules and Unicode Support Guide

## 🎯 Project Rules

This project follows specific rules and standards to ensure consistency and proper functionality:

### 1. PowerShell Preference
- **Always use PowerShell commands** when possible
- All scripts are written in PowerShell for Windows compatibility
- Batch files are provided as fallbacks
- PowerShell scripts include proper unicode handling

### 2. Unicode and Emoji Support
- **Full unicode support** for all text and emojis
- **Emoji usage** in UI elements and console output
- **Greek letters** for financial calculations (α, β, γ, δ, θ, etc.)
- **Mathematical symbols** (∑, ∏, ∫, ∂, ∇, etc.)
- **Currency symbols** ($, €, £, ¥, ₹, etc.)

### 3. Code Standards
- **Black formatting** with 88-character line length
- **Type hints** for all function parameters and returns
- **Google-style docstrings** for documentation
- **Comprehensive testing** with pytest
- **Pre-commit hooks** for code quality

## 🛠️ Available Scripts

### Setup Scripts
```powershell
# Complete development environment setup
.\scripts\setup-development.ps1

# Install dependencies only
.\scripts\install-dependencies.ps1

# Setup unicode support
.\scripts\setup-unicode.ps1
```

### Development Scripts
```powershell
# Format code with Black
.\scripts\format-code.ps1

# Lint code with Flake8
.\scripts\lint-code.ps1

# Run tests with coverage
.\scripts\run-tests.ps1

# Test unicode/emoji support
.\scripts\test-unicode.ps1
```

### Launcher Scripts
```powershell
# Create desktop shortcut
.\create-shortcut.ps1

# Run dashboard (enhanced)
.\run.ps1

# Run dashboard (basic)
.\PortfolioDashboard.bat
```

## 🔧 Unicode Configuration

### Environment Variables
The project automatically sets these environment variables for unicode support:
```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTHONLEGACYWINDOWSSTDIO = "0"
```

### Console Encoding
PowerShell scripts set proper console encoding:
```powershell
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### Python Configuration
Python scripts use the `project_rules.py` module for unicode handling:
```python
from project_rules import setup_unicode, safe_unicode, validate_text

# Setup unicode environment
setup_unicode()

# Get unicode-safe text
safe_text = safe_unicode("📊 Portfolio Data 💰")

# Validate unicode text
validation = validate_text("🚀 Dashboard with emojis")
```

## 🎨 Emoji Usage Guidelines

### UI Elements
- **Status indicators**: ✅ ❌ ⚠️ 🔄
- **Navigation**: 🏠 📊 📈 💰 🎯
- **Actions**: 🚀 ⚡ 🔧 📝 🗑️
- **Financial data**: 💵 💎 📊 📈 📉

### Console Output
- **Success**: ✅ 🎉 🚀
- **Errors**: ❌ ⚠️ 🔥
- **Info**: 💡 🔍 📋
- **Progress**: 🔄 ⏳ 📊

### File Types
- **Scripts**: 🔧 🛠️ ⚙️
- **Data**: 📊 📈 💰
- **Documentation**: 📝 📚 📖
- **Tests**: 🧪 ✅ ❌

## 🚀 Quick Start with Rules

1. **Setup Environment**:
   ```powershell
   .\scripts\setup-development.ps1
   ```

2. **Create Desktop Shortcut**:
   ```powershell
   .\create-shortcut.ps1
   ```

3. **Test Unicode Support**:
   ```powershell
   .\scripts\test-unicode.ps1
   ```

4. **Start Dashboard**:
   - Double-click desktop shortcut, or
   - Run `.\run.ps1`

## 🔍 Troubleshooting

### Unicode Issues
If you see garbled characters or missing emojis:
```powershell
# Run unicode setup
.\scripts\setup-unicode.ps1

# Test unicode support
.\scripts\test-unicode.ps1
```

### PowerShell Execution Policy
If scripts won't run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Python Environment
If Python can't find modules:
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Or install dependencies
.\scripts\install-dependencies.ps1
```

## 📋 Project Structure

```
Portfolio_Dashboard/
├── scripts/                    # PowerShell scripts
│   ├── setup-development.ps1   # Complete dev setup
│   ├── install-dependencies.ps1 # Install deps
│   ├── setup-unicode.ps1       # Unicode setup
│   ├── test-unicode.ps1        # Unicode testing
│   ├── format-code.ps1         # Code formatting
│   ├── lint-code.ps1           # Code linting
│   └── run-tests.ps1           # Test runner
├── project_rules.py            # Rules and unicode handling
├── create-shortcut.ps1         # Desktop shortcut creator
├── PortfolioDashboard.bat      # Batch launcher
├── PortfolioDashboard.vbs      # VBS launcher
└── run.ps1                     # Enhanced PowerShell launcher
```

## 🎉 Benefits

### For Users
- **Easy setup** with one-click scripts
- **Desktop shortcut** for instant access
- **Beautiful UI** with emojis and unicode
- **No Python knowledge** required

### For Developers
- **Consistent standards** across the project
- **PowerShell-first** approach
- **Full unicode support** for international users
- **Comprehensive tooling** for development

### For the Project
- **Professional appearance** with proper unicode
- **Cross-platform compatibility** (Windows focus)
- **Maintainable code** with clear standards
- **User-friendly** setup and usage
