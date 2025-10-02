# Create Desktop Shortcut Script
# This script creates a desktop shortcut to run the Portfolio Dashboard

Write-Host "🖥️  Creating Desktop Shortcut for Portfolio Dashboard" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Gray

# Set up unicode environment
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Get the current script directory and project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$desktopPath = [Environment]::GetFolderPath("Desktop")

Write-Host "📁 Project root: $projectRoot" -ForegroundColor Cyan
Write-Host "🖥️  Desktop path: $desktopPath" -ForegroundColor Cyan

# Create a launcher batch file
$launcherBatch = Join-Path $projectRoot "PortfolioDashboard.bat"
$launcherContent = @'
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
'@

try {
    $launcherContent | Out-File -FilePath $launcherBatch -Encoding UTF8
    Write-Host "✅ Launcher batch file created: PortfolioDashboard.bat" -ForegroundColor Green
} catch {
    Write-Host "❌ Error creating launcher batch file: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Create a PowerShell launcher script
$launcherPs1 = Join-Path $projectRoot "PortfolioDashboard.ps1"
$launcherPs1Content = @'
# Portfolio Dashboard PowerShell Launcher
# Desktop shortcut version

Write-Host "🚀 Portfolio Dashboard" -ForegroundColor Green
Write-Host "=" * 30 -ForegroundColor Gray

# Set up unicode environment
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Change to project directory
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Start dashboard
Write-Host "`n🎯 Starting dashboard..." -ForegroundColor Cyan
Write-Host "🌐 http://localhost:8000" -ForegroundColor Yellow
Write-Host "📊 Press Ctrl+C to stop" -ForegroundColor Yellow

try {
    python main.py
} catch {
    Write-Host "`n❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
'@

try {
    $launcherPs1Content | Out-File -FilePath $launcherPs1 -Encoding UTF8
    Write-Host "✅ PowerShell launcher created: PortfolioDashboard.ps1" -ForegroundColor Green
} catch {
    Write-Host "❌ Error creating PowerShell launcher: $($_.Exception.Message)" -ForegroundColor Red
}

# Create desktop shortcut for batch file
$shortcutPath = Join-Path $desktopPath "Portfolio Dashboard.lnk"
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $launcherBatch
$Shortcut.WorkingDirectory = $projectRoot
$Shortcut.Description = "Portfolio Dashboard - Financial Portfolio Management System"
$Shortcut.IconLocation = "shell32.dll,137"  # Chart icon

try {
    $Shortcut.Save()
    Write-Host "✅ Desktop shortcut created: Portfolio Dashboard.lnk" -ForegroundColor Green
} catch {
    Write-Host "❌ Error creating desktop shortcut: $($_.Exception.Message)" -ForegroundColor Red
}

# Create desktop shortcut for PowerShell version
$shortcutPathPs1 = Join-Path $desktopPath "Portfolio Dashboard (PowerShell).lnk"
$ShortcutPs1 = $WshShell.CreateShortcut($shortcutPathPs1)
$ShortcutPs1.TargetPath = "powershell.exe"
$ShortcutPs1.Arguments = "-ExecutionPolicy Bypass -File `"$launcherPs1`""
$ShortcutPs1.WorkingDirectory = $projectRoot
$ShortcutPs1.Description = "Portfolio Dashboard (PowerShell) - Financial Portfolio Management System"
$ShortcutPs1.IconLocation = "shell32.dll,137"  # Chart icon

try {
    $ShortcutPs1.Save()
    Write-Host "✅ PowerShell desktop shortcut created: Portfolio Dashboard (PowerShell).lnk" -ForegroundColor Green
} catch {
    Write-Host "❌ Error creating PowerShell desktop shortcut: $($_.Exception.Message)" -ForegroundColor Red
}

# Create a VBS launcher for better icon and no console window
$vbsLauncher = Join-Path $projectRoot "PortfolioDashboard.vbs"
$vbsContent = @'
Set WshShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' Get the directory where this VBS file is located
strScriptPath = objFSO.GetParentFolderName(WScript.ScriptFullName)

' Change to the project directory
WshShell.CurrentDirectory = strScriptPath

' Set environment variables
WshShell.Environment("Process")("PYTHONIOENCODING") = "utf-8"
WshShell.Environment("Process")("PYTHONUTF8") = "1"

' Run the Python script
WshShell.Run "python main.py", 1, False
'@

try {
    $vbsContent | Out-File -FilePath $vbsLauncher -Encoding ASCII
    Write-Host "✅ VBS launcher created: PortfolioDashboard.vbs" -ForegroundColor Green
} catch {
    Write-Host "❌ Error creating VBS launcher: $($_.Exception.Message)" -ForegroundColor Red
}

# Create desktop shortcut for VBS version (no console window)
$shortcutPathVbs = Join-Path $desktopPath "Portfolio Dashboard (Silent).lnk"
$ShortcutVbs = $WshShell.CreateShortcut($shortcutPathVbs)
$ShortcutVbs.TargetPath = $vbsLauncher
$ShortcutVbs.WorkingDirectory = $projectRoot
$ShortcutVbs.Description = "Portfolio Dashboard (Silent) - Runs without console window"
$ShortcutVbs.IconLocation = "shell32.dll,137"  # Chart icon

try {
    $ShortcutVbs.Save()
    Write-Host "✅ Silent desktop shortcut created: Portfolio Dashboard (Silent).lnk" -ForegroundColor Green
} catch {
    Write-Host "❌ Error creating silent desktop shortcut: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n🎉 Desktop shortcuts created successfully!" -ForegroundColor Green
Write-Host "`n📋 Available shortcuts:" -ForegroundColor Cyan
Write-Host "• Portfolio Dashboard.lnk - Standard batch launcher" -ForegroundColor White
Write-Host "• Portfolio Dashboard (PowerShell).lnk - PowerShell launcher" -ForegroundColor White
Write-Host "• Portfolio Dashboard (Silent).lnk - Runs without console window" -ForegroundColor White
Write-Host "`n💡 Double-click any shortcut to start the dashboard!" -ForegroundColor Yellow
Write-Host "🌐 Dashboard will open at: http://localhost:8000" -ForegroundColor Cyan
