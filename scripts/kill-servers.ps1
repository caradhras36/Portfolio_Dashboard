# Kill All Servers Script
# This script stops all running Python servers and processes

Write-Host "🛑 Killing All Servers..." -ForegroundColor Red
Write-Host "=" * 40 -ForegroundColor Gray

# Set up unicode environment
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Function to kill processes safely
function Stop-ProcessSafely {
    param($ProcessName, $Description)
    
    try {
        $processes = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
        if ($processes) {
            Write-Host "🔍 Found $($processes.Count) $Description process(es)" -ForegroundColor Yellow
            $processes | Stop-Process -Force
            Write-Host "✅ Killed $Description process(es)" -ForegroundColor Green
        } else {
            Write-Host "ℹ️  No $Description processes found" -ForegroundColor Cyan
        }
    } catch {
        Write-Host "⚠️  Error killing $Description : $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Kill Python processes
Write-Host "`n🐍 Checking Python processes..." -ForegroundColor Cyan
Stop-ProcessSafely "python" "Python"

# Kill uvicorn processes
Write-Host "`n🚀 Checking uvicorn processes..." -ForegroundColor Cyan
Stop-ProcessSafely "uvicorn" "uvicorn"

# Kill main.py processes
Write-Host "`n📄 Checking main.py processes..." -ForegroundColor Cyan
Stop-ProcessSafely "main" "main.py"

# Check for processes using port 8000
Write-Host "`n🌐 Checking port 8000..." -ForegroundColor Cyan
try {
    $port8000 = netstat -ano | Select-String ":8000"
    if ($port8000) {
        Write-Host "🔍 Found connections on port 8000:" -ForegroundColor Yellow
        $port8000 | ForEach-Object { Write-Host "  $_" -ForegroundColor White }
        
        # Extract PIDs and kill them
        $pids = $port8000 | ForEach-Object {
            $parts = $_ -split '\s+'
            if ($parts.Length -gt 4) {
                $parts[4]
            }
        } | Where-Object { $_ -match '^\d+$' } | Sort-Object -Unique
        
        if ($pids) {
            Write-Host "🔪 Killing processes using port 8000..." -ForegroundColor Red
            $pids | ForEach-Object {
                try {
                    Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
                    Write-Host "  ✅ Killed PID $_" -ForegroundColor Green
                } catch {
                    Write-Host "  ⚠️  Could not kill PID $_" -ForegroundColor Yellow
                }
            }
        }
    } else {
        Write-Host "✅ No active connections on port 8000" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  Error checking port 8000: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Check for any remaining Python-related processes
Write-Host "`n🔍 Final check for Python-related processes..." -ForegroundColor Cyan
try {
    $remaining = Get-Process | Where-Object {
        $_.ProcessName -like "*python*" -or 
        $_.ProcessName -like "*uvicorn*" -or 
        $_.ProcessName -like "*main*" -or
        $_.ProcessName -like "*fastapi*"
    }
    
    if ($remaining) {
        Write-Host "🔍 Found remaining processes:" -ForegroundColor Yellow
        $remaining | ForEach-Object {
            Write-Host "  $($_.ProcessName) (PID: $($_.Id))" -ForegroundColor White
        }
        Write-Host "🔪 Killing remaining processes..." -ForegroundColor Red
        $remaining | Stop-Process -Force
        Write-Host "✅ Killed remaining processes" -ForegroundColor Green
    } else {
        Write-Host "✅ No remaining Python-related processes found" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  Error in final check: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "`n🎉 Server cleanup complete!" -ForegroundColor Green
Write-Host "💡 All Python servers and processes have been stopped" -ForegroundColor Cyan
Write-Host "🌐 Port 8000 is now available for new servers" -ForegroundColor Cyan
