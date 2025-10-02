# Kill All Servers Script
Write-Host "🛑 Killing All Servers..." -ForegroundColor Red

# Kill Python processes
Write-Host "🐍 Stopping Python processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# Kill uvicorn processes  
Write-Host "🚀 Stopping uvicorn processes..." -ForegroundColor Yellow
Get-Process uvicorn -ErrorAction SilentlyContinue | Stop-Process -Force

# Kill main processes
Write-Host "📄 Stopping main processes..." -ForegroundColor Yellow
Get-Process main -ErrorAction SilentlyContinue | Stop-Process -Force

# Check port 8000
Write-Host "🌐 Checking port 8000..." -ForegroundColor Yellow
$port8000 = netstat -ano | Select-String ":8000"
if ($port8000) {
    Write-Host "Found connections on port 8000" -ForegroundColor Red
    $port8000 | ForEach-Object { Write-Host $_ -ForegroundColor White }
} else {
    Write-Host "✅ Port 8000 is free" -ForegroundColor Green
}

Write-Host "✅ All servers stopped!" -ForegroundColor Green
