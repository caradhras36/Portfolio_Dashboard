# Unicode and Emoji Setup Script
# This script configures the environment for proper unicode/emoji handling

Write-Host "🔧 Setting up Unicode and Emoji Support..." -ForegroundColor Green

# Set console encoding for proper unicode display
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Set Python environment variables for unicode support
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTHONLEGACYWINDOWSSTDIO = "0"

# Set PowerShell execution policy for script running
try {
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    Write-Host "✅ PowerShell execution policy set to RemoteSigned" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not set execution policy: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Test unicode/emoji display
Write-Host "`n🧪 Testing Unicode/Emoji Display:" -ForegroundColor Cyan
Write-Host "📊 Portfolio Dashboard" -ForegroundColor Blue
Write-Host "🚀 FastAPI Backend" -ForegroundColor Green
Write-Host "💰 Financial Data" -ForegroundColor Yellow
Write-Host "📈 Charts & Analytics" -ForegroundColor Magenta
Write-Host "⚡ Real-time Updates" -ForegroundColor Red

# Test special characters
Write-Host "`n🔤 Testing Special Characters:" -ForegroundColor Cyan
Write-Host "Greek letters: α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω" -ForegroundColor White
Write-Host "Math symbols: ∑ ∏ ∫ ∂ ∇ ± × ÷ ≤ ≥ ≠ ≈ ∞" -ForegroundColor White
Write-Host "Currency: $ € £ ¥ ₹ ₽ ₩ ₪ ₫ ₨" -ForegroundColor White

Write-Host "`n✅ Unicode setup complete!" -ForegroundColor Green
Write-Host "💡 You can now use emojis and unicode characters in your scripts and output." -ForegroundColor Cyan
