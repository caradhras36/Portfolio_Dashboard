# Simple Unicode Test Script
Write-Host "🧪 Testing Unicode and Emoji Support" -ForegroundColor Green
Write-Host "=" * 40 -ForegroundColor Gray

# Set up unicode environment
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Test basic emojis
Write-Host "`n📱 Basic Emojis:" -ForegroundColor Cyan
Write-Host "😀 😃 😄 😁 😆 😅 🤣 😂 🙂 🙃 😉 😊 😇 🥰 😍 🤩 😘 😗 😚 😙" -ForegroundColor White

# Test financial emojis
Write-Host "`n💰 Financial Emojis:" -ForegroundColor Cyan
Write-Host "💰 💵 💴 💶 💷 💸 💳 💎 📊 📈 📉 📋 📌 📍 📎 📏 📐 ✂️ 🗂️" -ForegroundColor White

# Test Greek letters
Write-Host "`n🔤 Greek Letters:" -ForegroundColor Cyan
Write-Host "α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω" -ForegroundColor White
Write-Host "Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω" -ForegroundColor White

# Test mathematical symbols
Write-Host "`n🧮 Mathematical Symbols:" -ForegroundColor Cyan
Write-Host "∑ ∏ ∫ ∂ ∇ ± × ÷ ≤ ≥ ≠ ≈ ∞ √ ∛ ∜ ∝ ∟ ∠ ∡ ∢ ∣ ∤ ∥ ∦" -ForegroundColor White

# Test currency symbols
Write-Host "`n💱 Currency Symbols:" -ForegroundColor Cyan
Write-Host "$ € £ ¥ ₹ ₽ ₩ ₪ ₫ ₨ ₦ ₡ ₱ ₲ ₴ ₵ ₸ ₺ ₼ ₽ ₾ ₿" -ForegroundColor White

# Test special characters
Write-Host "`n📊 Special Characters:" -ForegroundColor Cyan
Write-Host "° ′ ″ ‴ ‵ ‶ ‷ ‸ ‹ › ※ ‼ ‽ ‾ ‿ ⁀ ⁁ ⁂ ⁃ ⁅ ⁆ ⁇ ⁈ ⁉" -ForegroundColor White

Write-Host "`n✅ Unicode test complete!" -ForegroundColor Green
Write-Host "💡 If you can see all the characters above, unicode support is working!" -ForegroundColor Cyan
