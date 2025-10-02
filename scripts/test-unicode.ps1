# Unicode and Emoji Testing Script
# This script tests various unicode characters and emojis

Write-Host "🧪 Unicode and Emoji Testing Script" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Gray

# Test basic emojis
Write-Host "`n📱 Basic Emojis:" -ForegroundColor Cyan
$emojis = @("😀", "😃", "😄", "😁", "😆", "😅", "🤣", "😂", "🙂", "🙃", "😉", "😊", "😇", "🥰", "😍", "🤩", "😘", "😗", "😚", "😙", "😋", "😛", "😜", "🤪", "😝", "🤑", "🤗", "🤭", "🤫", "🤔", "🤐", "🤨", "😐", "😑", "😶", "😏", "😒", "🙄", "😬", "🤥", "😌", "😔", "😪", "🤤", "😴", "😷", "🤒", "🤕", "🤢", "🤮", "🤧", "🥵", "🥶", "🥴", "😵", "🤯", "🤠", "🥳", "😎", "🤓", "🧐", "😕", "😟", "🙁", "☹️", "😮", "😯", "😲", "😳", "🥺", "😦", "😧", "😨", "😰", "😥", "😢", "😭", "😱", "😖", "😣", "😞", "😓", "😩", "😫", "🥱", "😤", "😡", "😠", "🤬", "😈", "👿", "💀", "☠️", "💩", "🤡", "👹", "👺", "👻", "👽", "👾", "🤖", "😺", "😸", "😹", "😻", "😼", "😽", "🙀", "😿", "😾")
for ($i = 0; $i -lt [Math]::Min(20, $emojis.Length); $i++) {
    Write-Host $emojis[$i] -NoNewline
    if (($i + 1) % 10 -eq 0) { Write-Host "" }
}
Write-Host "`n"

# Test financial emojis
Write-Host "💰 Financial Emojis:" -ForegroundColor Cyan
$financialEmojis = @("💰", "💵", "💴", "💶", "💷", "💸", "💳", "💎", "⚖️", "🔍", "📊", "📈", "📉", "📋", "📌", "📍", "📎", "📏", "📐", "✂️", "🗂️", "🗃️", "🗄️", "🗑️", "🔒", "🔓", "🔏", "🔐", "🔑", "🗝️", "🔨", "⛏️", "⚒️", "🛠️", "🔧", "🔩", "⚙️", "🗜️", "⚖️", "🔗", "⛓️", "🧰", "🧲", "⚗️", "🧪", "🧫", "🧬", "🔬", "🔭", "📡", "💉", "💊", "🩹", "🩺", "🚪", "🛏️", "🛋️", "🚽", "🚿", "🛁", "🛀", "🧴", "🧷", "🧹", "🧺", "🧻", "🚰", "🚰", "🧼", "🧽", "🧯", "🛒", "🚬", "⚰️", "⚱️", "🗿", "🏧", "🚮", "🚰", "♿", "🚹", "🚺", "🚻", "🚼", "🚾", "🛂", "🛃", "🛄", "🛅", "⚠️", "🚸", "⛔", "🚫", "🚳", "🚭", "🚯", "🚱", "🚷", "📵", "🔞", "☢️", "☣️", "⬆️", "↗️", "➡️", "↘️", "⬇️", "↙️", "⬅️", "↖️", "↕️", "↔️", "↩️", "↪️", "⤴️", "⤵️", "🔃", "🔄", "🔙", "🔚", "🔛", "🔜", "🔝")
for ($i = 0; $i -lt [Math]::Min(20, $financialEmojis.Length); $i++) {
    Write-Host $financialEmojis[$i] -NoNewline
    if (($i + 1) % 10 -eq 0) { Write-Host "" }
}
Write-Host "`n"

# Test Greek letters (important for financial calculations)
Write-Host "`n🔤 Greek Letters (for financial calculations):" -ForegroundColor Cyan
$greekLetters = @("α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω", "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ", "Ι", "Κ", "Λ", "Μ", "Ν", "Ξ", "Ο", "Π", "Ρ", "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω")
Write-Host "Lowercase: " -NoNewline
for ($i = 0; $i -lt 24; $i++) {
    Write-Host $greekLetters[$i] -NoNewline
    if (($i + 1) % 12 -eq 0) { Write-Host ""; Write-Host "          " -NoNewline }
}
Write-Host "`nUppercase: " -NoNewline
for ($i = 24; $i -lt 48; $i++) {
    Write-Host $greekLetters[$i] -NoNewline
    if (($i + 1) % 12 -eq 0) { Write-Host ""; Write-Host "          " -NoNewline }
}
Write-Host "`n"

# Test mathematical symbols
Write-Host "`n🧮 Mathematical Symbols:" -ForegroundColor Cyan
$mathSymbols = @("∑", "∏", "∫", "∂", "∇", "±", "×", "÷", "≤", "≥", "≠", "≈", "∞", "√", "∛", "∜", "∝", "∟", "∠", "∡", "∢", "∣", "∤", "∥", "∦", "∧", "∨", "∩", "∪", "∫", "∬", "∭", "∮", "∯", "∰", "∱", "∲", "∳", "∴", "∵", "∶", "∷", "∸", "∹", "∺", "∻", "∼", "∽", "∾", "∿", "≀", "≁", "≂", "≃", "≄", "≅", "≆", "≇", "≈", "≉", "≊", "≋", "≌", "≍", "≎", "≏", "≐", "≑", "≒", "≓", "≔", "≕", "≖", "≗", "≘", "≙", "≚", "≛", "≜", "≝", "≞", "≟", "≠", "≡", "≢", "≣", "≤", "≥", "≦", "≧", "≨", "≩", "≪", "≫", "≬", "≭", "≮", "≯", "≰", "≱", "≲", "≳", "≴", "≵", "≶", "≷", "≸", "≹", "≺", "≻", "≼", "≽", "≾", "≿", "⊀", "⊁", "⊂", "⊃", "⊄", "⊅", "⊆", "⊇", "⊈", "⊉", "⊊", "⊋", "⊌", "⊍", "⊎", "⊏", "⊐", "⊑", "⊒", "⊓", "⊔", "⊕", "⊖", "⊗", "⊘", "⊙", "⊚", "⊛", "⊜", "⊝", "⊞", "⊟", "⊠", "⊡", "⊢", "⊣", "⊤", "⊥", "⊦", "⊧", "⊨", "⊩", "⊪", "⊫", "⊬", "⊭", "⊮", "⊯", "⊰", "⊱", "⊲", "⊳", "⊴", "⊵", "⊶", "⊷", "⊸", "⊹", "⊺", "⊻", "⊼", "⊽", "⊾", "⊿", "⋀", "⋁", "⋂", "⋃", "⋄", "⋅", "⋆", "⋇", "⋈", "⋉", "⋊", "⋋", "⋌", "⋍", "⋎", "⋏", "⋐", "⋑", "⋒", "⋓", "⋔", "⋕", "⋖", "⋗", "⋘", "⋙", "⋚", "⋛", "⋜", "⋝", "⋞", "⋟", "⋠", "⋡", "⋢", "⋣", "⋤", "⋥", "⋦", "⋧", "⋨", "⋩", "⋪", "⋫", "⋬", "⋭", "⋮", "⋯", "⋰", "⋱", "⋲", "⋳", "⋴", "⋵", "⋶", "⋷", "⋸", "⋹", "⋺", "⋻", "⋼", "⋽", "⋾", "⋿")
for ($i = 0; $i -lt [Math]::Min(30, $mathSymbols.Length); $i++) {
    Write-Host $mathSymbols[$i] -NoNewline
    if (($i + 1) % 15 -eq 0) { Write-Host ""; Write-Host "              " -NoNewline }
}
Write-Host "`n"

# Test currency symbols
Write-Host "`n💱 Currency Symbols:" -ForegroundColor Cyan
$currencies = @("$", "€", "£", "¥", "₹", "₽", "₩", "₪", "₫", "₨", "₦", "₡", "₱", "₲", "₴", "₵", "₸", "₺", "₼", "₽", "₾", "₿", "＄", "￠", "￡", "￢", "￣", "￤", "￥", "￦")
for ($i = 0; $i -lt $currencies.Length; $i++) {
    Write-Host $currencies[$i] -NoNewline
    if (($i + 1) % 10 -eq 0) { Write-Host ""; Write-Host "              " -NoNewline }
}
Write-Host "`n"

# Test special characters that might be used in financial data
Write-Host "`n📊 Special Characters for Financial Data:" -ForegroundColor Cyan
$specialChars = @("°", "′", "″", "‴", "‵", "‶", "‷", "‸", "‹", "›", "※", "‼", "‽", "‾", "‿", "⁀", "⁁", "⁂", "⁃", "⁅", "⁆", "⁇", "⁈", "⁉", "⁊", "⁋", "⁌", "⁍", "⁎", "⁏", "⁐", "⁑", "⁒", "⁓", "⁔", "⁕", "⁖", "⁗", "⁘", "⁙", "⁚", "⁛", "⁜", "⁝", "⁞", "⁰", "ⁱ", "⁲", "⁳", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹", "⁺", "⁻", "⁼", "⁽", "⁾", "ⁿ", "₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉", "₊", "₋", "₌", "₍", "₎", "ₐ", "ₑ", "ₒ", "ₓ", "ₔ", "ₕ", "ₖ", "ₗ", "ₘ", "ₙ", "ₚ", "ₛ", "ₜ", "₝", "₞", "₟", "₠", "₡", "₢", "₣", "₤", "₥", "₦", "₧", "₨", "₩", "₪", "₫", "€", "₭", "₮", "₯", "₰", "₱", "₲", "₳", "₴", "₵", "₶", "₷", "₸", "₹", "₺", "₻", "₼", "₽", "₾", "₿")
for ($i = 0; $i -lt [Math]::Min(25, $specialChars.Length); $i++) {
    Write-Host $specialChars[$i] -NoNewline
    if (($i + 1) % 10 -eq 0) { Write-Host ""; Write-Host "              " -NoNewline }
}
Write-Host "`n"

Write-Host "`n✅ Unicode testing complete!" -ForegroundColor Green
Write-Host "💡 If you can see all the characters above, unicode support is working correctly." -ForegroundColor Cyan
