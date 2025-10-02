# Simple Desktop Shortcut Creator
Write-Host "Creating desktop shortcut..." -ForegroundColor Green

$WshShell = New-Object -comObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ProjectPath = Get-Location

# Create shortcut for batch file
$Shortcut = $WshShell.CreateShortcut("$DesktopPath\Portfolio Dashboard.lnk")
$Shortcut.TargetPath = "$ProjectPath\PortfolioDashboard.bat"
$Shortcut.WorkingDirectory = $ProjectPath.Path
$Shortcut.Description = "Portfolio Dashboard - Financial Portfolio Management System"
$Shortcut.IconLocation = "shell32.dll,137"
$Shortcut.Save()

Write-Host "✅ Desktop shortcut created: Portfolio Dashboard.lnk" -ForegroundColor Green
Write-Host "💡 Double-click the shortcut to start the dashboard!" -ForegroundColor Cyan
