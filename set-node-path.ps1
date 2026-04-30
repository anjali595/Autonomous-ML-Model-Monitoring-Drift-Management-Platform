$profilePath = 'C:\Users\Asus\OneDrive\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1'
$nodePath = 'D:\Mini-Project\my-monitoring-app\frontend\node\node-v24.14.1-win-x64'

# Ensure profile directory exists
New-Item -ItemType Directory -Force -Path (Split-Path $profilePath) | Out-Null

# Write clean profile snippet to prepend portable Node to PATH if present
$content = @"
$nodePath = "$nodePath"
if (Test-Path "$nodePath\node.exe") {
  if (-not ($env:PATH -split ';' | Where-Object { $_ -ieq $nodePath })) {
    $env:PATH = "$nodePath;$env:PATH"
  }
}
"@

Set-Content -Path $profilePath -Value $content -Force
