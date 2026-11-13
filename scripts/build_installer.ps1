# Build the Windows installer from packaging\installer.iss
[CmdletBinding()]
param(
    [string]$Iscc = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $Iscc)) { throw "ISCC not found: $Iscc" }
if (-not (Test-Path (Join-Path $root "dist\PrivateVoice_Ahee52"))) {
    throw "Missing package folder: dist\PrivateVoice_Ahee52"
}

# Inno cannot overwrite files held by a running app
Get-Process pythonw, python -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "*PrivateVoice*" } |
    ForEach-Object { try { $_.Kill() } catch {} }

$out = Join-Path $root "dist\installer"
if (Test-Path $out) { Remove-Item $out -Recurse -Force }

& $Iscc /Qp (Join-Path $root "packaging\installer.iss")
if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE" }

Get-ChildItem $out | Select-Object Name, @{n = "GB"; e = { [math]::Round($_.Length / 1GB, 2) } }
