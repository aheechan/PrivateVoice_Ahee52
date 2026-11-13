# Build an incremental update zip (app, assets, tools, runtime patch, README, icon)
[CmdletBinding()]
param(
    [string]$Version = "1.0.5",
    [string]$Package = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Package) { $Package = Join-Path $root "dist\PrivateVoice_Ahee52" }

$stage = Join-Path $root "dist\update_stage"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

foreach ($d in "app", "assets", "tools") {
    Copy-Item (Join-Path $Package $d) (Join-Path $stage $d) -Recurse -Force
}
foreach ($f in "README.txt", "app.ico") {
    Copy-Item (Join-Path $Package $f) $stage -Force
}

# Runtime patch: huggingface_hub must stay < 1.0 for the GPT-SoVITS training scripts
$sp = Join-Path $Package "runtime\Lib\site-packages"
$dst = Join-Path $stage "runtime\Lib\site-packages"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item (Join-Path $sp "huggingface_hub") (Join-Path $dst "huggingface_hub") -Recurse -Force
Get-ChildItem $sp -Directory -Filter "huggingface_hub-*" |
    ForEach-Object { Copy-Item $_.FullName $dst -Recurse -Force }

Get-ChildItem $stage -Recurse -Directory -Filter __pycache__ |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

$zip = Join-Path $root "dist\PrivateVoice_Ahee52-$Version-update.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip -CompressionLevel Optimal
Get-Item $zip | Select-Object Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 2) } }
