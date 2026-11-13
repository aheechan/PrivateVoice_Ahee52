# Repository sanity check before pushing to GitHub
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "== tracked files ==" -ForegroundColor Cyan
git ls-files | Measure-Object | Select-Object -ExpandProperty Count

Write-Host "`n== large or forbidden paths ==" -ForegroundColor Cyan
$bad = git ls-files | Select-String -Pattern '(^|/)(models_cache|characters|datasets|results|logs|dist|\.venv|\.uvcache|nltk_data)/|\.(pth|ckpt|bin|safetensors|wav|mp3)$'
if ($bad) {
    $bad | ForEach-Object { Write-Host "  BLOCKED: $($_.Line)" -ForegroundColor Red }
} else {
    Write-Host "  clean"
}

Write-Host "`n== secrets ==" -ForegroundColor Cyan
$hits = git grep -n -I -E 'sk-[A-Za-z0-9]{16,}|api_key_enc: *[A-Za-z0-9+/=]{40,}' -- . 2>$null
if ($hits) { $hits } else { Write-Host "  none" }

Write-Host "`n== remote ==" -ForegroundColor Cyan
git remote -v

Write-Host "`n== status ==" -ForegroundColor Cyan
git status --short
