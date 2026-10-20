param(
    [string]$Python = $env:EVIDENCELAB_PYTHON,
    [string]$DataDir = (Join-Path $PSScriptRoot "artifacts\demo-data")
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (Get-NetTCPConnection -State Listen -LocalPort 8792 -ErrorAction SilentlyContinue) {
    throw "8792 is already in use. Stop only its known idle demo process before switching mode."
}
if (-not $Python) { $Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $Python)) { throw "Set -Python to an installed EvidenceLab Python environment." }
$env:PYTHONDONTWRITEBYTECODE = "1"
& $Python -B scripts/serve_offline.py --data-dir $DataDir --port 8792
