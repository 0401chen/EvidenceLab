param(
    [string]$Python = $env:EVIDENCELAB_PYTHON,
    [string]$ConfigFile = $env:EVIDENCELAB_CONFIG_FILE,
    [string]$DataDir = (Join-Path $PSScriptRoot "artifacts\demo-data"),
    [string]$AuditDir = ""
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (Get-NetTCPConnection -State Listen -LocalPort 8792 -ErrorAction SilentlyContinue) {
    throw "8792 is already in use. This script will not stop or replace another service."
}
if (-not $Python) { $Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $Python)) { throw "Set -Python to an installed EvidenceLab Python environment." }
if (-not $ConfigFile -and (Test-Path -LiteralPath (Join-Path $PSScriptRoot ".env"))) {
    $ConfigFile = Join-Path $PSScriptRoot ".env"
}
$env:PYTHONDONTWRITEBYTECODE = "1"
$arguments = @("-B", "scripts/serve_demo_ai.py", "--data-dir", $DataDir, "--port", "8792")
if ($ConfigFile) { $arguments += @("--config-file", $ConfigFile) }
if ($AuditDir) { $arguments += @("--audit-dir", $AuditDir) }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Demo service stopped with an error; check config and existing library paths." }
