param(
    [int]$Port = 8791,
    [string]$DataDir = (Join-Path $PSScriptRoot "artifacts\previews\online"),
    [string]$ConfigFile = "",
    [string]$AllowedModels = "",
    [string]$Python = (Join-Path $PSScriptRoot ".venv\Scripts\python.exe")
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if ($Port -lt 1024 -or $Port -gt 65535 -or $Port -in @(8000,8790,8792,8793)) {
    throw "Choose an independent preview port; formal and offline ports are protected."
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "Preview port is occupied. Choose another -Port; no service will be stopped."
}
if (-not (Test-Path -LiteralPath $Python)) { throw "Project Python environment not found." }
$frontend = Join-Path $PSScriptRoot "artifacts\preview-build\frontend-dist"
# Never overwrite frontend/dist used by the formal service. No dependency install.
& npm.cmd run build --prefix frontend -- --outDir ../artifacts/preview-build/frontend-dist
if ($LASTEXITCODE -ne 0) { throw "Preview frontend build failed." }
$arguments = @("-B", "scripts/serve_preview_ai.py", "--port", "$Port", "--data-dir", $DataDir, "--frontend-dir", $frontend)
if ($ConfigFile) { $arguments += @("--config-file", $ConfigFile) }
if ($AllowedModels) { $arguments += @("--allowed-models", $AllowedModels) }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Online preview failed; formal services were not replaced." }
