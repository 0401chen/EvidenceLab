param(
    [switch]$Install,
    [int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host 'Creating Python virtual environment...'
    & python -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11+ is required.' }
    $Install = $true
}
if ($Install) {
    & $pythonPath -m pip install -r (Join-Path $projectRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
}

$frontendRoot = Join-Path $projectRoot 'frontend'
$frontendIndex = Join-Path $frontendRoot 'dist\index.html'
if ($Install -or -not (Test-Path -LiteralPath $frontendIndex)) {
    Push-Location -LiteralPath $frontendRoot
    try {
        if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot 'node_modules'))) {
            & npm.cmd ci
            if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
        }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }
}
Write-Host "EvidenceLab: http://127.0.0.1:$Port"
Write-Host 'Stop with Ctrl+C. Model settings are read from the project .env file.'
& $pythonPath -m uvicorn backend.app.main:app --host 127.0.0.1 --port $Port
