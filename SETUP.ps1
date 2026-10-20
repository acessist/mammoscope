$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (!(Test-Path '.venv/Scripts/python.exe')) {
    if (!(Get-Command py -ErrorAction SilentlyContinue)) {
        throw 'Install Python 3.12 with the Windows Python launcher, then run SETUP.ps1 again.'
    }
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python 3.12 environment.' }
}
& '.venv/Scripts/python.exe' -m ensurepip
if ($LASTEXITCODE -ne 0) { throw 'Could not initialize pip.' }
& '.venv/Scripts/python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Host 'Setup complete. Open START.cmd to launch MammoScope.'
