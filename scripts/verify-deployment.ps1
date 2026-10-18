[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Virtual environment not found: $python"
}

$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONPYCACHEPREFIX = Join-Path $repoRoot 'experiment\cache\pycache'
$env:SANS_AUTO_SKILL_EVOLUTION = '0'

Push-Location $repoRoot
try {
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'pip check failed' }

    & $python -m compileall -q agents
    if ($LASTEXITCODE -ne 0) { throw 'compileall failed' }

    & $python -m agents.main --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'CLI help failed' }

    $baseTemp = Join-Path $repoRoot ("experiment\cache\pytest\deployment-" + [guid]::NewGuid().ToString('N'))
    if (Test-Path -LiteralPath $baseTemp) { throw "Unexpected existing temp path: $baseTemp" }
    & $python -m pytest -q --basetemp $baseTemp tests\test_windows_compat.py
    if ($LASTEXITCODE -ne 0) { throw 'Windows compatibility tests failed' }

    & $python -c "import asyncio; from agents.tools import execute_tool; result=asyncio.run(execute_tool('read_file', {'file_path':'README.md'})); print(result.splitlines()[0])"
    if ($LASTEXITCODE -ne 0) { throw 'read_file smoke test failed' }
}
finally {
    Pop-Location
}
