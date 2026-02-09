Param(
    [string]$PythonVersion = "3.11",
    [string]$VenvPath = ".venv-ml"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Checking Python $PythonVersion"
$pythonLauncherFound = $false
$pyList = ""
try {
    $pyList = py -0p
    if ($pyList -match $PythonVersion) {
        $pythonLauncherFound = $true
    }
} catch {
    $pythonLauncherFound = $false
}

$pythonExeOverride = Join-Path $env:LOCALAPPDATA "Programs\\Python\\Python311\\python.exe"
if (-not $pythonLauncherFound -and -not (Test-Path $pythonExeOverride)) {
    throw "Python $PythonVersion not found. Install it first (winget install -e --id Python.Python.3.11)."
}

Write-Host "[2/4] Creating venv at $VenvPath"
if ($pythonLauncherFound) {
    py -$PythonVersion -m venv $VenvPath
}
else {
    & $pythonExeOverride -m venv $VenvPath
}

$pythonExe = Join-Path $VenvPath "Scripts\\python.exe"
Write-Host "[3/4] Upgrading pip"
& $pythonExe -m pip install --upgrade pip

Write-Host "[4/4] Installing requirements"
& $pythonExe -m pip install -r ml/requirements.txt

Write-Host "Environment ready. Activate with: $VenvPath\\Scripts\\Activate.ps1"
