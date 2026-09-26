$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$minPythonVersion = "3.10"

if (Test-Path "$scriptDir\venv\Scripts\python.exe") {
    $pythonBin = "$scriptDir\venv\Scripts\python.exe"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonBin = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonBin = "python3"
} else {
    Write-Error "Python is not installed"
    exit 1
}

& $pythonBin -c "import sys; sys.exit(0 if sys.version_info >= tuple(map(int, '$minPythonVersion'.split('.'))) else 1)"
if ($LASTEXITCODE -ne 0) {
    $version = & $pythonBin --version 2>&1
    Write-Error "Erro: e necessario Python >= $minPythonVersion (encontrado: $version em $pythonBin)."
    Write-Error "Veja o arquivo .python-version para a versao recomendada."
    exit 1
}

& $pythonBin src/main.py
