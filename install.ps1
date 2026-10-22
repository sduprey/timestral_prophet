# Sets up a uv-managed virtual environment for timestral_vs_prophet.ipynb.
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File .\install.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$PythonVersion = "3.11"

# Versions pinned to the commits used by Mistral's own fev-bench submission.
$Packages = @(
    "fev @ git+https://github.com/autogluon/fev@f9e2c7d3152ed7d15c3b1c2877bb3ae7d977046f",
    "timestral @ git+https://github.com/mistralai/timestral@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff",
    "prophet",
    "pandas",
    "matplotlib",
    "pyyaml",
    "joblib",
    "statsmodels",
    "ipykernel"
)

uv venv .venv --python $PythonVersion
if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }

uv pip install --python .venv $Packages
if ($LASTEXITCODE -ne 0) { throw "uv pip install failed" }

# Register the env as a Jupyter kernel so the notebook can select it.
.\.venv\Scripts\python.exe -m ipykernel install --user --name timestral --display-name "Python (timestral)"

Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1  (or pick the 'Python (timestral)' kernel)"
