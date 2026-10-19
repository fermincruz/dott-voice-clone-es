$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$env:PIP_CACHE_DIR = Join-Path $root "pip-cache"
$env:HF_HOME = Join-Path $root "model_cache"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME "hub"

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    py -3.12 -m venv (Join-Path $root ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Python 3.12 environment" }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

& $venvPython -m pip install -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Project requirements installation failed" }

& $venvPython -m pip install --upgrade --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "CPU PyTorch installation failed" }

& $venvPython -m pip install voxcpm
if ($LASTEXITCODE -ne 0) { throw "VoxCPM installation failed" }

& $venvPython -c "import torch, voxcpm; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available()); print('VoxCPM:', voxcpm.__file__)"
if ($LASTEXITCODE -ne 0) { throw "VoxCPM verification failed" }
