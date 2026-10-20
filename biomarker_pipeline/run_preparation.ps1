# Pipeline de preparação — Random Projection (versão PowerShell)
# Equivalente a run_preparation.sh, para rodar no Windows/PowerShell.
# Uso: .\run_preparation.ps1 [tcc1|tcc2] [N] [--k K | --pattern MASCARA...]   (padrão: tcc2 1000, k=3)
#      N=0 → sem projeção. Ex.: .\run_preparation.ps1 tcc2 4000 --k 4
param([string]$Dataset = "tcc2", [int]$N = 1000,
      [Parameter(ValueFromRemainingArguments = $true)][string[]]$RepArgs = @())
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "=== Pipeline de preparação — Random Projection (dataset $Dataset, N=$N, representação: $RepArgs) ==="

# Define o Python a usar: venv do projeto (venv/) se existir, senão .venv/, senão o python do PATH
if (Test-Path ".\venv\Scripts\python.exe") {
    $py = ".\venv\Scripts\python.exe"
} elseif (Test-Path ".\.venv\Scripts\python.exe") {
    $py = ".\.venv\Scripts\python.exe"
} else {
    $py = "python"
}

Write-Host "[1/2] Gerando matrizes posicionais k-mer..."
& $py src\01_kmer_matrix.py --dataset $Dataset @RepArgs

Write-Host "[2/2] Projecao Random + vetores finais..."
& $py src\02_project_random.py --dataset $Dataset --n-components $N @RepArgs

if ($Dataset -eq "tcc1") { $proc = "data\processed" } else { $proc = "data\processed_v2" }
Write-Host ""
Write-Host "=== Preparacao concluida ==="
Write-Host "Arrays para classificacao (representacoes alem do k=3 continuo ficam em $proc\{representacao}\):"
Write-Host "  $proc\X_n{dim}.npy - (n_seq, dim); dim = N, ou o numero de k-mers se N=0"
Write-Host "  $proc\y.npy        - (n_seq,) labels 1/0"
Write-Host "  $proc\metadata.json"
