# Pipeline de preparação — Random Projection (versão PowerShell)
# Equivalente a run_preparation.sh, para rodar no Windows/PowerShell.
# Uso: .\run_preparation.ps1 [tcc1|tcc2] [N]     (padrão: tcc2 1000)
param([string]$Dataset = "tcc2", [int]$N = 1000)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "=== Pipeline de preparação — Random Projection (dataset $Dataset, N=$N) ==="

# Define o Python a usar: venv do projeto se existir, senão o python do PATH
if (Test-Path ".\venv\Scripts\python.exe") {
    $py = ".\venv\Scripts\python.exe"
} elseif (Test-Path ".\.venv\Scripts\python.exe") {
    $py = ".\.venv\Scripts\python.exe"
} else {
    $py = "python"
}

Write-Host "[1/2] Gerando matrizes posicionais k-mer..."
& $py src\01_kmer_matrix.py --dataset $Dataset

Write-Host "[2/2] Projecao Random + vetores finais..."
& $py src\02_project_random.py --dataset $Dataset --n-components $N

if ($Dataset -eq "tcc1") { $proc = "data\processed"; $models = "models" } else { $proc = "data\processed_v2"; $models = "models\v2" }
Write-Host ""
Write-Host "=== Preparacao concluida ==="
Write-Host "Arrays para classificacao:"
Write-Host "  $proc\X.npy        - (n_seq, $N)"
Write-Host "  $proc\y.npy        - (n_seq,) labels 1/0"
Write-Host "  $proc\metadata.json"
Write-Host "Vetores individuais: $proc\vectors\"
Write-Host "Modelo:              $models\random_projection.pkl"
