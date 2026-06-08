# Pipeline de preparação — Random Projection (versão PowerShell)
# Equivalente a run_preparation.sh, para rodar no Windows/PowerShell.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "=== Pipeline de preparação — Random Projection ==="

# Define o Python a usar: venv do projeto se existir, senão o python do PATH
if (Test-Path ".\venv\Scripts\python.exe") {
    $py = ".\venv\Scripts\python.exe"
} elseif (Test-Path ".\.venv\Scripts\python.exe") {
    $py = ".\.venv\Scripts\python.exe"
} else {
    $py = "python"
}

Write-Host "[1/2] Gerando matrizes posicionais k-mer..."
& $py src\01_kmer_matrix.py

Write-Host "[2/2] Projecao Random + vetores finais..."
& $py src\02_project_random.py

Write-Host ""
Write-Host "=== Preparacao concluida ==="
Write-Host "Arrays para classificacao:"
Write-Host "  data\processed\X.npy        - (n_seq, 500)"
Write-Host "  data\processed\y.npy        - (n_seq,) labels 1/0"
Write-Host "  data\processed\metadata.json"
Write-Host "Vetores individuais: data\processed\vectors\"
Write-Host "Modelo:              models\random_projection.pkl"
