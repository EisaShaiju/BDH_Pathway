# BDH Narrative Consistency Classifier - Quick Start Script (PowerShell)

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "BDH Narrative Classifier" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# Check if Docker is installed
try {
    docker --version | Out-Null
} catch {
    Write-Host "Error: Docker is not installed" -ForegroundColor Red
    exit 1
}

# Build Docker image
Write-Host "`n[1/3] Building Docker image..." -ForegroundColor Yellow
docker-compose build

# Train model
Write-Host "`n[2/3] Training model..." -ForegroundColor Yellow
docker-compose --profile train up train

# Run inference
Write-Host "`n[3/3] Running inference..." -ForegroundColor Yellow
docker-compose --profile inference up inference

Write-Host "`n==================================" -ForegroundColor Green
Write-Host "✓ Pipeline completed!" -ForegroundColor Green
Write-Host "Results saved to: outputs/submission.csv" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green
