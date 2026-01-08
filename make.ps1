# BDH Classifier - Command Shortcuts
# Usage: .\make.ps1 <command>

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

function Show-Help {
    Write-Host @"
BDH Narrative Classifier - Available Commands

BUILD:
  .\make.ps1 build              Build Docker image
  .\make.ps1 install            Install Python dependencies locally

TRAIN:
  .\make.ps1 train              Train model (Docker)
  .\make.ps1 train-local        Train model (local Python)
  .\make.ps1 train-cpu          Train on CPU (local)

INFERENCE:
  .\make.ps1 infer              Run inference (Docker)
  .\make.ps1 infer-local        Run inference (local Python)
  .\make.ps1 infer-pathway      Run with Pathway streaming

EVALUATION:
  .\make.ps1 eval               Evaluate model on validation set

UTILITIES:
  .\make.ps1 clean              Clean outputs and models
  .\make.ps1 logs               View Docker logs
  .\make.ps1 shell              Open shell in Docker container

PIPELINE:
  .\make.ps1 pipeline           Run full pipeline (build → train → infer)
  .\make.ps1 quick              Quick test (CPU, small epochs)

"@ -ForegroundColor Cyan
}

switch ($Command) {
    "build" {
        Write-Host "Building Docker image..." -ForegroundColor Yellow
        docker-compose build
    }
    
    "install" {
        Write-Host "Installing dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
    }
    
    "train" {
        Write-Host "Training model (Docker)..." -ForegroundColor Yellow
        docker-compose --profile train up train
    }
    
    "train-local" {
        Write-Host "Training model (local)..." -ForegroundColor Yellow
        python train.py --train-csv train.csv --output models/bdh_classifier.pt
    }
    
    "train-cpu" {
        Write-Host "Training on CPU..." -ForegroundColor Yellow
        python train.py --train-csv train.csv --device cpu --batch-size 4 --output models/bdh_cpu.pt
    }
    
    "infer" {
        Write-Host "Running inference (Docker)..." -ForegroundColor Yellow
        docker-compose --profile inference up inference
    }
    
    "infer-local" {
        Write-Host "Running inference (local)..." -ForegroundColor Yellow
        python classify.py --test-csv test.csv --model models/bdh_classifier.pt --output outputs/submission.csv
    }
    
    "infer-pathway" {
        Write-Host "Running Pathway inference..." -ForegroundColor Yellow
        python pathway_pipeline.py --input test.csv --output outputs/submission.csv --model models/bdh_classifier.pt --mode pathway
    }
    
    "eval" {
        Write-Host "Evaluating model..." -ForegroundColor Yellow
        python evaluate.py --model models/bdh_classifier.pt --train-csv train.csv
    }
    
    "clean" {
        Write-Host "Cleaning outputs..." -ForegroundColor Yellow
        if (Test-Path "outputs") { Remove-Item -Recurse -Force outputs\* }
        if (Test-Path "models") { Remove-Item -Recurse -Force models\*.pt }
        Write-Host "✓ Cleaned" -ForegroundColor Green
    }
    
    "logs" {
        Write-Host "Docker logs:" -ForegroundColor Yellow
        docker-compose logs
    }
    
    "shell" {
        Write-Host "Opening Docker shell..." -ForegroundColor Yellow
        docker run -it --rm -v ${PWD}:/app bdh-classifier:latest /bin/bash
    }
    
    "pipeline" {
        Write-Host "Running full pipeline..." -ForegroundColor Yellow
        .\run.ps1
    }
    
    "quick" {
        Write-Host "Quick test (CPU, 10 epochs)..." -ForegroundColor Yellow
        python train.py --train-csv train.csv --device cpu --batch-size 4 --epochs 10 --output models/bdh_quick.pt
        python classify.py --test-csv test.csv --model models/bdh_quick.pt --device cpu --output outputs/quick_submission.csv
        Write-Host "✓ Done! See outputs/quick_submission.csv" -ForegroundColor Green
    }
    
    "help" {
        Show-Help
    }
    
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Show-Help
    }
}
