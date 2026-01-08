#!/bin/bash

# BDH Narrative Classifier - Quick Start Script

echo "=================================="
echo "BDH Narrative Classifier"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Build Docker image
echo -e "\n[1/3] Building Docker image..."
docker-compose build

# Train model
echo -e "\n[2/3] Training model..."
docker-compose --profile train up train

# Run inference
echo -e "\n[3/3] Running inference..."
docker-compose --profile inference up inference

echo -e "\n=================================="
echo "✓ Pipeline completed!"
echo "Results saved to: outputs/submission.csv"
echo "=================================="
