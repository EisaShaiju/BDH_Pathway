# BDH Narrative Classifier - Complete Usage Guide

## Table of Contents

1. [Installation Options](#installation)
2. [Docker Quickstart](#docker-quickstart)
3. [Local Development](#local-development)
4. [Pathway Streaming vs Batch](#pathway-modes)
5. [Complete Workflow](#complete-workflow)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Option 1: Docker (Recommended)

**Requirements:**

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- 8GB+ RAM
- NVIDIA GPU + nvidia-docker (optional for GPU acceleration)

**Verify Installation:**

```powershell
# Windows PowerShell
docker --version
docker-compose --version
```

```bash
# Linux/Mac
docker --version
docker-compose --version
```

### Option 2: Local Python Environment

**Requirements:**

- Python 3.11+
- CUDA 11.8+ (for GPU support)
- 8GB+ RAM

**Setup:**

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Docker Quickstart

### One-Command Pipeline (Windows)

```powershell
# Run full pipeline: build → train → inference
.\run.ps1
```

### One-Command Pipeline (Linux/Mac)

```bash
# Make executable
chmod +x run.sh

# Run
./run.sh
```

### Step-by-Step Docker

**1. Build Image:**

```bash
docker-compose build
```

**2. Train Model (GPU):**

```bash
docker-compose --profile train up train
```

Expected output:

```
Epoch 1/50
Training: 100%|█████| 5/5 [00:02<00:00]
Validation: 100%|█████| 2/2 [00:00<00:00]
Train Loss: 0.6234 | Val Loss: 0.5912
✓ Saved best model
```

**3. Run Inference:**

```bash
# Batch mode (simpler, faster)
docker-compose --profile inference up inference

# Pathway streaming mode
docker-compose --profile inference-pathway up inference-pathway
```

**4. CPU-Only Mode:**

```bash
docker-compose --profile inference-cpu up inference-cpu
```

**5. View Results:**

```bash
cat outputs/submission.csv
```

---

## Local Development

### Training

**Basic Training:**

```bash
python train.py \
    --train-csv train.csv \
    --output models/bdh_classifier.pt
```

**With Custom Hyperparameters:**

```bash
python train.py \
    --train-csv train.csv \
    --batch-size 8 \
    --epochs 100 \
    --lr 0.0005 \
    --device cuda \
    --output models/bdh_custom.pt
```

**Fine-tuning from Pre-trained:**

```bash
python train.py \
    --train-csv train.csv \
    --pretrained models/bdh_pretrained.pt \
    --freeze-bdh \
    --output models/bdh_finetuned.pt
```

**Training Output:**

```
Loading data...
Train samples: 65, Val samples: 16
Class distribution - Train: [41 24], Val: [10  6]

Initializing model...
Model parameters: 2,456,832

==================================================
Epoch 1/50
==================================================

Training: 100%|███████████| 5/5 [00:02<00:00,  2.13it/s, loss=0.687]
Validation: 100%|█████████| 2/2 [00:00<00:00,  8.45it/s, loss=0.634]

Train Loss: 0.6821 | Val Loss: 0.6344
Train Acc: 0.5538 | Val Acc: 0.6250
Train F1: 0.4921 | Val F1: 0.5714
Val Precision: 0.6000 | Val Recall: 0.5455

✓ Saved best model (Val Loss: 0.6344, F1: 0.5714)
```

### Inference

**Generate Predictions:**

```bash
python classify.py \
    --test-csv test.csv \
    --model models/bdh_classifier.pt \
    --output outputs/submission.csv
```

**Custom Batch Size:**

```bash
python classify.py \
    --test-csv test.csv \
    --model models/bdh_classifier.pt \
    --output outputs/submission.csv \
    --batch-size 64 \
    --device cuda
```

**Inference Output:**

```
Running batch inference on test.csv
Loading model from models/bdh_classifier.pt on cuda
Loaded model from models/bdh_classifier.pt

Processed 32/61 samples
Processed 61/61 samples

Predictions saved to outputs/submission.csv

Prediction distribution:
consistent     34
contradict     27
Name: prediction, dtype: int64

Mean confidence: 0.7823
```

### Evaluation

**Evaluate on Validation Set:**

```bash
python evaluate.py \
    --model models/bdh_classifier.pt \
    --train-csv train.csv \
    --device cuda
```

**Evaluation Output:**

```
============================================================
EVALUATION REPORT
============================================================

Accuracy:  0.8125
Precision: 0.7500
Recall:    0.8571
F1 Score:  0.8000

Confusion Matrix:
[[9 1]
 [2 4]]

True Negatives:  9
False Positives: 1
False Negatives: 2
True Positives:  4

Classification Report:
              precision    recall  f1-score   support

  Consistent       0.82      0.90      0.86        10
  Contradict       0.80      0.67      0.73         6

    accuracy                           0.81        16
   macro avg       0.81      0.78      0.79        16
weighted avg       0.81      0.81      0.81        16

============================================================

Confusion matrix saved to outputs/confusion_matrix.png
```

---

## Pathway Modes

### When to Use Each Mode

| Feature          | Batch Mode                 | Pathway Streaming             |
| ---------------- | -------------------------- | ----------------------------- |
| **Use Case**     | Static datasets (test.csv) | Real-time data streams        |
| **Complexity**   | Simple                     | More complex                  |
| **Performance**  | Faster for small datasets  | Better for continuous streams |
| **Dependencies** | PyTorch only               | PyTorch + Pathway             |
| **Best For**     | Hackathon submission       | Production deployment         |

### Batch Mode (Recommended for Hackathon)

```bash
python pathway_pipeline.py \
    --input test.csv \
    --output outputs/submission.csv \
    --model models/bdh_classifier.pt \
    --mode batch \
    --batch-size 32
```

**Advantages:**

- Simple, no Pathway server required
- Faster for static datasets
- Fewer dependencies

### Pathway Streaming Mode

```bash
python pathway_pipeline.py \
    --input test.csv \
    --output outputs/submission.csv \
    --model models/bdh_classifier.pt \
    --mode pathway
```

**Advantages:**

- Real-time processing
- Scalable to large streams
- Incremental state updates

**Requirements:**

- Pathway 0.8.0+
- May require separate Pathway server process

---

## Complete Workflow

### Scenario 1: Quick Testing (No GPU)

```bash
# 1. Train on CPU (small model)
python train.py \
    --train-csv train.csv \
    --batch-size 4 \
    --epochs 20 \
    --device cpu \
    --output models/bdh_cpu.pt

# 2. Inference
python classify.py \
    --test-csv test.csv \
    --model models/bdh_cpu.pt \
    --device cpu \
    --batch-size 8 \
    --output outputs/submission.csv
```

### Scenario 2: Best Performance (GPU)

```bash
# 1. Train with optimal settings
python train.py \
    --train-csv train.csv \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.001 \
    --device cuda \
    --output models/bdh_best.pt

# 2. Evaluate
python evaluate.py \
    --model models/bdh_best.pt \
    --train-csv train.csv \
    --device cuda

# 3. Inference with high batch size
python classify.py \
    --test-csv test.csv \
    --model models/bdh_best.pt \
    --device cuda \
    --batch-size 64 \
    --output outputs/submission.csv
```

### Scenario 3: Docker Production Pipeline

```bash
# Build once
docker-compose build

# Train
docker-compose --profile train up train

# Inference (multiple runs with different models)
docker-compose --profile inference up inference

# View results
docker run --rm -v $(pwd)/outputs:/data alpine cat /data/submission.csv
```

### Scenario 4: Ensemble for Robustness

```bash
# Train 3 models with different seeds
for seed in 1 2 3; do
    python train.py \
        --train-csv train.csv \
        --output models/bdh_seed${seed}.pt \
        --seed ${seed}
done

# Average predictions (requires custom ensemble.py)
python ensemble.py \
    --models models/bdh_seed*.pt \
    --test-csv test.csv \
    --output outputs/ensemble_submission.csv
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Symptoms:**

```
RuntimeError: CUDA out of memory. Tried to allocate 256.00 MiB
```

**Solutions:**

```bash
# 1. Reduce batch size
python train.py --batch-size 4

# 2. Use CPU
python train.py --device cpu

# 3. Enable gradient accumulation (modify train.py)
# accumulation_steps = 4
```

### Issue: Pathway Import Error

**Symptoms:**

```
ModuleNotFoundError: No module named 'pathway'
```

**Solutions:**

```bash
# 1. Install Pathway
pip install pathway>=0.8.0

# 2. Use batch mode instead
python pathway_pipeline.py --mode batch
```

### Issue: Docker GPU Not Detected

**Symptoms:**

```
docker: Error response from daemon: could not select device driver
```

**Solutions:**

```bash
# 1. Install nvidia-docker
# Windows: https://docs.nvidia.com/cuda/wsl-user-guide/
# Linux: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 2. Use CPU profile
docker-compose --profile inference-cpu up inference-cpu
```

### Issue: Low Accuracy (< 60%)

**Potential Causes & Solutions:**

1. **Insufficient Training:**

   ```bash
   # Increase epochs
   python train.py --epochs 100
   ```

2. **Poor Generalization:**

   ```bash
   # Add dropout or reduce model size
   # Edit config.py: dropout = 0.2
   ```

3. **Data Imbalance:**

   ```bash
   # Check class distribution
   python -c "import pandas as pd; print(pd.read_csv('train.csv')['label'].value_counts())"
   ```

4. **Need Pre-training:**
   ```bash
   # Use pre-trained weights (if available)
   python train.py --pretrained models/bdh_pretrained.pt
   ```

### Issue: Predictions All Same Class

**Symptoms:**

```
Prediction distribution:
consistent    61
Name: prediction, dtype: int64
```

**Solutions:**

```bash
# 1. Check model confidence
python classify.py --test-csv test.csv --model models/bdh_classifier.pt

# 2. Retrain with class weights
# Modify train.py to add class_weight parameter

# 3. Adjust decision threshold
# Modify model.py predict() to use custom threshold
```

### Issue: Docker Build Fails

**Symptoms:**

```
ERROR [internal] load metadata for docker.io/library/python:3.11-slim
```

**Solutions:**

```bash
# 1. Check Docker daemon is running
docker info

# 2. Pull base image manually
docker pull python:3.11-slim

# 3. Use cached build
docker-compose build --no-cache
```

---

## Performance Optimization Tips

### 1. Data Loading

```python
# In data_utils.py, increase num_workers (if not on Windows)
DataLoader(..., num_workers=4)
```

### 2. Mixed Precision Training

```python
# Already enabled via torch.cuda.amp in train.py
# Provides ~2x speedup on modern GPUs
```

### 3. Model Compilation

```python
# In train.py, add:
model = torch.compile(model)  # PyTorch 2.0+
```

### 4. Batch Size Tuning

```bash
# Find optimal batch size
for bs in 8 16 32 64; do
    python train.py --batch-size $bs --epochs 5
done
```

### 5. Early Stopping

```bash
# Reduce patience for faster experimentation
python train.py --patience 5
```

---

## Next Steps

After successful training and inference:

1. **Analyze Results:**

   ```bash
   python evaluate.py --model models/bdh_classifier.pt
   cat outputs/error_analysis.csv
   ```

2. **Improve Model:**

   - Try different pooling strategies in `model.py`
   - Adjust BDH architecture in `config.py`
   - Add data augmentation

3. **Submit to Hackathon:**

   ```bash
   # Final submission file
   head outputs/submission.csv
   ```

4. **Deploy (Optional):**

   ```bash
   # Build production Docker image
   docker build -t bdh-classifier:prod .

   # Push to registry
   docker tag bdh-classifier:prod your-registry/bdh-classifier
   docker push your-registry/bdh-classifier
   ```

---

## Useful Commands Cheat Sheet

```bash
# Build everything
docker-compose build

# Train
docker-compose --profile train up

# Inference (batch)
docker-compose --profile inference up

# Inference (Pathway)
docker-compose --profile inference-pathway up

# CPU inference
docker-compose --profile inference-cpu up

# Clean up
docker-compose down
docker system prune -a

# View logs
docker-compose logs train
docker-compose logs inference

# Interactive debugging
docker run -it --rm bdh-classifier:latest /bin/bash
```

---

For more details, see [README.md](README.md)
