# BDH Narrative Consistency Classifier

Track-B solution for Kharagpur Data Science Hackathon using Baby Dragon Hatchling (BDH) for long-context narrative reasoning.

## 🎯 Project Overview

Given:

- Full novel context (100k+ words)
- Character backstory snippet

**Task**: Classify whether the backstory is **consistent** or **contradictory** to the established narrative.

**Approach**:

- Use BDH's Hebbian working memory for stateful reasoning
- Pathway streaming pipeline for scalable inference
- Docker containerization for reproducibility

## 🏗️ Architecture

```
Input Text → Byte-Level Tokenization → BDH Encoder (6 layers)
    → Sparse Hebbian Activations → Mean Pooling
    → Classification Head → Binary Prediction
```

**Key Features**:

- ✅ Biologically-inspired attention mechanism
- ✅ Emergent attention from neuron dynamics
- ✅ Sparse, interpretable activations (ReLU)
- ✅ Hebbian synaptic plasticity for belief updates
- ✅ Pathway integration for streaming inference

## 📦 Quick Start (Docker)

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU + nvidia-docker (optional, can run on CPU)

### Run Full Pipeline

**Windows (PowerShell):**

```powershell
.\run.ps1
```

**Linux/Mac:**

```bash
chmod +x run.sh
./run.sh
```

This will:

1. Build Docker image
2. Train BDH classifier on `train.csv`
3. Generate predictions on `test.csv`
4. Save results to `outputs/submission.csv`

### Manual Docker Commands

**1. Build Image:**

```bash
docker-compose build
```

**2. Train Model:**

```bash
docker-compose --profile train up train
```

**3. Run Inference (Batch Mode):**

```bash
docker-compose --profile inference up inference
```

**4. Run Inference (Pathway Streaming):**

```bash
docker-compose --profile inference-pathway up inference-pathway
```

**5. CPU-Only Inference:**

```bash
docker-compose --profile inference-cpu up inference-cpu
```

## 🔧 Local Development (Without Docker)

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py \
    --train-csv train.csv \
    --batch-size 16 \
    --epochs 50 \
    --lr 0.001 \
    --device cuda \
    --output models/bdh_classifier.pt
```

### Inference

```bash
python classify.py \
    --test-csv test.csv \
    --model models/bdh_classifier.pt \
    --output outputs/submission.csv \
    --device cuda \
    --batch-size 32
```

### Evaluation

```bash
python evaluate.py \
    --model models/bdh_classifier.pt \
    --train-csv train.csv \
    --device cuda \
    --output-dir outputs
```

## 📊 Model Configuration

**BDH Architecture** ([config.py](config.py)):

- Layers: 6
- Embedding dim: 256
- Attention heads: 4
- Sparse multiplier: 128 (32,768 sparse neurons)
- Vocabulary: 256 (byte-level)
- Max sequence length: 512 tokens

**Training** ([train.py](train.py)):

- Optimizer: AdamW (lr=1e-3, weight_decay=0.1)
- Batch size: 16
- Early stopping: patience=10
- Mixed precision: AMP enabled
- Gradient clipping: 1.0

## 🗂️ Project Structure

```
BDH_Pathway/
├── bdh.py                  # BDH model implementation
├── model.py                # Classification wrapper
├── data_utils.py           # Data loading & preprocessing
├── train.py                # Training script
├── classify.py             # Inference script
├── evaluate.py             # Evaluation utilities
├── pathway_pipeline.py     # Pathway streaming pipeline
├── config.py               # Configuration
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose services
├── requirements.txt        # Python dependencies
├── train.csv               # Training data (81 samples)
├── test.csv                # Test data (61 samples)
├── models/                 # Saved checkpoints
├── outputs/                # Predictions & analysis
└── README.md               # This file
```

## 🔬 How BDH Works

### Hebbian Working Memory

```python
# In each BDH layer:
x_sparse = ReLU(x @ encoder)           # Activate sparse neurons
yKV = Attention(Q=x_sparse, K=x_sparse, V=x)  # Attend in sparse space
y_sparse = ReLU(yKV @ encoder_v)       # Value-based activations
xy_sparse = x_sparse * y_sparse        # Hebbian multiplication (belief update)
output = xy_sparse @ decoder           # Decode to dense
```

**Key Insight**: The `xy_sparse` term represents synaptic strengthening—when query and value neurons fire together, their multiplicative product amplifies, encoding relationships between narrative concepts.

### Stateful Reasoning

- **Input**: Character biography text
- **Process**: Sequential chunks activate concept neurons → contradictions trigger competing pathways → Hebbian gating maintains hypothesis states
- **Output**: Accumulated synaptic strengths → classification decision

## 🌊 Pathway Integration

Two modes available:

**1. Pathway Streaming** (for real-time processing):

```python
python pathway_pipeline.py \
    --input test.csv \
    --output outputs/submission.csv \
    --model models/bdh_classifier.pt \
    --mode pathway
```

**2. Batch Mode** (simpler, faster for static datasets):

```python
python pathway_pipeline.py \
    --input test.csv \
    --output outputs/submission.csv \
    --model models/bdh_classifier.pt \
    --mode batch
```

## 📈 Performance Metrics

After training, evaluate on validation set:

```bash
python evaluate.py \
    --model models/bdh_classifier.pt \
    --train-csv train.csv
```

Outputs:

- Accuracy, Precision, Recall, F1
- Confusion matrix visualization
- Error analysis CSV

## 🎓 Dataset Details

**Training Set**: 81 samples

- Books: "In Search of the Castaways", "The Count of Monte Cristo"
- Characters: Jacques Paganel, Thalcave, Kai-Koumou, Tom Ayrton, Noirtier, Faria
- Labels: `consistent` (63%) / `contradict` (37%)

**Test Set**: 61 samples (no labels)

**Text Length**: 30-200 words per biography

## 🚀 Advanced Usage

### Pre-training on Custom Corpus

To improve performance, pre-train BDH on literary data:

```python
# Download Project Gutenberg novels
# Train BDH as language model
python pretrain.py --corpus data/gutenberg/ --output models/bdh_pretrained.pt

# Fine-tune on classification task
python train.py --pretrained models/bdh_pretrained.pt --train-csv train.csv
```

### Ensemble Methods

Combine multiple models for robustness:

```python
# Train 5 models with different seeds
for seed in {1..5}; do
    python train.py --seed $seed --output models/bdh_$seed.pt
done

# Ensemble predictions
python ensemble.py --models models/bdh_*.pt --test-csv test.csv
```

## 🐛 Troubleshooting

### Out of Memory (GPU)

- Reduce batch size: `--batch-size 8`
- Use CPU: `--device cpu`
- Enable gradient accumulation in `train.py`

### Pathway Import Error

- Install: `pip install pathway>=0.8.0`
- Use batch mode: `--mode batch` (no Pathway required)

### Docker GPU Not Found

- Install nvidia-docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- Use CPU profile: `docker-compose --profile inference-cpu up`

## 📚 References

- **BDH Paper**: Kosowski et al. (2025). "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain." [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)
- **Official Repo**: https://github.com/pathwaycom/bdh
- **Pathway Docs**: https://pathway.com/developers/

## 📝 License

MIT License - see repository for details.

## 👥 Contributors

Kharagpur Data Science Hackathon - Track B Team

---

**Built with**: PyTorch, Pathway, Docker | **Powered by**: Baby Dragon Hatchling 🐉
