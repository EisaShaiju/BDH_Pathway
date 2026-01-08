# 🚀 QUICKSTART - BDH Narrative Classifier

## ⚡ 3-Step Solution

### Step 1: Build (1 minute)

```powershell
docker-compose build
```

### Step 2: Train (5-10 minutes)

```powershell
docker-compose --profile train up train
```

### Step 3: Generate Predictions (30 seconds)

```powershell
docker-compose --profile inference up inference
```

**Done!** Results in `outputs/submission.csv`

---

## 🎯 Without Docker

```powershell
# Install
pip install -r requirements.txt

# Train
python train.py --train-csv train.csv --output models/bdh_classifier.pt

# Predict
python classify.py --test-csv test.csv --model models/bdh_classifier.pt --output outputs/submission.csv
```

---

## 🛠️ Shortcuts (PowerShell)

```powershell
# Full pipeline
.\run.ps1

# Individual commands
.\make.ps1 train          # Train model
.\make.ps1 infer          # Run inference
.\make.ps1 eval           # Evaluate
.\make.ps1 quick          # Fast CPU test
```

---

## 📊 Expected Output

**Submission file format:**

```csv
id,prediction,confidence
95,contradict,0.8234
136,consistent,0.7891
59,consistent,0.9102
...
```

---

## 🐛 Troubleshooting

**GPU not found?**

```powershell
docker-compose --profile inference-cpu up inference-cpu
```

**Out of memory?**

```powershell
python train.py --batch-size 4 --device cpu
```

**Need help?**

```powershell
.\make.ps1 help
```

---

## 📚 Full Documentation

- [README.md](README.md) - Complete project overview
- [USAGE.md](USAGE.md) - Detailed usage guide
- [config.py](config.py) - Model configuration

---

## 🎓 What This Does

1. **Loads** character biographies from train.csv (81 samples)
2. **Trains** BDH neural network with Hebbian memory
3. **Classifies** test.csv biographies (61 samples)
4. **Outputs** consistent/contradict predictions

**Tech Stack:** BDH (biologically-inspired Transformer) + Pathway + Docker

---

## ✅ Verification

Check your results:

```powershell
# View predictions
cat outputs/submission.csv | Select -First 10

# Count predictions
python -c "import pandas as pd; print(pd.read_csv('outputs/submission.csv')['prediction'].value_counts())"
```

Expected distribution: ~30-35 consistent, ~25-30 contradict

---

**Ready to go!** Run `.\run.ps1` and you're done. 🎉
