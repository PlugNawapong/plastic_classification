# PCA Training - Quick Start Guide

## 🎯 Goal
Train a pixel-wise 1D CNN classifier with PCA dimensionality reduction: **459 bands → 150 components → 2-3x faster, same accuracy**

## ⚡ Fastest Path (Google Colab - Recommended)

### 5-Minute Setup:

```bash
# 1. See instructions
./upload_to_colab.sh
```

Then follow on-screen steps:
1. Upload `training_dataset/` and `Ground_Truth/` to Google Drive
2. Upload `train_pca_colab.ipynb` to Colab
3. Enable GPU runtime
4. Run all cells
5. Wait 2-3 hours
6. Download results

**Detailed guide:** [COLAB_SETUP_GUIDE.md](COLAB_SETUP_GUIDE.md)

## 🔬 Alternative: Local Training

### If you have GPU:

```bash
# Analyze data (2 min)
python analyze_pca_benefits.py

# Train with comparison (2-4 hours)
python train_with_pca.py --compare --compare-configs 100 150 200
```

**Detailed guide:** [README_PCA_TRAINING.md](README_PCA_TRAINING.md)

## 📚 Documentation Files

| File | When to Use |
|------|-------------|
| **[PCA_TRAINING_SUMMARY.md](PCA_TRAINING_SUMMARY.md)** | 📖 Start here - Complete overview |
| **[COLAB_SETUP_GUIDE.md](COLAB_SETUP_GUIDE.md)** | ☁️ Google Colab training (recommended) |
| **[README_PCA_TRAINING.md](README_PCA_TRAINING.md)** | 💻 Local training + general guide |
| **[PCA_GUIDE.md](PCA_GUIDE.md)** | 📊 Understand PCA in detail |
| **[PCA_SUMMARY.md](PCA_SUMMARY.md)** | ⚡ Quick PCA reference |

## 🛠️ Implementation Files

| File | Purpose |
|------|---------|
| `pca_band_selection.py` | PCA implementation |
| `train_with_pca.py` | Local training script |
| `train_pca_colab.ipynb` | Colab notebook (GPU) |
| `analyze_pca_benefits.py` | Data analysis tool |
| `upload_to_colab.sh` | Setup helper |

## 📋 Checklist

### Before Starting:
- [ ] Dataset normalized (already done ✓)
- [ ] Labels file ready (Ground_Truth/labels.json)
- [ ] Choose: Colab or Local?

### For Colab:
- [ ] Run `./upload_to_colab.sh` for instructions
- [ ] Upload dataset to Google Drive
- [ ] Upload notebook to Colab
- [ ] Enable GPU runtime
- [ ] Run all cells

### After Training:
- [ ] Download best model + PCA reducer
- [ ] Note best configuration (e.g., PCA-150)
- [ ] Ready for inference

## 🎯 Expected Results

```
BEST CONFIGURATION: PCA-150
  Validation Accuracy: 95.4%
  Number of Bands: 150 (67% reduction)
  Training Speed: 2x faster
  Inference Speed: 2x faster
```

## 🚀 Next Steps

**Start training:**
```bash
./upload_to_colab.sh  # See Colab instructions
```

**Ready? Run this command to begin!**
