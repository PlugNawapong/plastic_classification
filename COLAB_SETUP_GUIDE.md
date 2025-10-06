# Google Colab Training Setup Guide

## Quick Start (5 Steps)

### 1. Prepare Your Dataset

**Upload to Google Drive:**

```
MyDrive/
└── plastic_classification/
    ├── training_dataset/
    │   ├── header.json
    │   ├── ImagesStack001.png
    │   ├── ImagesStack002.png
    │   └── ... (all 459 bands)
    └── Ground_Truth/
        └── labels.json
```

**How to upload:**
1. Open Google Drive in browser
2. Create folder `plastic_classification`
3. Upload your `training_dataset` folder
4. Upload your `Ground_Truth` folder

### 2. Open Colab Notebook

**Option A: Upload notebook**
1. Go to https://colab.research.google.com
2. Click `File → Upload notebook`
3. Upload `train_pca_colab.ipynb`

**Option B: Upload to Drive first**
1. Upload `train_pca_colab.ipynb` to your Google Drive
2. Right-click → Open with → Google Colaboratory

### 3. Select GPU Runtime

**IMPORTANT:** Enable GPU for faster training

1. Click `Runtime → Change runtime type`
2. Select `GPU` as Hardware accelerator
3. For Colab Pro+: Choose T4, V100, or A100
4. Click `Save`

### 4. Update Paths in Notebook

In the notebook, find this cell:

```python
# ==============================================================================
# CONFIGURE THESE PATHS
# ==============================================================================

DRIVE_BASE = '/content/drive/MyDrive/plastic_classification'
```

Update `DRIVE_BASE` to match your Google Drive folder structure.

### 5. Run All Cells

1. Click `Runtime → Run all`
2. Authorize Google Drive access when prompted
3. Wait for training to complete (~30-60 minutes)

## Training Configuration

### Default Settings (optimized for Colab Pro+):

```python
CONFIG = {
    'n_classes': 11,
    'batch_size': 1024,      # Large batch for GPU
    'n_epochs': 50,
    'learning_rate': 0.001,
    'dropout_rate': 0.5,
    'train_split': 0.9,
    'pca_configs': [None, 50, 100, 150, 200],
}
```

### What It Does:

1. **Without PCA** (baseline): Uses all 459 bands
2. **PCA-50**: Reduces to 50 components
3. **PCA-100**: Reduces to 100 components
4. **PCA-150**: Reduces to 150 components
5. **PCA-200**: Reduces to 200 components

### Expected Training Time (Colab Pro+ with A100):

| Configuration | Time per Epoch | Total Time (50 epochs) |
|---------------|----------------|------------------------|
| No PCA (459)  | ~60 sec        | ~50 min                |
| PCA-200       | ~40 sec        | ~35 min                |
| PCA-150       | ~30 sec        | ~25 min                |
| PCA-100       | ~25 sec        | ~20 min                |
| PCA-50        | ~20 sec        | ~17 min                |

**Total for all 5 configs: ~2.5 hours**

## Customize Training

### Change PCA Configurations

Test different component counts:

```python
CONFIG = {
    ...
    'pca_configs': [None, 100, 150],  # Test only these
}
```

### Reduce Epochs (for quick testing)

```python
CONFIG = {
    ...
    'n_epochs': 20,  # Faster training
}
```

### Use Subset of Data (for testing)

```python
CONFIG = {
    ...
    'max_samples': 50000,  # Use only 50k pixels
}
```

### Adjust Batch Size

**For T4 GPU (16GB):**
```python
'batch_size': 512
```

**For V100/A100 (40GB+):**
```python
'batch_size': 2048
```

## Monitor Training

### In Colab:

- Watch progress bars for each epoch
- Check printed validation accuracy
- GPU usage: `Runtime → View resources`

### Key Metrics to Watch:

```
Epoch 1/50: Train Loss=1.2345, Train Acc=65.43% | Val Loss=1.1234, Val Acc=68.21%
✓ Best model saved (Val Acc: 68.21%)
```

**Good signs:**
- ✓ Validation accuracy increasing
- ✓ Train/val gap < 5%
- ✓ Loss decreasing steadily

**Warning signs:**
- ⚠️ Validation accuracy plateauing early
- ⚠️ Train/val gap > 10% (overfitting)
- ⚠️ Loss increasing or unstable

## Download Results

### Automatic Download

At the end of notebook:

```python
from google.colab import files
files.download('/content/colab_results.zip')
```

### Manual Download

Results are saved in Google Drive:

```
MyDrive/plastic_classification/colab_results/
├── no_pca/
│   ├── best_model.pth
│   └── history.json
├── pca_50/
│   ├── best_model.pth
│   ├── pca_model.pkl
│   └── history.json
├── pca_100/
│   ├── best_model.pth
│   ├── pca_model.pkl
│   └── history.json
├── pca_150/
│   ├── best_model.pth
│   ├── pca_model.pkl
│   └── history.json
├── pca_200/
│   ├── best_model.pth
│   ├── pca_model.pkl
│   └── history.json
├── comparison_results.json
└── comparison_plot.png
```

### What to Download:

**For best configuration (e.g., PCA-150):**
- `best_model.pth` - Trained model weights
- `pca_model.pkl` - PCA reducer (if used)
- `history.json` - Training history

**For analysis:**
- `comparison_results.json` - All results
- `comparison_plot.png` - Visualization

## Use Results Locally

### 1. Load Best Model

```python
import torch
from pca_band_selection import PCABandSelector

# Load PCA reducer
pca_reducer = PCABandSelector.load('pca_150/pca_model.pkl')

# Load model
from model import SpectralCNN1D
model = SpectralCNN1D(n_bands=150, n_classes=11)
checkpoint = torch.load('pca_150/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded: {checkpoint['val_acc']:.2f}% validation accuracy")
```

### 2. Use for Inference

```python
# Load test spectrum
test_spectrum = load_spectrum(x, y)  # Shape: (459,)

# Apply PCA
test_spectrum_pca = pca_reducer.transform(test_spectrum)  # Shape: (150,)

# Predict
with torch.no_grad():
    output = model(torch.from_numpy(test_spectrum_pca).float().unsqueeze(0))
    prediction = output.argmax(1).item()

print(f"Predicted class: {prediction}")
```

## Troubleshooting

### Problem: Out of Memory

**Solution:**
```python
CONFIG['batch_size'] = 256  # Reduce batch size
```

### Problem: Colab Disconnects

**Solutions:**
1. Keep browser tab active
2. Use Colab Pro+ for longer runtime
3. Save checkpoints frequently (already implemented)
4. Resume training from checkpoint if disconnected

### Problem: Training Too Slow

**Solutions:**
1. Check GPU is enabled: `Runtime → Change runtime type`
2. Reduce number of PCA configs to test
3. Reduce epochs for quick test
4. Use smaller dataset (`max_samples=50000`)

### Problem: Can't Mount Drive

**Solution:**
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Problem: Path Not Found

**Solution:**
Check your folder structure:
```python
!ls /content/drive/MyDrive/plastic_classification/
!ls /content/drive/MyDrive/plastic_classification/training_dataset/
```

## Performance Tips

### 1. Use Mixed Precision Training (for A100)

Add to training code:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(spectra)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Increase Num Workers (if RAM allows)

```python
'num_workers': 4  # Faster data loading
```

### 3. Pin Memory

Already enabled:
```python
DataLoader(..., pin_memory=True)
```

### 4. Benchmark Mode

Add at start:
```python
torch.backends.cudnn.benchmark = True
```

## Expected Results

### Typical Accuracy (based on 459-band hyperspectral):

| Configuration | Val Accuracy | Speed vs Baseline |
|---------------|--------------|-------------------|
| No PCA        | 94-95%       | 1.0x (baseline)   |
| PCA-200       | 94-95%       | 1.5x faster       |
| PCA-150       | 94-96%       | 2.0x faster       |
| PCA-100       | 93-95%       | 2.5x faster       |
| PCA-50        | 91-93%       | 3.0x faster       |

**Best choice:** Usually PCA-150 (best accuracy/speed tradeoff)

## Next Steps After Training

1. **Download best model** and PCA reducer
2. **Apply to test images** for classification
3. **Visualize predictions** with your existing visualization tools
4. **Apply post-processing** (median filter, morphological ops)
5. **Calculate metrics** (confusion matrix, per-class accuracy)

## FAQ

**Q: Do I need Colab Pro+ or is free tier OK?**
- Free tier works but slower (12-hour limit, slower GPUs)
- Pro+ recommended for full comparison (all PCA configs)

**Q: Can I stop and resume training?**
- Yes, checkpoints are saved
- Load with: `model.load_state_dict(torch.load('checkpoint.pth'))`

**Q: How much RAM is needed?**
- Standard RAM (12GB) is sufficient
- High-RAM not required for this task

**Q: Should I use TPU instead of GPU?**
- No, GPU is better for this task (1D convolutions)

**Q: Can I use my own model architecture?**
- Yes! Just replace the `SpectralCNN1D` class
- Keep input/output dimensions compatible

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify paths and GPU settings
3. Try with smaller dataset first (`max_samples=10000`)
4. Check Colab runtime logs for errors
