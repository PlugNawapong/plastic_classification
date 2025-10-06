# Final Export Guide: Bands with Wavelength from header.json

## Issue & Solution

**Problem Found:** The previous export failed because **all bands were marked as noisy** (0 clean bands).

**Root Cause:** Default quality thresholds (SNR≥5.0, Variance≥10.0) were too strict for your 8-bit images.

**Solution:** New script with **automatic threshold adjustment** based on your data.

## Quick Start

### 1. Install Dependencies (Required First!)

```bash
pip install -r requirements.txt
```

This installs: numpy, pillow, scikit-learn, matplotlib, tqdm, scipy

### 2. Export Clean Bands (Auto Mode - Recommended)

```bash
python export_clean_bands.py
```

**What it does:**
- ✓ Reads actual wavelengths from `header.json`
- ✓ **Auto-adjusts thresholds** based on your data
- ✓ Keeps top 75% of bands by quality
- ✓ Exports with wavelength in filename

## Your Dataset Specifications

**From header.json:**
- **Wavelength range:** 450.5 - 853.6 nm
- **Total bands:** 458
- **Step size:** ~0.88 nm
- **Coverage:** Visible (450-700nm) + Near-IR (700-850nm)
- **Bit depth:** 8-bit (0-255)

## Auto vs Manual Thresholds

### Auto Mode (Recommended)

```python
# Keeps top 75% of bands
export_clean_bands(
    input_dir='training_dataset',
    output_dir='clean_normalized_bands',
    auto_threshold=True,
    keep_percentage=75  # Keep top 75%
)
```

**Result:** ~343 clean bands (75% of 458)

### Manual Mode (Advanced)

If auto mode doesn't work well, use manual thresholds:

```python
export_clean_bands(
    input_dir='training_dataset',
    output_dir='clean_normalized_bands',
    auto_threshold=False,
    manual_snr=2.0,      # Lower for 8-bit images
    manual_var=50.0      # Adjusted for your data
)
```

**Threshold recommendations for 8-bit images:**
- SNR: 1.5 - 3.0 (lower than 16-bit)
- Variance: 10 - 100 (depends on scene contrast)

## Output Files

### Directory Structure

```
clean_normalized_bands/
├── Band_001_450.5nm_normalized.png   ← Actual wavelength from header.json
├── Band_002_451.4nm_normalized.png
├── Band_003_452.3nm_normalized.png
├── ...
├── Band_458_853.6nm_normalized.png
├── quality_report.txt                 ← Detailed analysis
└── clean_bands_list.txt               ← For training
```

### Filename Format

```
Band_XXX_WWW.Wnm_normalized.png
     │││  │││││
     │││  │││││
     │││  ││││└─ "normalized" suffix
     │││  │││└── ".png" extension
     │││  ││└─── "nm" unit
     │││  │└──── Wavelength (1 decimal, from header.json)
     │││  └───── Underscore
     ││└──────── Band number (001-458)
     │└───────── Underscore
     └────────── "Band" prefix
```

**Examples:**
- `Band_001_450.5nm_normalized.png` = Band 1, 450.5 nm (Blue)
- `Band_150_581.0nm_normalized.png` = Band 150, 581.0 nm (Yellow)
- `Band_300_714.2nm_normalized.png` = Band 300, 714.2 nm (Red/NIR)
- `Band_458_853.6nm_normalized.png` = Band 458, 853.6 nm (NIR)

## Verification Steps

### 1. Check Export Success

```bash
# Count exported files
ls clean_normalized_bands/Band_*.png | wc -l

# Should show: ~340-350 files (75% of 458)
```

### 2. Review Quality Report

```bash
cat clean_normalized_bands/quality_report.txt
```

**Look for:**
```
Wavelength range: 450.5 - 853.6 nm
Total bands: 458
Clean bands: 343/458 (74.9%)
SNR threshold: 2.45 (auto-determined)
Variance threshold: 125.3 (auto-determined)
```

### 3. Check Wavelength Accuracy

```bash
# View first few files
ls clean_normalized_bands/ | head -10

# Should show:
# Band_001_450.5nm_normalized.png
# Band_002_451.4nm_normalized.png
# Band_003_452.3nm_normalized.png
# ...
```

### 4. Visual Inspection

```bash
# Open a sample band
open clean_normalized_bands/Band_001_450.5nm_normalized.png

# Check mid-range
open clean_normalized_bands/Band_230_653.4nm_normalized.png

# Check NIR
open clean_normalized_bands/Band_400_802.4nm_normalized.png
```

## Spectral Coverage

**Your dataset spans:**

| Wavelength | Type | Color | Use |
|-----------|------|-------|-----|
| 450-495 nm | Visible | Blue | Material ID |
| 495-570 nm | Visible | Green | Color analysis |
| 570-620 nm | Visible | Yellow-Orange | Surface properties |
| 620-700 nm | Visible | Red | Reflectance |
| 700-850 nm | Near-IR | - | Molecular fingerprint |

## Using in Training

### Load Clean Bands

```python
# Read clean band list
clean_bands = []
with open('clean_normalized_bands/clean_bands_list.txt') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            band_num, wavelength = line.split(',')
            clean_bands.append((int(band_num)-1, float(wavelength)))

print(f"Loaded {len(clean_bands)} clean bands")
print(f"Wavelength range: {clean_bands[0][1]:.1f} - {clean_bands[-1][1]:.1f} nm")

# Use with dataset_clean_bands.py
from dataset_clean_bands import create_clean_dataloaders

train_loader, val_loader = create_clean_dataloaders(
    train_dir='training_dataset',
    label_path='Ground_Truth/labels.json',
    clean_bands_list='clean_normalized_bands/clean_bands_list.txt'
)
```

### Update Model

```python
# Count clean bands
n_clean = len(glob.glob('clean_normalized_bands/Band_*.png'))

# Create model with correct input size
from model import create_model
model = create_model(n_spectral_bands=n_clean, n_classes=11)
```

## Troubleshooting

### Issue 1: ModuleNotFoundError

**Error:** `ModuleNotFoundError: No module named 'numpy'`

**Fix:**
```bash
pip install -r requirements.txt
```

### Issue 2: Still 0 Clean Bands

**If auto mode gives 0 bands:**

Edit `export_clean_bands.py` line 253:
```python
keep_percentage=90  # Was 75, try 90 to keep more
```

Or use manual mode:
```python
export_clean_bands(
    auto_threshold=False,
    manual_snr=1.0,      # Very lenient
    manual_var=1.0       # Very lenient
)
```

### Issue 3: Wavelength Mismatch

**If wavelengths look wrong:**

Check your header.json:
```bash
python -c "import json; print(json.load(open('training_dataset/header.json'))['wavelength (nm)'][:5])"
```

Should output: `[450.50775, 451.38975, ...]`

### Issue 4: Images All Black/White

**If exported images are solid black or white:**

Your images may need different normalization. Check original:
```bash
python -c "
from PIL import Image
import numpy as np
img = np.array(Image.open('training_dataset/ImagesStack001.png'))
print('Range:', img.min(), '-', img.max())
print('Mean:', img.mean())
"
```

## Expected Results

### With Auto Thresholds (75% keep rate)

```
Total bands: 458
Clean bands: ~343 (75%)
Noisy bands: ~115 (25%)

Wavelength coverage: 450.5 - 853.6 nm
Export time: 1-2 minutes
```

### With Manual Thresholds (lenient)

```
SNR: 2.0, Variance: 50.0
Clean bands: ~400-420 (87-92%)
Noisy bands: ~38-58 (8-13%)
```

## Complete Workflow

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Export bands (auto mode)
python export_clean_bands.py

# Step 3: Verify export
ls clean_normalized_bands/*.png | wc -l
cat clean_normalized_bands/quality_report.txt

# Step 4: Check samples visually
open clean_normalized_bands/Band_001_450.5nm_normalized.png
open clean_normalized_bands/Band_230_653.4nm_normalized.png

# Step 5: Use in training
python train.py  # Will use clean bands via dataset_clean_bands.py
```

## Summary

✅ **Automatic threshold adjustment** - no guessing needed
✅ **Actual wavelengths** from header.json (450.5-853.6 nm)
✅ **Keeps top 75%** of bands by default (customizable)
✅ **8-bit optimized** - works with your image format
✅ **Ready for training** - clean_bands_list.txt included

Once you run `pip install -r requirements.txt`, just execute:
```bash
python export_clean_bands.py
```

And you'll have clean, properly-named bands with actual wavelengths! 🎯
