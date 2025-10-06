# Export Bands with Wavelength from header.json

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Export with actual wavelengths from header.json
python export_bands_with_header.py
```

## What This Does

✓ Reads **actual wavelength data** from `header.json`
✓ Exports clean bands with **real wavelength in filename**
✓ Automatic noise filtering (SNR and variance analysis)
✓ Band-wise normalization for optimal training

## Wavelength Information

**From your header.json:**
- **Range:** 450.5 - 853.6 nm
- **Bands:** 458 spectral bands
- **Step size:** ~0.88 nm
- **Spectrum:** Visible (450-700nm) to Near-IR (700-850nm)

## Output Files

**Directory:** `clean_normalized_bands/`

**Exported files with actual wavelengths:**
```
Band_001_450.5nm_normalized.png    (450.5 nm - Blue)
Band_002_451.4nm_normalized.png    (451.4 nm)
Band_003_452.3nm_normalized.png    (452.3 nm)
...
Band_230_653.4nm_normalized.png    (653.4 nm - Red)
...
Band_458_853.6nm_normalized.png    (853.6 nm - Near-IR)
```

**Only clean (non-noisy) bands exported!**

## Reports Generated

### 1. quality_report.txt
```
WAVELENGTH INFORMATION (from header.json):
  Total bands: 458
  Wavelength range: 450.51 - 853.58 nm
  Average step: 0.88 nm

SUMMARY:
  Clean bands: 380/458 (83.0%)
  Noisy bands: 78/458 (17.0%)

CLEAN BANDS (EXPORTED)
Band   Wavelength      SNR        Variance     Filename
1      450.51 nm       15.32      1234.56      Band_001_450.5nm_normalized.png
...

WAVELENGTH RANGE OF CLEAN BANDS:
  Min wavelength: 450.51 nm
  Max wavelength: 853.58 nm
  Coverage: 403.07 nm
```

### 2. clean_bands_list.txt
```
# Clean Band Indices and Wavelengths
# Format: Band_Index, Wavelength (nm)
# Data source: header.json

1, 450.51
2, 451.39
3, 452.27
...

# Python list (0-based indices):
clean_band_indices = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    ...
]

# Wavelengths (nm):
clean_wavelengths = [
    450.51, 451.39, 452.27, 453.15, 454.04,
    ...
]
```

### 3. clean_vs_noisy_comparison.png
Visual comparison showing clean vs noisy bands

## Spectral Range Coverage

**Your dataset covers:**

| Range | Wavelength | Color/Type | Applications |
|-------|-----------|------------|--------------|
| 450-495 nm | Blue | Visible | Material identification |
| 495-570 nm | Green | Visible | Color analysis |
| 570-590 nm | Yellow | Visible | Plastic type detection |
| 590-620 nm | Orange | Visible | Surface properties |
| 620-750 nm | Red | Visible | Reflectance analysis |
| 750-850 nm | Near-IR | Near-infrared | Molecular fingerprinting |

## Noise Filtering

**Automatic quality checks:**
- ✓ SNR ≥ 5.0 (signal quality)
- ✓ Variance ≥ 10.0 (information content)
- ✓ Saturation < 10% (not overexposed)
- ✓ Signal > 10 (adequate intensity)

**Expected results:**
- ~380-400 clean bands (83-87%)
- ~60-80 noisy bands filtered out

## Verification

```bash
# 1. Count exported files
ls clean_normalized_bands/Band_*_normalized.png | wc -l

# 2. Check quality report
cat clean_normalized_bands/quality_report.txt

# 3. View comparison
open clean_normalized_bands/clean_vs_noisy_comparison.png

# 4. Verify wavelength naming
ls clean_normalized_bands/ | head -10
# Should show: Band_001_450.5nm_normalized.png, etc.
```

## Usage in Training

The exported bands use **actual wavelengths from header.json**, not calculated estimates!

```python
# Load clean bands with actual wavelengths
from dataset_clean_bands import create_clean_dataloaders

train_loader, val_loader = create_clean_dataloaders(
    train_dir='training_dataset',
    label_path='Ground_Truth/labels.json',
    clean_bands_list='clean_normalized_bands/clean_bands_list.txt'
)

# Get wavelength info
with open('clean_normalized_bands/clean_bands_list.txt') as f:
    for line in f:
        if 'clean_wavelengths' in line:
            # Parse wavelength list
            pass
```

## Adjusting Thresholds

Edit `export_bands_with_header.py` to change filtering:

```python
export_clean_bands_with_header(
    input_dir='training_dataset',
    output_dir='clean_normalized_bands',
    snr_threshold=5.0,        # Adjust: 3.0 (lenient) to 10.0 (strict)
    variance_threshold=10.0,  # Adjust: 5.0 (lenient) to 50.0 (strict)
)
```

## Key Differences from Previous Version

| Feature | Previous (calculated) | New (from header.json) |
|---------|----------------------|------------------------|
| Wavelength source | Calculated (900-1700nm) | **Actual from header.json** ✓ |
| Range | 900-1700 nm | **450.5-853.6 nm** ✓ |
| Accuracy | Estimated | **Precise (from sensor)** ✓ |
| Spectrum type | NIR | **Visible + Near-IR** ✓ |

## Complete Workflow

```bash
# 1. Export with actual wavelengths
python export_bands_with_header.py

# 2. Verify export
ls clean_normalized_bands/*.png | wc -l
cat clean_normalized_bands/quality_report.txt

# 3. Check wavelength accuracy
head -20 clean_normalized_bands/clean_bands_list.txt

# 4. Visual inspection
open clean_normalized_bands/clean_vs_noisy_comparison.png

# 5. Train with clean bands
python train.py  # Using dataset_clean_bands.py
```

## Summary

✅ **Actual wavelengths from header.json** (not calculated)
✅ **Range: 450.5-853.6 nm** (Visible + Near-IR)
✅ **Precise naming:** Band_XXX_WWW.Wnm_normalized.png
✅ **Automatic noise filtering** (~83-87% clean)
✅ **Ready for training** with accurate spectral info

Your bands now have the **correct, precise wavelength information** directly from your sensor's calibration data! 🎯
