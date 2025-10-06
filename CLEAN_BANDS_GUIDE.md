# Clean Bands Export and Usage Guide

## Overview

This guide explains how to:
1. Export normalized bands with wavelength information
2. Filter out noisy bands automatically
3. Use only clean bands for training

## Step 1: Export with Wavelength and Noise Filtering

### Run the Export Script

```bash
python export_with_wavelength.py
```

### What Happens

1. **Loads all spectral bands** from `training_dataset/`
2. **Calculates wavelength** for each band (900-1700nm range)
3. **Analyzes quality** using SNR and variance metrics
4. **Filters noisy bands** based on thresholds
5. **Exports only clean bands** with wavelength in filename
6. **Generates reports** for verification

### Output Files

**Directory:** `clean_normalized_bands/`

**Clean band images:**
```
Band_001_900.0nm_normalized.png    (Band 1, 900.0 nm) - if clean
Band_002_901.7nm_normalized.png    (Band 2, 901.7 nm) - if clean
Band_003_903.5nm_normalized.png    (Band 3, 903.5 nm) - if clean
...
Band_459_1700.0nm_normalized.png   (Band 459, 1700.0 nm) - if clean
```

**Filename format:** `Band_XXX_WWWW.Wnm_normalized.png`
- `XXX` = Band number (001-459)
- `WWWW.W` = Wavelength in nanometers
- Only clean (non-noisy) bands are exported

**Reports:**
- `quality_report.txt` - Detailed quality analysis
- `clean_bands_list.txt` - List of clean band indices
- `clean_vs_noisy_comparison.png` - Visual comparison

## Step 2: Review Quality Report

```bash
cat clean_normalized_bands/quality_report.txt
```

### What to Check

**Summary section:**
```
Clean bands: 380/459 (82.8%)
Noisy bands: 79/459 (17.2%)
```

**Clean bands section:**
- Lists all clean bands with SNR, variance, and filename
- Higher SNR = better quality (typical: 5-50)
- Higher variance = more information (typical: >10)

**Noisy bands section:**
- Lists rejected bands with reasons:
  - "Low SNR" - Signal-to-noise ratio too low
  - "Low variance" - Band has almost no variation (dead)
  - "High saturation" - Too many saturated pixels
  - "Near-zero signal" - Almost all zeros

### Example Quality Report

```
CLEAN BANDS (EXPORTED)
Band   Wavelength   SNR        Variance     Filename
1      900.0 nm     15.32      1234.56      Band_001_900.0nm_normalized.png
2      901.7 nm     18.45      1567.89      Band_002_901.7nm_normalized.png
...

NOISY BANDS (SKIPPED)
Band   Wavelength   SNR        Variance     Issues
50     983.0 nm     2.14       5.43         Low SNR, Low variance
125    1450.5 nm    1.87       8.91         Low SNR
...
```

## Step 3: Use Clean Bands List

### View Clean Band Indices

```bash
cat clean_normalized_bands/clean_bands_list.txt
```

**Content format:**
```
# Clean Band Indices (1-based)
# Format: Band_Index, Wavelength (nm)

1, 900.0
2, 901.7
3, 903.5
...
455, 1693.2
459, 1700.0

# Total clean bands: 380

# Python list (0-based indices):
clean_band_indices = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, ...
]
```

## Step 4: Train with Clean Bands Only

### Method 1: Use Pre-built Dataset (Recommended)

```python
from dataset_clean_bands import create_clean_dataloaders

# This automatically loads only clean bands
train_loader, val_loader = create_clean_dataloaders(
    train_dir='training_dataset',
    label_path='Ground_Truth/labels.json',
    clean_bands_list='clean_normalized_bands/clean_bands_list.txt',
    batch_size=640
)

# Train model
from model import create_model

# Model will auto-adjust to number of clean bands
n_clean_bands = train_loader.dataset.dataset.n_bands  # e.g., 380
model = create_model(n_spectral_bands=n_clean_bands, n_classes=11)
```

### Method 2: Manual Integration

Update your training script:

```python
# Load clean band indices
def load_clean_indices(path='clean_normalized_bands/clean_bands_list.txt'):
    indices = []
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                idx = int(line.split(',')[0]) - 1  # Convert to 0-based
                indices.append(idx)
    return indices

# In your data loading code:
clean_indices = load_clean_indices()

# Load only clean bands
all_bands = load_all_bands('training_dataset')
clean_bands = all_bands[clean_indices]  # Filter

# Create model with clean band count
model = create_model(n_spectral_bands=len(clean_indices), n_classes=11)
```

## Noise Detection Criteria

The script filters bands based on:

### 1. Signal-to-Noise Ratio (SNR)
```python
SNR = mean_intensity / std_deviation
```
- **Threshold:** SNR ≥ 5.0 (default)
- **Clean:** SNR > 5.0
- **Noisy:** SNR < 5.0

### 2. Variance
```python
Variance = std_deviation²
```
- **Threshold:** Variance ≥ 10.0 (default)
- **Clean:** High variance (lots of information)
- **Noisy:** Low variance (dead/flat band)

### 3. Saturation
```python
Saturation = % pixels at max value
```
- **Threshold:** < 10% saturated
- **Clean:** < 10% saturated pixels
- **Noisy:** > 10% saturated (overexposed)

### 4. Signal Level
```python
Mean intensity check
```
- **Clean:** Mean > 10
- **Noisy:** Mean < 10 (near-zero signal)

## Adjusting Thresholds

Edit `export_with_wavelength.py` to customize:

```python
export_clean_bands(
    input_dir='training_dataset',
    output_dir='clean_normalized_bands',
    snr_threshold=5.0,        # ← Adjust: Higher = stricter
    variance_threshold=10.0,   # ← Adjust: Higher = stricter
    wl_min=900,               # Start wavelength (nm)
    wl_max=1700               # End wavelength (nm)
)
```

### Threshold Recommendations

| Use Case | SNR Threshold | Variance Threshold | Result |
|----------|---------------|-------------------|--------|
| **Maximum quality** | 10.0 | 50.0 | ~60% bands (very strict) |
| **High quality** | 7.0 | 20.0 | ~75% bands (strict) |
| **Balanced (default)** | 5.0 | 10.0 | ~80-85% bands |
| **Lenient** | 3.0 | 5.0 | ~90% bands |
| **Keep most** | 2.0 | 1.0 | ~95% bands |

## Wavelength Calculation

For 459 bands covering 900-1700nm:

```python
wavelength_step = (1700 - 900) / (459 - 1) = 1.75 nm

Band 1:   900.0 nm
Band 2:   901.7 nm
Band 3:   903.5 nm
...
Band 230: 1300.0 nm (middle)
...
Band 459: 1700.0 nm
```

**Formula:**
```
wavelength(band_idx) = 900 + (1700 - 900) × (band_idx / 458)
```

## Expected Results

### Typical Quality Distribution

For hyperspectral imaging (900-1700nm NIR range):

```
Total bands: 459

Clean bands: ~380-400 (83-87%)
  - Good SNR and variance
  - Suitable for training

Noisy bands: ~60-80 (13-17%)
  - Water absorption bands (~1400-1450nm, ~1900nm)
  - Edge artifacts (very low/high wavelengths)
  - Sensor noise regions
  - Low reflectance regions
```

### Common Noisy Wavelengths

- **~950-970nm:** Atmospheric water absorption
- **~1100-1150nm:** Low signal region
- **~1380-1450nm:** Strong water absorption
- **~1650-1700nm:** Edge of sensor range

## Visual Verification

### Check Comparison Image

```bash
open clean_normalized_bands/clean_vs_noisy_comparison.png
```

**What to look for:**
- **Clean bands:** Clear patterns, good contrast, visible objects
- **Noisy bands:** Grainy, low contrast, mostly noise

### Sample Individual Bands

```bash
# View a clean band
open clean_normalized_bands/Band_230_1300.0nm_normalized.png

# Check if noisy bands were correctly filtered
ls clean_normalized_bands/ | grep "1420.0nm"
# Should not exist if 1420nm is noisy (water absorption)
```

## Performance Impact

### Benefits of Using Clean Bands Only

1. **Better accuracy:** Remove noise that confuses the model
   - Expected improvement: 1-3% accuracy boost

2. **Faster training:** Fewer bands = less computation
   - 459 → 380 bands = ~17% speedup

3. **Better generalization:** Model focuses on informative bands
   - Less overfitting to noise

4. **Smaller model:** Fewer input features
   - 459 bands: ~1.2M parameters
   - 380 bands: ~1.0M parameters

### Trade-offs

- **Pros:** Better accuracy, faster training, less noise
- **Cons:** Lose ~17% of spectral information (but it was noisy)
- **Verdict:** Almost always worth it! ✓

## Complete Workflow

```bash
# 1. Export clean bands with wavelength info
python export_with_wavelength.py

# 2. Review quality report
cat clean_normalized_bands/quality_report.txt
open clean_normalized_bands/clean_vs_noisy_comparison.png

# 3. Check clean band count
wc -l < clean_normalized_bands/clean_bands_list.txt

# 4. Train with clean bands
python train.py  # Update to use dataset_clean_bands.py

# 5. Compare accuracy
# Before: Train with all 459 bands → ~98.0% accuracy
# After:  Train with clean bands only → ~98.5-99.0% accuracy ✓
```

## Troubleshooting

### Too Few Clean Bands (<70%)

**Cause:** Thresholds too strict

**Solution:** Lower thresholds:
```python
snr_threshold=3.0,         # Was 5.0
variance_threshold=5.0,    # Was 10.0
```

### Too Many Noisy Bands Kept (>90% kept)

**Cause:** Thresholds too lenient

**Solution:** Raise thresholds:
```python
snr_threshold=7.0,         # Was 5.0
variance_threshold=20.0,   # Was 10.0
```

### Wavelength Incorrect

**Cause:** Wrong wavelength range

**Solution:** Update range:
```python
wl_min=900,   # Check your sensor specs
wl_max=1700,  # Check your sensor specs
```

### Model Input Size Mismatch

**Error:** `Expected input size X, got Y`

**Solution:** Update model:
```python
# Count clean bands
n_clean = count_clean_bands('clean_normalized_bands/clean_bands_list.txt')

# Create model with correct size
model = create_model(n_spectral_bands=n_clean, n_classes=11)
```

## Summary

✓ **Export with wavelength:** `python export_with_wavelength.py`
✓ **Files named:** `Band_XXX_WWWW.Wnm_normalized.png`
✓ **Only clean bands exported** (typically 80-85% of total)
✓ **Quality report generated** with detailed analysis
✓ **Ready for training** with `dataset_clean_bands.py`
✓ **Expected improvement:** 1-3% accuracy boost + 17% faster training

Your data is now clean, properly named, and ready for optimal model performance! 🎯
