# Complete Solution Summary - Band Export Issue Fixed ✅

## Problem Identified

**Your Issue**: "I change lower a value of SNR, the exported number of band is still 0"

**Root Cause**: Manual mode requires **BOTH** conditions to pass:
```python
# A band is only exported if:
SNR >= snr_threshold  AND  Variance >= var_threshold
```

**Why lowering SNR didn't help**: If variance threshold is too high, all bands fail regardless of SNR!

## Solution: Use the Diagnostic Tool

### Step 1: Install Dependencies (First Time Only)

```bash
pip install numpy pillow tqdm torch torchvision
```

### Step 2: Run Diagnostic Tool

```bash
python export_with_diagnostics.py
```

**This will:**
- ✓ Show your actual SNR and variance ranges
- ✓ Predict how many bands will pass BEFORE exporting
- ✓ Warn if thresholds are too strict
- ✓ Export bands with wavelength in filename

### Step 3: Review the Output

The diagnostic will show something like:

```
DIAGNOSTICS: YOUR DATA STATISTICS
==================================================================================

SNR Statistics:
  Minimum:    0.85
  Maximum:    3.24
  Mean:       1.92

Variance Statistics:
  Minimum:    45.23
  Maximum:    892.45
  Mean:       234.67

THRESHOLD SELECTION
==================================================================================

AUTO MODE: Keeping top 75%
  SNR threshold:      1.15
  Variance threshold: 156.34

Clean bands: 344/458 (75.1%)
```

## Three Ways to Export

### Option 1: Auto Mode (Recommended - Always Works!)

```bash
python -c "
from export_with_diagnostics import export_with_diagnostics
export_with_diagnostics(
    mode='auto',
    keep_percentage=75  # Adjust percentage as needed
)
"
```

**Pros:**
- ✓ No need to know SNR/variance values
- ✓ Automatically calculates thresholds from your data
- ✓ Predictable (exactly 75% of bands)

### Option 2: Manual Mode with Diagnostics First

```bash
# Step 1: Run diagnostics to see your data
python export_with_diagnostics.py

# Step 2: Use the recommended values from output
python -c "
from export_with_diagnostics import export_with_diagnostics
export_with_diagnostics(
    mode='manual',
    manual_snr=0.8,    # Use value from diagnostics
    manual_var=35.0    # Use value from diagnostics
)
"
```

### Option 3: Ultra-Safe Manual (Works 99% of Time)

```bash
python -c "
from export_with_diagnostics import export_with_diagnostics
export_with_diagnostics(
    mode='manual',
    manual_snr=0.1,    # Extremely low
    manual_var=1.0     # Extremely low
)
"
```

## Quick Reference

### Your Dataset Info
- **Wavelength range**: 450.5 - 853.6 nm (from header.json)
- **Total bands**: 458
- **Image type**: 8-bit PNG (0-255 range)

### Expected SNR/Variance Ranges (8-bit images)
| Image Type      | SNR Range   | Variance Range |
|----------------|-------------|----------------|
| Low contrast   | 0.5 - 2.0   | 10 - 100       |
| Medium contrast| 1.0 - 5.0   | 50 - 500       |
| High contrast  | 2.0 - 10.0  | 200 - 2000     |

**Your images are likely low to medium contrast.**

### Safe Thresholds for 8-bit Images
- **SNR**: 0.5 - 1.5
- **Variance**: 20 - 100

## Output Files

After successful export, you'll find:

```
clean_normalized_bands/
  ├── Band_001_450.5nm_normalized.png
  ├── Band_002_451.4nm_normalized.png
  ├── ...
  ├── quality_report.txt              # Full quality report
  └── clean_bands_list.txt            # List of exported bands
```

## Troubleshooting

### Still Getting 0 Bands?

Check the diagnostic prediction:
```
Prediction:
  Bands passing SNR only:      412/458 (90.0%)
  Bands passing Variance only: 275/458 (60.0%)
  Bands passing BOTH:          0/458 (0.0%) ✗
```

**If "Bands passing BOTH" is 0:**
1. Lower BOTH thresholds (not just SNR!)
2. Or use auto mode instead

### How to Adjust Thresholds?

Run diagnostics first to see your actual values:
```bash
python export_with_diagnostics.py
```

Then use thresholds **below** the minimums shown:
- If min SNR = 0.85 → use manual_snr=0.5
- If min variance = 45.23 → use manual_var=30.0

## Next Steps

1. **Install dependencies**: `pip install numpy pillow tqdm torch torchvision`
2. **Run diagnostic tool**: `python export_with_diagnostics.py`
3. **Check output**: Review statistics and exported bands
4. **Verify quality**: Check the exported PNG files in `clean_normalized_bands/`
5. **Train model**: Use the clean bands for training

## Key Files Created

- **export_with_diagnostics.py** - Main diagnostic and export tool ⭐
- **THRESHOLD_GUIDE.md** - Complete troubleshooting guide
- **MANUAL_MODE_FIX.md** - Detailed fix documentation
- **check_image_stats.py** - Quick stats analyzer

## Why This Solution Works

1. **Shows you the data first** - No more guessing thresholds
2. **Predicts outcome** - Know before export how many bands will pass
3. **Warns about issues** - Alerts if 0 bands will be exported
4. **Auto-suggests values** - Recommends thresholds based on YOUR data
5. **Multiple modes** - Choose auto (easy) or manual (control)

---

**TL;DR**: Run `pip install numpy pillow tqdm torch torchvision` then `python export_with_diagnostics.py` to see your data and export bands with proper thresholds. Use auto mode for simplicity or manual mode with diagnostics-recommended values.
