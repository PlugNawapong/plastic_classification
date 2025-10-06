# Manual Mode Issue & Fix

## The Problem

**Why changing SNR gives 0 bands:**

In manual mode, a band must pass **ALL** conditions to be clean:

```python
# Line 129-134 in export_bands_fixed.py
if metric['snr'] < snr_threshold:
    is_clean = False              # ← Fails here

if metric['variance'] < var_threshold:
    is_clean = False              # ← OR fails here!
```

**Result:** Even if you lower SNR, if **variance threshold is too high**, all bands still fail!

## The Fix: Check Your Actual Data First

### Step 1: Analyze Your Images

```bash
python check_image_stats.py
```

This will show you the **actual SNR and variance** values in your images, like:

```
SNR Statistics:
  Min:    0.85
  Max:    3.45
  Mean:   1.92

Variance Statistics:
  Min:    45.23
  Max:    892.45
  Mean:   234.67
```

### Step 2: Use Appropriate Thresholds

**If your actual SNR range is 0.85-3.45:**

```bash
# WRONG - will give 0 bands!
manual_snr=5.0    # ← Higher than max SNR (3.45)
manual_var=500.0  # ← Higher than max variance

# CORRECT - will work!
manual_snr=0.5    # ← Below min SNR (0.85)
manual_var=30.0   # ← Below min variance (45.23)
```

## Quick Diagnostic Commands

### Check What Values You Have

```bash
# See actual SNR/variance in your images
python check_image_stats.py

# Output will show recommended thresholds:
# VERY LENIENT: SNR=0.43, Variance=22.6
# LENIENT:      SNR=0.68, Variance=36.2
# MODERATE:     SNR=1.15, Variance=140.8
```

### Export with Data-Driven Thresholds

```bash
# After running check_image_stats.py, use its recommendations:

# Very lenient (keeps ~95%)
python -c "
from export_bands_fixed import export_clean_bands
export_clean_bands(
    use_manual=True,
    manual_snr=0.5,     # Use value from check_image_stats.py
    manual_var=25.0,    # Use value from check_image_stats.py
    output_dir='clean_bands'
)
"
```

## Common Mistakes

### Mistake 1: Using 16-bit Thresholds on 8-bit Data

```bash
# WRONG for 8-bit images:
manual_snr=5.0      # Too high!
manual_var=500.0    # Too high!

# CORRECT for 8-bit images:
manual_snr=1.0      # Realistic
manual_var=50.0     # Realistic
```

### Mistake 2: Copying Values from Different Datasets

```bash
# Don't use thresholds from papers/examples
# Each dataset is different!

# Instead, analyze YOUR data:
python check_image_stats.py
# Then use the recommended values
```

### Mistake 3: Lowering Only SNR

```bash
# WRONG - variance is still too high:
manual_snr=0.1      # Very low ✓
manual_var=1000.0   # Still too high! ✗

# CORRECT - lower BOTH:
manual_snr=0.1      # Very low ✓
manual_var=10.0     # Also low ✓
```

## Complete Workflow

### Step 1: Analyze Your Data
```bash
python check_image_stats.py
```

**Example output:**
```
RECOMMENDED THRESHOLDS

VERY LENIENT (keep ~95% of bands):
  SNR:      0.42
  Variance: 22.61

LENIENT (keep ~85-90% of bands):
  SNR:      0.68
  Variance: 36.18
```

### Step 2: Export with Correct Thresholds
```bash
# Copy values from Step 1
python -c "
from export_bands_fixed import export_clean_bands
export_clean_bands(
    use_manual=True,
    manual_snr=0.68,     # ← From check_image_stats.py
    manual_var=36.18,    # ← From check_image_stats.py
    output_dir='clean_bands'
)
"
```

### Step 3: Verify
```bash
ls clean_bands/Band_*.png | wc -l
# Should show: ~390-410 bands (85-90%)
```

## Alternative: Use Percentile Mode (Easier!)

Instead of guessing thresholds, use percentile mode:

```bash
# Just specify what % to keep - thresholds auto-calculated!
python -c "
from export_bands_fixed import export_clean_bands
export_clean_bands(
    keep_percentage=85,    # Keep top 85%
    use_manual=False,      # Auto mode
    output_dir='clean_bands'
)
"
```

**Benefits:**
- No need to know SNR/variance values
- Automatically adapts to your data
- Predictable output (exactly 85% of bands)

## Summary

**The manual mode issue:**
1. ✗ Both SNR AND variance thresholds must pass
2. ✗ If either is too high, you get 0 bands
3. ✗ 8-bit images have lower SNR/variance than 16-bit

**The fix:**
1. ✓ Run `python check_image_stats.py` first
2. ✓ Use the recommended thresholds from the output
3. ✓ Lower BOTH SNR and variance thresholds
4. ✓ Or just use percentile mode (easier!)

**Quick command to always work:**
```bash
# Very lenient - almost always works
python -c "
from export_bands_fixed import export_clean_bands
export_clean_bands(
    use_manual=True,
    manual_snr=0.1,    # Very low
    manual_var=1.0,    # Very low
    output_dir='clean_bands'
)
"
```

This will keep ~95-99% of bands! 🎯
