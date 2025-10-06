# Complete Threshold Guide - Why Manual Mode Gives 0 Bands

## The Root Cause

**Manual mode requires BOTH conditions to pass:**

```python
# A band is only clean if:
SNR >= snr_threshold  AND  Variance >= var_threshold
```

**If either threshold is too high → 0 bands exported!**

## The Solution: Use Diagnostics First

### Best Approach: Run with Diagnostics

```bash
python export_with_diagnostics.py
```

**This will:**
1. ✓ Load your images
2. ✓ Show actual SNR and variance ranges
3. ✓ Predict how many bands will pass
4. ✓ Warn you if thresholds are too strict
5. ✓ Export only if bands will pass

### Example Output

```
DIAGNOSTICS: YOUR DATA STATISTICS
==================================================================================

SNR Statistics:
  Minimum:    0.85
  Maximum:    3.24
  Mean:       1.92
  25th %ile:  1.45
  75th %ile:  2.35

Variance Statistics:
  Minimum:    45.23
  Maximum:    892.45
  Mean:       234.67
  25th %ile:  156.34
  75th %ile:  312.89

THRESHOLD SELECTION
==================================================================================

MANUAL MODE: User-specified thresholds
  SNR threshold:      2.0
  Variance threshold: 50.0

  Prediction:
    Bands passing SNR only:      275/458 (60.0%)
    Bands passing Variance only: 412/458 (90.0%)
    Bands passing BOTH:          245/458 (53.5%) ✓  ← This is what you'll get!
```

## Three Ways to Export

### Method 1: Auto Mode (Easiest - Always Works!)

```bash
python -c "
from export_with_diagnostics import export_with_diagnostics
export_with_diagnostics(
    mode='auto',
    keep_percentage=75    # Just specify percentage
)
"
```

**Pros:**
- ✓ Always works
- ✓ No need to know SNR/variance values
- ✓ Predictable (exactly 75% of bands)

### Method 2: Auto-Suggested Manual (Safe)

```bash
# Let the script suggest thresholds based on your data
python -c "
from export_with_diagnostics import export_with_diagnostics
export_with_diagnostics(
    mode='manual'    # Thresholds auto-suggested from data
)
"
```

**Pros:**
- ✓ Automatically uses your data's min values * 0.8
- ✓ Almost always keeps 85-90% of bands
- ✓ Safe default

### Method 3: Full Manual (Advanced)

```bash
# Specify exact thresholds (see diagnostics first!)
python -c "
from export_with_diagnostics import export_with_diagnostics
export_with_diagnostics(
    mode='manual',
    manual_snr=1.0,      # Based on diagnostics
    manual_var=50.0      # Based on diagnostics
)
"
```

**Important:** The diagnostics will **predict** how many bands will pass BEFORE exporting!

## Common Problems & Fixes

### Problem 1: "0 bands exported"

**Cause:** Thresholds too high

**Diagnostic output:**
```
Prediction:
  Bands passing BOTH: 0/458 (0.0%) ✓

⚠ WARNING: With these thresholds, 0 bands will be exported!
⚠ Try lower values:
   SNR:      0.42 (below minimum)
   Variance: 22.6 (below minimum)
```

**Fix:**
```bash
python -c "
from export_with_diagnostics import export_with_diagnostics
export_with_diagnostics(
    mode='manual',
    manual_snr=0.42,    # Use suggested value
    manual_var=22.6     # Use suggested value
)
"
```

### Problem 2: "Changing SNR doesn't help"

**Cause:** Variance threshold is the bottleneck

**Example:**
```
Your settings:
  SNR: 0.1 (very low)    ← 450 bands pass this
  Var: 1000 (very high)  ← Only 5 bands pass this

Result: Only 5 bands pass BOTH ← Variance is the problem!
```

**Fix:** Lower variance too!
```bash
manual_snr=0.1,     # Keep low
manual_var=10.0     # Lower this too!
```

### Problem 3: "How do I know what values to use?"

**Answer:** Run diagnostics first!

```bash
# Step 1: See your data
python export_with_diagnostics.py

# Step 2: Look at the output
# SNR min: 0.85, max: 3.24
# Var min: 45.23, max: 892.45

# Step 3: Choose thresholds BELOW the minimums
manual_snr=0.5      # Below 0.85 ✓
manual_var=30.0     # Below 45.23 ✓
```

## Quick Reference: Typical Values

### 8-bit Images (0-255)

| Image Type | SNR Range | Variance Range |
|-----------|-----------|----------------|
| Low contrast | 0.5 - 2.0 | 10 - 100 |
| Medium contrast | 1.0 - 5.0 | 50 - 500 |
| High contrast | 2.0 - 10.0 | 200 - 2000 |

**Your dataset is likely:** Low to medium contrast (8-bit)

**Safe thresholds for 8-bit:**
- SNR: 0.5 - 1.5
- Variance: 20 - 100

### 16-bit Images (0-65535)

| Image Type | SNR Range | Variance Range |
|-----------|-----------|----------------|
| Low contrast | 2.0 - 10.0 | 500 - 5000 |
| Medium contrast | 5.0 - 20.0 | 2000 - 20000 |
| High contrast | 10.0 - 50.0 | 10000 - 100000 |

## Decision Tree

```
Do you know your SNR/variance values?
│
├─ NO → Use export_with_diagnostics.py (shows you values)
│       Then choose thresholds below minimums
│
└─ YES → Do you want exact control?
         │
         ├─ NO → Use auto mode (keep_percentage=75)
         │
         └─ YES → Use manual mode with values BELOW minimums:
                  manual_snr < min_snr
                  manual_var < min_variance
```

## Example Commands (Copy & Paste)

### Ultra-Safe (Works 99% of time)

```bash
python -c "
from export_with_diagnostics import export_with_diagnostics
export_with_diagnostics(
    mode='manual',
    manual_snr=0.1,     # Extremely low
    manual_var=1.0      # Extremely low
)
"
```

### Recommended (Good balance)

```bash
python -c "
from export_with_diagnostics import export_with_diagnostics
export_with_diagnostics(
    mode='auto',
    keep_percentage=75  # Keep top 75%
)
"
```

### Custom (After checking diagnostics)

```bash
# First run to see values:
python export_with_diagnostics.py

# Then use values from output:
python -c "
from export_with_diagnostics import export_with_diagnostics
export_with_diagnostics(
    mode='manual',
    manual_snr=0.8,     # From diagnostics: min * 0.8
    manual_var=35.0     # From diagnostics: min * 0.8
)
"
```

## Summary

**Why manual mode gives 0 bands:**
1. ✗ BOTH SNR and variance must pass
2. ✗ If either threshold is too high → 0 bands
3. ✗ 8-bit images have lower values than expected

**The fix:**
1. ✓ Run `export_with_diagnostics.py` first
2. ✓ See actual SNR/variance ranges in YOUR data
3. ✓ Use thresholds BELOW the minimums
4. ✓ Or just use auto mode!

**Easiest solution:**
```bash
python -c "
from export_with_diagnostics import export_with_diagnostics
export_with_diagnostics(mode='auto', keep_percentage=75)
"
```

Done! 🎯
